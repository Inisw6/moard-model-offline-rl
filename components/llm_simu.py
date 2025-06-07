import json
import random
import requests
from typing import Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from functools import lru_cache
import logging

from .personas import PersonaConfig, create_persona_from_user_data

@dataclass
class ContentInfo:
    """콘텐츠 정보를 위한 데이터 클래스"""
    index: int
    content_id: str
    type: str
    title: str
    url: str
    description: str

class LLMUserSimulator:
    """
    Ollama를 활용한 사용자 시뮬레이터.
    페르소나 DB에서 가져온 MBTI와 투자 레벨로 PersonaConfig를 생성하여 
    콘텐츠 추천에 대한 반응을 시뮬레이션합니다.
    """
    
    # 콘텐츠 타입별 기본 체류시간 범위 (초)
    CONTENT_DWELL_TIMES = {
        "youtube": (60, 600),    # 1-10분
        "blog": (90, 480),       # 1.5-8분
        "news": (30, 300),       # 30초-5분
        "default": (30, 300)
    }
    
    def __init__(
        self, 
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        debug: bool = False
    ):
        """
        Args:
            ollama_url (str): Ollama 서버 URL
            model (str): 사용할 모델명
            debug (bool): 디버깅 출력 여부
        """
        self.ollama_url = ollama_url.rstrip('/')  # 후행 슬래시 제거
        self.model = model
        self.debug = debug
        
        # 연결 상태 캐싱
        self._connection_checked = False
        self._is_available = False
        
        # API 설정
        self._api_config = {
            "timeout": 30,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000
            }
        }
    
    @property
    def is_available(self) -> bool:
        """Ollama 서버 사용 가능 여부 (지연 초기화)"""
        if not self._connection_checked:
            self._is_available = self._test_ollama_connection()
            self._connection_checked = True
        return self._is_available
    
    @lru_cache(maxsize=1)
    def _test_ollama_connection(self) -> bool:
        """Ollama 서버 연결 테스트 (캐싱됨)"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                logging.warning(f"Ollama 서버 응답 오류: {response.status_code}")
                return False
                
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            
            if self.model not in model_names:
                logging.warning(f"모델 {self.model}을 찾을 수 없습니다. 사용 가능한 모델: {model_names}")
                return False
                
            logging.info(f"Ollama 서버 연결 성공. 모델 {self.model} 사용 가능.")
            return True
            
        except requests.RequestException as e:
            logging.warning(f"Ollama 서버 연결 실패: {e}")
            return False
    
    def simulate_user_response(
        self, 
        persona_id: int,
        mbti: str,
        investment_level: int,
        recommended_contents: List[Dict],
        current_context: Optional[Dict] = None
    ) -> str:
        """
        페르소나 정보를 기반으로 사용자 반응 시뮬레이션.
        
        Args:
            persona_id: 페르소나 ID
            mbti: MBTI 유형
            investment_level: 투자 레벨 (1=초보, 2=중급, 3=고급)
            recommended_contents: 추천된 콘텐츠 리스트
            current_context: 현재 컨텍스트 정보 (옵션)
            
        Returns:
            str: LLM 원본 응답 텍스트
        """
        # 페르소나 생성
        persona = create_persona_from_user_data(
            user_id=persona_id,
            mbti=mbti,
            investment_level=investment_level
        )

        if not recommended_contents:
            return ""
        
        # Ollama 연결 확인
        if not self.is_available:
            raise RuntimeError("Ollama 서버를 사용할 수 없습니다.")
        
        # LLM 기반 시뮬레이션 실행
        return self._ollama_based_simulation(
            persona, recommended_contents, current_context
        )
    
    def _ollama_based_simulation(
        self, 
        persona: PersonaConfig, 
        recommended_contents: List[Dict],
        current_context: Optional[Dict] = None
    ) -> str:
        """Ollama를 활용한 사용자 반응 시뮬레이션 (원본 텍스트 반환)"""
        
        # 콘텐츠 정보 준비
        contents_info, content_ids = self._prepare_content_info(recommended_contents)
        
        # 프롬프트 생성
        user_prompt = self._build_user_prompt(persona, contents_info, content_ids)
        
        # 디버깅 출력
        if self.debug:
            print("📝 LLM에게 보내는 프롬프트:")
            print("-" * 40)
            print(user_prompt)
            print("-" * 40)
            print()
        
        # API 호출
        response = self._call_ollama_api(user_prompt)
        
        # 원본 응답 텍스트 반환
        llm_output = response.get("response", "")
        
        if self.debug:
            print("🤖 LLM 원본 응답:")
            print("-" * 40)
            print(llm_output)
            print("-" * 40)
            print()
        
        return llm_output
    
    def _prepare_content_info(self, recommended_contents: List[Dict]) -> tuple[List[ContentInfo], List[str]]:
        """콘텐츠 정보를 효율적으로 준비"""
        contents_info = []
        content_ids = []
        
        for i, content in enumerate(recommended_contents):
            content_id = content.get("id", f"content_{i}")
            content_ids.append(content_id)
            
            # ContentInfo 객체 생성 (메모리 효율적)
            info = ContentInfo(
                index=i,
                content_id=content_id,
                type=content.get("type", "unknown"),
                title=content.get("title", "제목 없음"),
                url=content.get("url", ""),
                description=content.get("description", "설명 없음")[:200]
            )
            contents_info.append(info)
        
        return contents_info, content_ids
    
    ## 프롬포트엔지니어링 하는 부분
    def _build_user_prompt(
        self, 
        persona: PersonaConfig, 
        contents_info: List[ContentInfo], 
        content_ids: List[str],
    ) -> str:
        """
        LLM에게 **단 하나의 유효 JSON 객체**만 반환하도록 요구.
        - 외부 키: responses·timestamp·persona_id·simulation_method
        - 내부 배열 요소 수 == len(content_ids)
        """
        
        # 1) 콘텐츠 설명 줄
        content_info_text = "\n".join(
            f"{info.content_id}: {info.type} - {info.title}" for info in contents_info
        )
        
        # 2) 공통 메타값
        persona_id = f"{persona.mbti}_{persona.investment_level}_{persona.user_id}"
        
        # 3) 프롬프트 본문
        prompt = f"""
너는 주식 콘텐츠 클릭 시뮬레이터다. 입력 정보에 따른 페르소나를 기반으로 행동해라.

### 입력 정보
- persona_id: {persona_id}
- 투자등급(investment_level): {persona.investment_level}
- 위험 성향(risk_tolerance): {persona.risk_tolerance:.1f}
- 변동성 수용 정도(volatility_tolerance): {persona.volatility_tolerance:.1f}
- 배당 선호 정도(dividend_preference): {persona.dividend_preference:.1f}
- 결정 속도(decision_speed): {persona.decision_speed:.1f}
- 사회적 영향 민감도(social_influence): {persona.social_influence:.1f}
- 투자 기간(investment_horizon): {persona.investment_horizon.value}
- 분석 선호(analysis_preference): {persona.analysis_preference.value}
- 전문가 의존도(expert_reliance): {persona.expert_reliance:.1f}
- 채널 가중치: 유튜브 {persona.preferences['youtube']:.1f}, 블로그 {persona.preferences['blog']:.1f}, 뉴스 {persona.preferences['news']:.1f}

### 후보 콘텐츠
{content_info_text}

### 출력 형식 (**아래 JSON 배열을 그대로, 값만 채워서** 반환) — 다른 글자·공백·백틱·설명 금지
[
{chr(10).join(
    f'  {{"content_id": "{cid}", "clicked": true/false, "dwell_time_seconds": 0}}'
    + (',' if i < len(content_ids) - 1 else '')
    for i, cid in enumerate(content_ids)
)}
]

### 투자 레벨 정의
- investment_level은 1(초보·소액) ~ 5(전문·대규모) 사이 정수.

### 공통 비율 정의
- 모든 비율 값은 0.0 ~ 1.0 사이 실수.
- 값이 클수록 해당 속성이 차지하는 비중(선호·민감도·강도)이 크다.
  예) channel_weight 1.0 → 반드시 우선 고려, 0.0 → 전혀 고려 안 함.

### 하드 규칙
1. 배열 길이는 **{len(content_ids)}** 개.
2. clicked == false ➜ dwell_time_seconds == 0  
   clicked == true  ➜ dwell_time_seconds ∈ [30,300] (정수)
3. 키·따옴표·콤마·대소문자 일체 수정 금지.
4. JSON 배열 이외 텍스트·마크다운 블록·주석 **절대 출력 금지**.
"""
        return prompt.strip()

    def _call_ollama_api(self, prompt: str) -> Dict:
        """Ollama API 호출 (최적화됨)"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": self._api_config["options"]
        }
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=self._api_config["timeout"]
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API 오류: {response.status_code} - {response.text}")
        
        return response.json()
    
    def reset_connection_cache(self) -> None:
        """연결 캐시 리셋 (테스트 또는 재연결 시 사용)"""
        self._connection_checked = False
        self._is_available = False
        self._test_ollama_connection.cache_clear() 