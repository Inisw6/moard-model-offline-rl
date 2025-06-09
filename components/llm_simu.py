import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any

import requests

from components.personas import PersonaConfig, create_persona_from_user_data


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
        "youtube": (60, 600),  # 1-10분
        "blog": (90, 480),  # 1.5-8분
        "news": (30, 300),  # 30초-5분
        "default": (30, 300),
    }

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.2:2b",
        debug: bool = False,
    ) -> None:
        """
        Ollama LLM API 클라이언트를 초기화합니다.

        Args:
            ollama_url (str): Ollama 서버의 베이스 URL. 기본값은 "http://localhost:11434".
            model (str): 사용할 LLM 모델 이름. 예: "llama3.2:2b".
            debug (bool): 디버그 로그 출력 여부. True일 경우 로그가 출력됩니다.
        """
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.debug = debug

        # 연결 상태 캐싱
        self._connection_checked = False
        self._is_available = False

        # API 설정
        self._api_config = {
            "timeout": 30,
            "options": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 1000},
        }

    @property
    def is_available(self) -> bool:
        """
        Ollama 서버가 현재 사용 가능한지 확인합니다.

        Returns:
            bool: 서버 연결 가능 여부. 최초 호출 시 한 번만 실제 연결을 테스트하고 캐시합니다.
        """
        if not self._connection_checked:
            self._is_available = self._test_ollama_connection()
            self._connection_checked = True
        return self._is_available

    @lru_cache(maxsize=1)
    def _test_ollama_connection(self) -> bool:
        """
        Ollama 서버에 연결 가능 여부를 테스트합니다.

        서버 상태 확인을 위해 `/api/tags` 엔드포인트에 GET 요청을 보냅니다.
        요청이 성공하고 지정된 모델이 서버에 존재하면 True를 반환합니다.

        Returns:
            bool: 서버에 성공적으로 연결되었고 모델이 존재하면 True, 아니면 False
        """
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                logging.warning(f"Ollama 서버 응답 오류: {response.status_code}")
                return False

            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]

            if self.model not in model_names:
                logging.warning(
                    "모델 '%s'을(를) 찾을 수 없습니다. 사용 가능한 모델: %s",
                    self.model,
                    model_names,
                )
                return False

            logging.info("Ollama 서버 연결 성공. 모델 '%s' 사용 가능.", self.model)
            return True

        except requests.RequestException as e:
            logging.info("Ollama 서버 연결 성공. 모델 '%s' 사용 가능.", self.model)
            return False

    def simulate_user_response(
        self,
        persona_id: int,
        mbti: str,
        investment_level: int,
        recommended_contents: List[Dict],
        current_context: Optional[Dict] = None,
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
            user_id=persona_id, mbti=mbti, investment_level=investment_level
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
        self, persona: PersonaConfig, recommended_contents: List[Dict]
    ) -> str:
        """
        Ollama를 활용한 사용자 반응 시뮬레이션을 수행합니다.

        주어진 페르소나와 추천 콘텐츠 리스트를 기반으로 LLM에게 사용자 응답을 생성하도록 요청합니다.
        시뮬레이션 결과로 원본 텍스트(JSON 포맷 예상)를 반환합니다.

        Args:
            persona (PersonaConfig): 사용자 페르소나 정보.
            recommended_contents (List[Dict]): 추천된 콘텐츠 리스트.

        Returns:
            str: LLM으로부터 받은 원본 응답 텍스트.
        """

        # 콘텐츠 정보 준비
        contents_info, content_ids = self._prepare_content_info(recommended_contents)

        # 프롬프트 생성
        user_prompt = self._build_user_prompt(persona, contents_info, content_ids)

        # 디버깅 출력
        if self.debug:
            logging.debug("LLM에게 보내는 프롬프트:\n%s", user_prompt)

        # API 호출
        response = self._call_ollama_api(user_prompt)

        # 원본 응답 텍스트 반환
        llm_output = response.get("response", "")

        if self.debug:
            logging.debug("LLM 원본 응답:\n%s\n", llm_output)

        return llm_output

    def _prepare_content_info(
        self, recommended_contents: List[Dict]
    ) -> Tuple[List[ContentInfo], List[str]]:
        """
        추천 콘텐츠 리스트로부터 콘텐츠 정보와 ID 목록을 생성합니다.

        각 콘텐츠 항목에서 필요한 정보를 추출하여 `ContentInfo` 객체를 생성하고,
        동시에 콘텐츠 ID 리스트를 수집합니다. 콘텐츠 정보는 안전한 기본값으로 보완됩니다.

        Args:
            recommended_contents (List[Dict]): 추천된 콘텐츠들의 딕셔너리 리스트.

        Returns:
            Tuple[List[ContentInfo], List[str]]:
                - 콘텐츠 정보 리스트 (`ContentInfo` 객체들).
                - 콘텐츠 ID 리스트 (str 타입).
        """
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
                description=content.get("description", "설명 없음")[:200],
            )
            contents_info.append(info)

        return contents_info, content_ids

    # 프롬포트엔지니어링 하는 부분
    def _build_user_prompt(
        self,
        persona: PersonaConfig,
        contents_info: List[ContentInfo],
        content_ids: List[str],
    ) -> str:
        """
        LLM에게 단 하나의 JSON 배열만 출력하도록 프롬프트를 구성합니다.

        프롬프트는 페르소나 정보 및 콘텐츠 설명을 기반으로 하며, LLM의 출력 결과가
        아래 조건을 반드시 만족하도록 유도합니다.

        Args:
            persona (PersonaConfig): 사용자 페르소나 정보.
            contents_info (List[ContentInfo]): 콘텐츠 설명 목록.
            content_ids (List[str]): 추천된 콘텐츠 ID 목록.

        Returns:
            str: LLM에게 전달할 프롬프트 문자열.
        """

        # 1) 콘텐츠 설명 줄
        content_info_text = "\n".join(
            f"{info.content_id}: {info.type} - {info.title}\n   설명: {info.description}"
            for info in contents_info
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

    def _call_ollama_api(self, prompt: str) -> Dict[str, Any]:
        """
        Ollama API를 호출하여 사용자 프롬프트에 대한 LLM 응답을 가져옵니다.

        Args:
            prompt (str): LLM에게 전달할 사용자 프롬프트.

        Returns:
            Dict[str, Any]: Ollama의 응답 JSON 객체.

        Raises:
            RuntimeError: 응답 상태 코드가 200이 아닐 경우 예외 발생.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": self._api_config["options"],
        }

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=self._api_config["timeout"],
        )

        if response.status_code != 200:
            raise Exception(
                f"Ollama API 오류: {response.status_code} - {response.text}"
            )

        return response.json()

    def reset_connection_cache(self) -> None:
        """
        Ollama 연결 캐시를 초기화합니다.

        이 메서드는 연결 테스트 결과를 초기화하며, 서버 변경,
        재시도 또는 테스트 시 유용하게 사용됩니다.
        """
        self._connection_checked = False
        self._is_available = False
        self._test_ollama_connection.cache_clear()
