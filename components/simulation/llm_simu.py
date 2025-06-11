import logging
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any, Union, Literal

import requests

from components.simulation.personas import PersonaConfig, create_persona_from_user_data


class LLMProvider(str, Enum):
    """LLM 제공자 열거형.

    Attributes:
        OLLAMA: Ollama 로컬 LLM 서버
        OPENAI: OpenAI API 서비스
        OPENROUTER: OpenRouter API 서비스
    """

    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENROUTER = "openrouter"


@dataclass
class ContentInfo:
    """콘텐츠 정보를 위한 데이터 클래스.

    Args:
        index: 콘텐츠의 순서 인덱스
        content_id: 콘텐츠 고유 식별자
        type: 콘텐츠 타입
        title: 콘텐츠 제목
        url: 콘텐츠 URL
        description: 콘텐츠 설명 (요약)

    Attributes:
        index: 콘텐츠의 순서 인덱스
        content_id: 콘텐츠 고유 식별자
        type: 콘텐츠 타입
        title: 콘텐츠 제목
        url: 콘텐츠 URL
        description: 콘텐츠 설명 (요약)
    """

    index: int
    content_id: str
    type: str
    title: str
    url: str
    description: str


class LLMUserSimulator:
    """LLM을 활용한 사용자 반응 시뮬레이션 클래스.

    페르소나 정보와 추천 콘텐츠를 기반으로 LLM에게 사용자 반응 예측을 요청하고,
    JSON 포맷의 응답을 받아옵니다.

    Args:
        provider: LLM 제공자 ("ollama", "openai", "openrouter")
        model: 사용할 모델 이름
        api_base: API 기본 URL
        api_key: API 키 (OpenAI, OpenRouter용)
        temperature: 생성 텍스트의 무작위성 정도 (0.0-1.0)
        top_p: 누적 확률 임계값 (0.0-1.0)
        max_tokens: 최대 생성 토큰 수
        timeout: API 요청 타임아웃 (초)
        debug: 디버그 로그 출력 여부

    Attributes:
        provider: LLM 제공자
        model: 사용할 모델 이름
        api_base: API 기본 URL
        api_key: API 키
        temperature: 생성 텍스트의 무작위성 정도
        top_p: 누적 확률 임계값
        max_tokens: 최대 생성 토큰 수
        timeout: API 요청 타임아웃
        debug: 디버그 로그 출력 여부
        _connection_checked: LLM 서버 연결 테스트 여부 플래그
        _is_available: LLM 서버 사용 가능 여부
    """

    # 콘텐츠 타입별 기본 체류시간 범위 (초)
    CONTENT_DWELL_TIMES: Dict[str, Tuple[int, int]] = {
        "youtube": (60, 600),  # 1-10분
        "blog": (90, 480),  # 1.5-8분
        "news": (30, 300),  # 30초-5분
        "default": (30, 300),
    }

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3.2:3b",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1000,
        timeout: int = 30,
        debug: bool = False,
    ) -> None:
        """LLM API 시뮬레이터 초기화.

        Args:
            provider: LLM 제공자 ("ollama", "openai", "openrouter")
            model: 사용할 모델 이름
            api_base: API 기본 URL
            api_key: API 키 (OpenAI, OpenRouter용)
            temperature: 생성 텍스트의 무작위성 정도 (0.0-1.0)
            top_p: 누적 확률 임계값 (0.0-1.0)
            max_tokens: 최대 생성 토큰 수
            timeout: API 요청 타임아웃 (초)
            debug: 디버그 로그 출력 여부
        """
        self.provider = LLMProvider(provider)
        self.model = model
        self.api_base = api_base or self._get_default_api_base()
        self.api_key = api_key
        self.temperature = max(0.0, min(1.0, temperature))
        self.top_p = max(0.0, min(1.0, top_p))
        self.max_tokens = max(1, max_tokens)
        self.timeout = timeout
        self.debug = debug

        # 연결 상태 캐싱
        self._connection_checked: bool = False
        self._is_available: bool = False

    def _get_default_api_base(self) -> str:
        """제공자별 기본 API URL을 반환합니다.

        Returns:
            str: 기본 API URL
        """
        if self.provider == LLMProvider.OLLAMA:
            return "http://localhost:11434"
        elif self.provider == LLMProvider.OPENAI:
            return "https://api.openai.com/v1"
        else:  # OpenRouter
            return "https://api.openrouter.ai/api/v1"

    @property
    def is_available(self) -> bool:
        """LLM 서버의 현재 사용 가능 여부를 반환합니다.

        Returns:
            bool: 서버 연결 및 모델 존재 여부
        """
        if not self._connection_checked:
            self._is_available = self._test_connection()
            self._connection_checked = True
        return self._is_available

    @lru_cache(maxsize=1)
    def _test_connection(self) -> bool:
        """LLM 서버 및 모델 연결 가능 여부를 테스트합니다.

        Returns:
            bool: 서버에 연결되고, 모델이 존재하면 True, 아니면 False

        Raises:
            RuntimeError: API 키가 필요한 서비스에서 API 키가 없는 경우
        """
        try:
            if self.provider == LLMProvider.OLLAMA:
                response = requests.get(f"{self.api_base}/api/tags", timeout=5)
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

            elif self.provider in [LLMProvider.OPENAI, LLMProvider.OPENROUTER]:
                if not self.api_key:
                    raise RuntimeError(f"{self.provider.value} API 키가 필요합니다.")

                headers = {"Authorization": f"Bearer {self.api_key}"}
                if self.provider == LLMProvider.OPENAI:
                    response = requests.get(
                        f"{self.api_base}/models", headers=headers, timeout=5
                    )
                else:  # OpenRouter
                    response = requests.get(
                        f"{self.api_base}/models", headers=headers, timeout=5
                    )

                if response.status_code != 200:
                    logging.warning(f"API 응답 오류: {response.status_code}")
                    return False

            logging.info("LLM 서버 연결 성공. 모델 '%s' 사용 가능.", self.model)
            return True

        except (requests.Timeout, requests.ConnectionError) as e:
            logging.error(f"LLM 서버 연결 실패: {str(e)}")
            return False

    def simulate_user_response(
        self,
        persona_id: int,
        mbti: str,
        investment_level: int,
        recommended_contents: List[Dict[str, Any]],
        current_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """페르소나 정보를 기반으로 사용자 반응을 LLM으로 시뮬레이션합니다.

        Args:
            persona_id: 페르소나 ID
            mbti: MBTI 유형
            investment_level: 투자 레벨 (1=초보, 2=중급, 3=고급)
            recommended_contents: 추천된 콘텐츠 리스트
            current_context: 추가 컨텍스트 정보

        Returns:
            str: LLM 원본 응답 텍스트(JSON)

        Raises:
            RuntimeError: LLM 서버를 사용할 수 없는 경우
        """
        # 페르소나 생성
        persona = create_persona_from_user_data(
            user_id=persona_id, mbti=mbti, investment_level=investment_level
        )

        if not recommended_contents:
            return ""

        # LLM 연결 확인
        if not self.is_available:
            raise RuntimeError("LLM 서버를 사용할 수 없습니다.")

        # LLM 기반 시뮬레이션 실행
        return self._llm_based_simulation(persona, recommended_contents)

    def _llm_based_simulation(
        self, persona: PersonaConfig, recommended_contents: List[Dict[str, Any]]
    ) -> str:
        """LLM을 활용해 실제 사용자 반응 시뮬레이션을 수행합니다.

        Args:
            persona: 사용자 페르소나 정보
            recommended_contents: 추천된 콘텐츠 리스트

        Returns:
            str: LLM으로부터 받은 원본 응답 텍스트(JSON)

        Raises:
            RuntimeError: API 호출 중 오류가 발생한 경우
        """
        # 콘텐츠 정보 준비
        contents_info, content_ids = self._prepare_content_info(recommended_contents)

        # 프롬프트 생성
        user_prompt = self._build_user_prompt(persona, contents_info, content_ids)

        # API 호출
        response = self._call_llm_api(user_prompt)

        # 원본 응답 텍스트 반환
        return response

    def _prepare_content_info(
        self, recommended_contents: List[Dict[str, Any]]
    ) -> Tuple[List[ContentInfo], List[str]]:
        """추천 콘텐츠 리스트로부터 ContentInfo 객체와 ID 리스트를 생성합니다.

        Args:
            recommended_contents: 추천 콘텐츠 딕셔너리 리스트

        Returns:
            Tuple[List[ContentInfo], List[str]]:
                ContentInfo 객체 리스트와 콘텐츠 ID 리스트의 튜플
        """
        contents_info = []
        content_ids = []

        for i, content in enumerate(recommended_contents):
            content_id = content.get("id", f"content_{i}")
            content_ids.append(content_id)

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

    def _build_user_prompt(
        self,
        persona: PersonaConfig,
        contents_info: List[ContentInfo],
        content_ids: List[str],
    ) -> str:
        """LLM에 입력할 프롬프트 문자열을 생성합니다.

        Args:
            persona: 사용자 페르소나 정보
            contents_info: 콘텐츠 정보 객체 리스트
            content_ids: 추천 콘텐츠 ID 리스트

        Returns:
            str: LLM 프롬프트 문자열
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
        ***** JSON만 출력하세요. 다른 텍스트는 절대 금지입니다. *****

        주식 콘텐츠 클릭 시뮬레이터입니다. 다음 정보를 기반으로 JSON 배열만 반환하세요.

        투자자: {persona_id}
        투자등급: {persona.investment_level}
        위험성향: {persona.risk_tolerance:.1f}
        결정속도: {persona.decision_speed:.1f}
        채널선호: 유튜브 {persona.preferences['youtube']:.1f}, 블로그 {persona.preferences['blog']:.1f}, 뉴스 {persona.preferences['news']:.1f}

        후보 콘텐츠:
        {content_info_text}

        ***** 아래 JSON 형식으로만 응답하세요. 설명·주석·마크다운 없이 순수 JSON만! *****

        [
        {chr(10).join(
            f'  {{"content_id": "{cid}", "clicked": true/false, "dwell_time_seconds": (int)}}'
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

    def _call_llm_api(self, prompt: str) -> str:
        """LLM API를 호출하여 사용자 프롬프트에 대한 응답을 가져옵니다.

        Args:
            prompt: LLM에게 전달할 사용자 프롬프트

        Returns:
            str: LLM의 응답 텍스트

        Raises:
            RuntimeError: API 호출 중 오류가 발생한 경우
        """
        try:
            if self.provider == LLMProvider.OLLAMA:
                return self._call_ollama_api(prompt)
            elif self.provider == LLMProvider.OPENAI:
                return self._call_openai_api(prompt)
            else:  # OpenRouter
                return self._call_openrouter_api(prompt)
        except Exception as e:
            raise RuntimeError(f"LLM API 호출 오류: {str(e)}")

    def _call_ollama_api(self, prompt: str) -> str:
        """Ollama API를 호출합니다.

        Args:
            prompt: 사용자 프롬프트

        Returns:
            str: Ollama의 응답 텍스트

        Raises:
            RuntimeError: API 호출 중 오류가 발생한 경우
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
            },
        }

        response = requests.post(
            f"{self.api_base}/api/generate",
            json=payload,
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama API 오류: {response.status_code} - {response.text}"
            )

        return response.json().get("response", "")

    def _call_openai_api(self, prompt: str) -> str:
        """OpenAI API를 호출합니다.

        Args:
            prompt: 사용자 프롬프트

        Returns:
            str: OpenAI의 응답 텍스트

        Raises:
            RuntimeError: API 호출 중 오류가 발생한 경우
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenAI API 오류: {response.status_code} - {response.text}"
            )

        return response.json()["choices"][0]["message"]["content"]

    def _call_openrouter_api(self, prompt: str) -> str:
        """OpenRouter API를 호출합니다.

        Args:
            prompt: 사용자 프롬프트

        Returns:
            str: OpenRouter의 응답 텍스트

        Raises:
            RuntimeError: API 호출 중 오류가 발생한 경우
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/your-repo",  # OpenRouter 요구사항
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter API 오류: {response.status_code} - {response.text}"
            )

        return response.json()["choices"][0]["message"]["content"]

    def reset_connection_cache(self) -> None:
        """LLM 연결 캐시를 초기화합니다.

        이 메서드는 연결 테스트 결과를 초기화하며, 서버 변경,
        재시도 또는 테스트 시 유용하게 사용됩니다.
        """
        self._connection_checked = False
        self._is_available = False
        self._test_connection.cache_clear()
