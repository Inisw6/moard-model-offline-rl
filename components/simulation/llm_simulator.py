import logging
from typing import Any, Dict, List, Optional

from components.core.base import BaseResponseSimulator
from components.simulation.random_simulator import RandomResponseSimulator
from components.simulation.llm_response_handler import LLMResponseHandler
from components.simulation.llm_simu import LLMUserSimulator
from components.database.persona_db import get_persona_db


class LLMResponseSimulator(BaseResponseSimulator):
    """LLM 기반 사용자 반응 시뮬레이터.

    LLMUserSimulator를 사용해 페르소나 기반 사용자 반응을 생성하고,
    LLMResponseHandler로 파싱 및 검증합니다. 오류 발생 시 RandomResponseSimulator로 폴백합니다.
    """

    def __init__(
        self,
        llm_simulator: LLMUserSimulator,
        persona_id: Optional[int] = None,
        debug: bool = False,
    ) -> None:
        """LLMResponseSimulator 생성자.

        Args:
            llm_simulator (LLMUserSimulator): LLM 기반 사용자 시뮬레이터 인스턴스.
            persona_id (Optional[int]): 사용할 페르소나 ID. None이면 랜덤 선택.
            debug (bool): 디버그 모드 활성화 여부.
        """
        self.llm_simulator = llm_simulator
        self.response_handler = LLMResponseHandler(debug=debug)
        self.fallback_simulator = RandomResponseSimulator()
        self._init_persona(persona_id, debug)

    def _init_persona(self, persona_id: Optional[int], debug: bool) -> None:
        """페르소나 정보를 초기화합니다.

        Args:
            persona_id (Optional[int]): 지정할 페르소나 ID.
            debug (bool): 디버그 모드 활성화 여부.

        Raises:
            ValueError: 지정한 persona_id가 DB에 없을 경우.
        """
        persona_db = get_persona_db()

        if persona_id is None:
            persona = persona_db.get_random_persona()
            if debug:
                logging.info(
                    "랜덤 페르소나 선택: ID %d (%s, 레벨 %d)",
                    persona.persona_id,
                    persona.mbti,
                    persona.investment_level,
                )
        else:
            persona = persona_db.get_persona_by_id(persona_id)
            if not persona:
                raise ValueError(f"Persona {persona_id} not found in database")
            if debug:
                logging.info(
                    "지정 페르소나: ID %d (%s, 레벨 %d)",
                    persona.persona_id,
                    persona.mbti,
                    persona.investment_level,
                )

        self.persona_id = persona.persona_id
        self.persona_mbti = persona.mbti
        self.persona_investment_level = persona.investment_level

    def simulate_responses(
        self,
        selected_contents: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """LLM을 통해 페르소나 기반 사용자 반응을 시뮬레이션합니다.

        LLM 호출 및 응답 파싱/검증을 수행하고, 오류 시 랜덤 시뮬레이터로 폴백합니다.

        Args:
            selected_contents (List[Dict[str, Any]]): 추천된 콘텐츠 리스트.
            context (Dict[str, Any]): 시뮬레이션 컨텍스트.
                - "step_count": int
                - "session_logs": List[Dict[str, Any]]

        Returns:
            List[Dict[str, Any]]:
                [
                    {"content_id": Any, "clicked": bool, "dwell_time": int},
                    ...
                ]
        """
        try:
            logging.debug("LLM 시뮬레이터에 %d개 콘텐츠 전송", len(selected_contents))
            raw = self.llm_simulator.simulate_user_response(
                persona_id=self.persona_id,
                mbti=self.persona_mbti,
                investment_level=self.persona_investment_level,
                recommended_contents=selected_contents,
                current_context={
                    "step_count": context.get("step_count"),
                    "session_logs": context.get("session_logs", []),
                },
            )
            return self.response_handler.extract_all_responses(
                llm_raw_text=raw,
                all_contents=selected_contents,
            )
        except Exception as e:
            logging.error("LLM 시뮬레이션 오류: %s — 랜덤 시뮬레이터로 폴백합니다.", e)
            return self.fallback_simulator.simulate_responses(selected_contents, {})
