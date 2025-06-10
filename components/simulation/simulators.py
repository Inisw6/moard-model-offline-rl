import logging
import random
from typing import Any, Dict, List, Optional

from components.core.base import BaseResponseSimulator
from components.simulation.llm_response_handler import LLMResponseHandler
from components.simulation.llm_simu import LLMUserSimulator
from components.simulation.persona_db import get_persona_db


class RandomResponseSimulator(BaseResponseSimulator):
    """룰 기반(랜덤)으로 사용자 반응을 시뮬레이션하는 클래스입니다."""

    def simulate_responses(
        self, selected_contents: List[Dict], context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        각 콘텐츠에 대해 30% 확률로 클릭 및 랜덤 체류 시간을 시뮬레이션합니다.

        Args:
            selected_contents (List[Dict]): 추천된 콘텐츠 리스트.
            context (Dict[str, Any]): 사용되지 않습니다.

        Returns:
            List[Dict[str, Any]]: 시뮬레이션된 사용자 반응 리스트.
        """
        responses = []
        for content in selected_contents:
            clicked = random.random() < 0.3
            dwell_time = random.randint(60, 300) if clicked else 0
            responses.append(
                {
                    "content_id": content.get("id"),
                    "clicked": clicked,
                    "dwell_time": dwell_time,
                }
            )
        return responses


class LLMResponseSimulator(BaseResponseSimulator):
    """LLM을 사용하여 사용자 반응을 시뮬레이션하는 클래스입니다."""

    def __init__(
        self,
        llm_simulator: LLMUserSimulator,
        persona_id: Optional[int] = None,
        debug: bool = False,
    ):
        """
        Args:
            llm_simulator (LLMUserSimulator): LLM 기반 사용자 시뮬레이터.
            persona_id (Optional[int]): 페르소나 ID. None이면 랜덤 선택.
            debug (bool): 디버그 모드 활성화 여부.
        """
        self.llm_simulator = llm_simulator
        self.response_handler = LLMResponseHandler(debug=debug)
        self.fallback_simulator = RandomResponseSimulator()
        self._init_persona(persona_id, debug)

    def _init_persona(self, persona_id: Optional[int], debug: bool) -> None:
        """페르소나를 로드하고 속성에 저장합니다."""
        persona_db = get_persona_db()

        if persona_id is None:
            persona = persona_db.get_random_persona()
            if debug:
                logging.info(
                    "랜덤 페르소나 선택: ID%d (%s, 레벨%d)",
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
                    "지정 페르소나: ID%d (%s, 레벨%d)",
                    persona.persona_id,
                    persona.mbti,
                    persona.investment_level,
                )

        self.persona_id = persona.persona_id
        self.persona_mbti = persona.mbti
        self.persona_investment_level = persona.investment_level

    def simulate_responses(
        self, selected_contents: List[Dict], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        LLM을 사용하여 페르소나 기반으로 사용자 반응을 시뮬레이션합니다.
        오류 발생 시 랜덤 시뮬레이터로 대체됩니다.

        Args:
            selected_contents (List[Dict]): 추천된 콘텐츠 리스트.
            context (Dict[str, Any]): 시뮬레이션 컨텍스트. 다음 키를 포함해야 합니다:
                - step_count (int)
                - session_logs (List[Dict])

        Returns:
            List[Dict[str, Any]]: 시뮬레이션된 사용자 반응 리스트.
        """
        try:
            logging.debug(
                "Sending %d selected contents to LLM simulator", len(selected_contents)
            )

            raw_response = self.llm_simulator.simulate_user_response(
                persona_id=self.persona_id,
                mbti=self.persona_mbti,
                investment_level=self.persona_investment_level,
                recommended_contents=selected_contents,
                current_context={
                    "step_count": context["step_count"],
                    "session_logs": context["session_logs"],
                },
            )
            return self.response_handler.extract_all_responses(
                llm_raw_text=raw_response, all_contents=selected_contents
            )

        except Exception as e:
            logging.error(
                "LLM simulation error: %s. Falling back to random simulation.", e
            )
            return self.fallback_simulator.simulate_responses(selected_contents, {}) 