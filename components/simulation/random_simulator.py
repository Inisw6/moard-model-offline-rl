import random
from typing import Any, Dict, List

from components.core.base import BaseResponseSimulator


class RandomResponseSimulator(BaseResponseSimulator):
    """룰 기반(랜덤) 사용자 반응 시뮬레이터."""

    def simulate_responses(
        self, selected_contents: List[Dict], context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """각 콘텐츠에 대해 30% 확률로 클릭 및 랜덤 체류 시간을 시뮬레이션합니다.

        Args:
            selected_contents (List[Dict]): 추천된 콘텐츠 리스트.
            context (Dict[str, Any], optional): 사용되지 않음.

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
