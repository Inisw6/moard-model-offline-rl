from typing import Any, Dict, List, Tuple

from components.core.base import BaseRewardFn
from components.registry import register


@register("default")
class DefaultRewardFunction(BaseRewardFn):
    """기본(Default) 보상 함수.

    클릭 여부와 체류 시간에 기반하여 보상을 계산합니다.
    """

    def calculate(self, clicked: bool = False, dwell_time: int = 0) -> float:
        """단일 콘텐츠의 보상을 계산합니다.

        클릭 시 기본 1.0 + 체류 시간 보상, 클릭하지 않을 시 기본 0.1 + 소량 보상을 부여합니다.

        Args:
            clicked (bool): 클릭 여부 (True → 클릭, False → 뷰).
            dwell_time (int): 체류 시간(초).

        Returns:
            float: 계산된 보상 값.
                - clicked=True: 1.0 + min(dwell_time * 0.001, 0.5)
                - clicked=False: 0.1 + (dwell_time > 0 → min(dwell_time * 0.0005, 0.2))
        """
        if clicked:
            return 1.0 + min(dwell_time * 0.001, 0.5)
        else:
            return 0.1 + (min(dwell_time * 0.0005, 0.2) if dwell_time > 0 else 0.0)

    def calculate_from_topk_responses(
        self, all_responses: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[int, float]]:
        """top-k 응답을 기반으로 총 보상과 개별 보상을 계산합니다.

        Args:
            all_responses (List[Dict[str, Any]]):
                콘텐츠 응답 리스트. 각 요소는
                {
                    "content_id": int,
                    "clicked": bool,
                    "dwell_time": int
                }

        Returns:
            Tuple[float, Dict[int, float]]:
                total_reward: 모든 콘텐츠 보상의 합.
                individual_rewards: {content_id: reward} 형태의 개별 보상 매핑.
        """
        total_reward = 0.0
        individual_rewards: Dict[int, float] = {}

        for resp in all_responses:
            content_id = int(resp["content_id"])
            clicked = resp.get("clicked", False)
            dwell_time = resp.get("dwell_time", 0)

            reward = self.calculate(clicked=clicked, dwell_time=dwell_time)
            individual_rewards[content_id] = reward
            total_reward += reward

        return total_reward, individual_rewards
