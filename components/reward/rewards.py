from typing import Dict, List, Tuple

from components.core.base import BaseRewardFn
from components.registry import register


@register("default")
class DefaultRewardFunction(BaseRewardFn):
    """기본(Default) 보상 함수 클래스.

    클릭 여부와 체류시간을 기반으로 보상을 계산합니다.
    """

    def calculate(self, clicked: bool = False, dwell_time: int = 0) -> float:
        """클릭 및 체류시간에 따라 보상을 계산합니다.

        클릭하면 1.0의 기본 보상에 체류시간에 따른 추가 보상(최대 0.5)이 부여되고,
        클릭하지 않으면 0.1의 기본 보상에 체류시간에 따른 추가 보상(최대 0.2)이 부여됩니다.

        Args:
            clicked (bool): 클릭 여부.
            dwell_time (int): 체류시간(초 단위).

        Returns:
            float: 계산된 보상 값.
        """
        reward = 0.0
        if clicked:
            reward = 1.0
            # 체류시간에 따른 추가 보상
            reward += min(dwell_time * 0.001, 0.5)  # 최대 0.5 추가
        else:
            reward = 0.1  # 기본 VIEW 보상
            # 체류시간이 있는 경우 작은 보상 추가
            if dwell_time > 0:
                reward += min(dwell_time * 0.0005, 0.2)  # 최대 0.2 추가

        return reward

    def calculate_from_topk_responses(
        self, all_responses: List[Dict]
    ) -> Tuple[float, Dict[int, float]]:
        """top-k 콘텐츠에 대한 LLM 응답을 기반으로 총 보상과 개별 보상을 계산합니다.

        Args:
            all_responses (List[Dict]):
                top-k 콘텐츠에 대한 LLM 응답 리스트.
                각 원소는 {"content_id": int, "clicked": bool, "dwell_time": int}.

        Returns:
            Tuple[float, Dict[int, float]]:
                - 총 보상값(float)
                - 각 콘텐츠별 보상값 {content_id: reward, ...}
        """
        total_reward = 0.0
        individual_rewards = {}

        # 각 응답에 대한 보상 계산
        for response in all_responses:
            content_id = int(response["content_id"])
            clicked = response["clicked"]
            dwell_time = response["dwell_time"]

            # calculate 함수를 사용하여 개별 보상 계산
            reward = self.calculate(clicked=clicked, dwell_time=dwell_time)
            individual_rewards[content_id] = reward
            total_reward += reward

        return total_reward, individual_rewards
