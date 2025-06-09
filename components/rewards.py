from typing import Dict, List

from components.base import BaseRewardFn
from components.registry import register


@register("default")
class DefaultRewardFunction(BaseRewardFn):
    """
    기본 보상 함수.
    현재는 이벤트 타입(event_type)에만 기반하여 보상을 계산합니다.

    Args:
        content (dict): 콘텐츠 정보. 현재는 사용하지 않음. (향후 보상 로직 확장 시 사용 가능)
        event_type (str): 이벤트 타입 ("VIEW", "CLICK" 등)
    Returns:
        float: 이벤트에 따른 보상 값 (CLICK: 1.0, VIEW: 0.1)
    """

    def calculate(self, content: Dict, event_type: str = "VIEW") -> float:
        # content: 현재는 사용하지 않음. event_type만 사용.
        reward = 0.0
        if event_type == "CLICK":
            reward = 1.0
        elif event_type == "VIEW":
            reward = 0.1  # VIEW에 대한 작은 보상

        # 기존 content 기반 보상 로직은 주석 처리. 향후 아래 부분을 참고하여 확장할 수 있음.
        # click_in_content = content.get("clicked", 0)
        # dwell = content.get("dwell", 0)
        # emotion = content.get("emotion", 0)
        # return click_in_content*1.0 + dwell*0.01 + emotion*0.1 + reward_from_event

        return reward

    def calculate_from_responses(
        self, all_responses: List[Dict], selected_content: Dict, all_candidates: Dict
    ) -> float:
        """
        모든 후보에 대한 LLM 응답을 기반으로 보상을 계산합니다.

        Args:
            all_responses: 모든 후보에 대한 LLM 응답 리스트
                          [{"content_id": str, "clicked": bool, "dwell_time": int}, ...]
            selected_content: 선택된 콘텐츠 정보
            all_candidates: 전체 후보군 정보

        Returns:
            float: 계산된 보상 값
        """
        total_reward = 0.0
        selected_content_id = selected_content.get("id")

        # 각 후보에 대한 응답을 평가
        for response in all_responses:
            content_id = response["content_id"]
            clicked = response["clicked"]
            dwell_time = response["dwell_time"]

            # 기본 보상 계산
            if clicked:
                content_reward = 1.0  # 클릭 보상
                # 체류시간에 따른 추가 보상 (옵션)
                content_reward += min(dwell_time * 0.001, 0.5)  # 최대 0.5 추가
            else:
                content_reward = 0.1  # VIEW 보상

            # 선택된 콘텐츠에 대해서는 가중치 적용
            if content_id == selected_content_id:
                total_reward += content_reward * 1.0  # 선택된 콘텐츠는 100% 반영
            else:
                # 선택되지 않은 콘텐츠의 반응도 일부 반영 (탐색 장려)
                if clicked:
                    total_reward += (
                        content_reward * 0.1
                    )  # 클릭된 다른 콘텐츠는 10% 반영
                else:
                    total_reward += content_reward * 0.05  # VIEW만 된 콘텐츠는 5% 반영

        return total_reward

    def calculate_from_topk_responses(
        self, all_responses: List[Dict], selected_contents: List[Dict]
    ) -> float:
        """
        선택된 top-k 콘텐츠에 대한 LLM 응답을 기반으로 보상을 계산합니다.

        Args:
            all_responses: top-k 콘텐츠에 대한 LLM 응답 리스트
                          [{"content_id": str, "clicked": bool, "dwell_time": int}, ...]
            selected_contents: 선택된 top-k 콘텐츠 리스트

        Returns:
            float: 계산된 총 보상 값
        """
        total_reward = 0.0

        # 각 응답에 대한 보상 계산
        for response in all_responses:
            clicked = response["clicked"]
            dwell_time = response["dwell_time"]

            # 기본 보상 계산
            if clicked:
                content_reward = 1.0  # 클릭 보상
                # 체류시간에 따른 추가 보상
                content_reward += min(dwell_time * 0.001, 0.5)  # 최대 0.5 추가
            else:
                content_reward = 0.1  # VIEW 보상

            total_reward += content_reward

        return total_reward
