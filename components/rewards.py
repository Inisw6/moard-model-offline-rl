from typing import Dict
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
