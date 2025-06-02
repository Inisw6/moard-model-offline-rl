from components.base import BaseRewardFn
from components.registry import register

@register("default")
class DefaultRewardFunction(BaseRewardFn):
    def calculate(self, content: dict, event_type: str = "VIEW") -> float:
        reward = 0.0
        if event_type == "CLICK":
            reward = 1.0
        elif event_type == "VIEW":
            reward = 0.1 # VIEW에 대한 작은 보상
        
        # 기존 content 기반 보상 로직은 주석 처리 또는 필요시 event_type과 결합
        # click_in_content = content.get("clicked", 0) # 이 필드는 현재 content dict에 없음
        # dwell = content.get("dwell", 0) # 이 필드는 현재 content dict에 없음
        # emotion = content.get("emotion", 0) # 이 필드는 현재 content dict에 없음
        # return click_in_content*1.0 + dwell*0.01 + emotion*0.1 + reward_from_event
        return reward