from components.base import BaseRewardFn
from components.registry import register

@register("default")
class DefaultRewardFunction(BaseRewardFn):
    def calculate(self, content: dict) -> float:
        click = content.get("clicked", 0)
        dwell = content.get("dwell", 0)
        emotion = content.get("emotion", 0)
        return click*1.0 + dwell*0.01 + emotion*0.1