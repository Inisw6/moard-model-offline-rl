import numpy as np
from components.base import BaseEmbedder
from components.registry import register

@register("simple_concat")
class SimpleConcatBuilder(BaseEmbedder):
    def __init__(self, user_dim: int = 30, content_dim: int = 5):
        self.user_dim = user_dim
        self.content_dim = content_dim

    def output_dim(self) -> int:
        return self.user_dim

    def embed_user(self, user: dict) -> np.ndarray:
        logs = user["recent_logs"]
        emotion_avg = np.mean([l["emotion"] for l in logs])
        dwell_avg = np.mean([l["dwell"] for l in logs])
        counts = {"youtube":0, "blog":0, "news":0}
        for l in logs:
            counts[l["type"].lower()] += 1
        total = len(logs)
        type_vec = np.array([counts["youtube"]/total, counts["blog"]/total, counts["news"]/total])
        vec = np.concatenate([[emotion_avg, dwell_avg], type_vec, np.zeros(self.user_dim-5)])
        return vec.astype(np.float32)

    def embed_content(self, content: dict) -> np.ndarray:
        emotion = content.get("emotion", 0.0)
        dwell = content.get("dwell", 0.0)/30.0
        type_onehot = {"youtube":[1,0,0],"blog":[0,1,0],"news":[0,0,1]}[content["type"]]
        vec = np.array([emotion, dwell] + type_onehot)
        if len(vec) < self.content_dim:
            vec = np.concatenate([vec, np.zeros(self.content_dim-len(vec))])
        return vec.astype(np.float32)

    def estimate_preference(self, state: np.ndarray) -> dict:
        yt, bl, nw = state[2], state[3], state[4]
        return {"youtube":float(yt), "blog":float(bl), "news":float(nw)}