import numpy as np
from components.base import BaseCandidateGenerator
from components.registry import register

@register("top_k")
class TopKCandidateGenerator(BaseCandidateGenerator):
    def __init__(self, top_k: int):
        self.top_k = top_k

    def get_candidates(self, state):
        types = ["youtube", "blog", "news"]
        out = {}
        for t in types:
            pool = [{"id":i, "type":t,
                     "emotion":np.random.uniform(-1,1),
                     "dwell":np.random.uniform(0,30)}
                    for i in range(100)]
            scores = np.random.rand(len(pool))
            idxs = np.argsort(scores)[::-1][:self.top_k]
            out[t] = [pool[i] for i in idxs]
        return out