import numpy as np
from components.base import BaseCandidateGenerator
from components.registry import register
from .db_utils import get_contents
import pandas as pd

@register("top_k")
class TopKCandidateGenerator(BaseCandidateGenerator):
    def __init__(self, top_k: int):
        self.top_k = top_k
        self.all_contents_df = get_contents()

    def get_candidates(self, state):
        types = self.all_contents_df['type'].unique() if not self.all_contents_df.empty else ["youtube", "blog", "news"]
        out = {}

        if self.all_contents_df.empty:
            for t in types:
                out[t] = []
            return out

        for t in types:
            type_specific_contents_df = self.all_contents_df[self.all_contents_df['type'] == t]
            
            if type_specific_contents_df.empty:
                out[t] = []
                continue

            pool = type_specific_contents_df.to_dict('records')

            scores = np.random.rand(len(pool))
            
            valid_indices = np.where(np.isfinite(scores))[0]
            if len(valid_indices) < len(scores):
                pool = [pool[i] for i in valid_indices]
                scores = scores[valid_indices]
            
            if not pool:
                out[t] = []
                continue

            num_to_select = min(self.top_k, len(pool))
            idxs = np.argsort(scores)[::-1][:num_to_select]
            out[t] = [pool[i] for i in idxs]
            
        return out