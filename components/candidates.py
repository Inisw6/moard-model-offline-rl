import numpy as np
from typing import Dict, List, Any
from components.base import BaseCandidateGenerator
from components.registry import register
from .db_utils import get_contents


@register("top_k")
class TopKCandidateGenerator(BaseCandidateGenerator):
    """
    각 콘텐츠 타입별로 상위 K개의 후보를 반환하는 간단한 후보군 생성기입니다.
    현재는 임시로 무작위 점수를 사용해 상위 K개를 선정합니다.
    """

    def __init__(self, top_k: int):
        """
        Args:
            top_k (int): 각 타입별 반환할 후보군의 최대 개수
        """
        self.top_k = top_k
        self.all_contents_df = get_contents()
        # 성능 개선: 대량 데이터에서는 미리 records로 캐싱하거나, DataFrame을 활용해 nlargest로 최적화하는 방안을 추천
        # self.records_by_type = {...}  # 향후 캐시 구조 등으로 확장 가능

    def get_candidates(
        self, state: np.ndarray | None = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        현재 구현은 상태(state)를 사용하지 않고, 각 타입별로 임시로 무작위 점수를 이용해 상위 K개 콘텐츠를 반환합니다.

        Args:
            state (np.ndarray | None): 사용자 상태 (현재 미사용, 추후 확장 가능)

        Returns:
            Dict[str, List[Dict[str, Any]]]: {콘텐츠타입(str): [콘텐츠 dict, ...]}
        """
        types = (
            self.all_contents_df["type"].unique()
            if not self.all_contents_df.empty
            else ["youtube", "blog", "news"]
        )
        out: Dict[str, List[Dict[str, Any]]] = {}

        if self.all_contents_df.empty:
            for t in types:
                out[t] = []
            return out

        for t in types:
            type_specific_contents_df = self.all_contents_df[
                self.all_contents_df["type"] == t
            ]
            if type_specific_contents_df.empty:
                out[t] = []
                continue

            pool = type_specific_contents_df.to_dict("records")

            # TODO: 임시로 랜덤 점수 사용, 추후 모델/규칙 기반 점수로 대체 필요
            scores = np.random.rand(len(pool))
            # 아래 유효성 검사는 향후 점수가 모델에서 나올 때 NaN/Inf 발생 가능성에 대비한 것.
            valid_indices = np.where(np.isfinite(scores))[0]
            if len(valid_indices) < len(scores):
                pool = [pool[i] for i in valid_indices]
                scores = scores[valid_indices]
            if not pool:
                out[t] = []
                continue

            num_to_select = min(self.top_k, len(pool))
            # np.argsort(scores)[::-1][:num_to_select]는 내림차순 Top-K
            idxs = np.argsort(scores)[::-1][:num_to_select]
            out[t] = [pool[i] for i in idxs]
            # 참고: 대량 데이터에서는 DataFrame.nlargest(num_to_select, "score") 사용 추천

        return out
