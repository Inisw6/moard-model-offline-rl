from typing import Any, Dict, List

import pandas as pd

from components.core.base import BaseCandidateGenerator
from components.registry import register


@register("query")
class QueryCandidateGenerator(BaseCandidateGenerator):
    """쿼리(관심사) 기반 추천 후보 생성기.

    사용자의 텍스트 쿼리를 바탕으로, 각 콘텐츠 타입별로 최대 max_count_by_content개의 후보를 반환합니다.

    Attributes:
        max_count_by_content (int): 콘텐츠 타입별 최대 후보 개수.
        all_contents_df (pd.DataFrame): 전체 콘텐츠 목록 DataFrame.
    """

    def __init__(self, contents_df: pd.DataFrame, max_count_by_content: int) -> None:
        """QueryCandidateGenerator 생성자.

        Args:
            contents_df (pd.DataFrame): 전체 콘텐츠 정보 DataFrame.
            max_count_by_content (int): 콘텐츠 타입별 최대 후보 개수.
        """
        self.max_count_by_content: int = max_count_by_content
        self.all_contents_df: pd.DataFrame = contents_df

    def get_candidates(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """사용자 쿼리에 따른 콘텐츠 후보를 타입별로 반환합니다.

        Args:
            query (str): 필터링할 검색어.

        Returns:
            Dict[str, List[Dict[str, Any]]]:
                {
                    콘텐츠 타입명: [콘텐츠 dict, ...],
                    ...
                }
        """
        # 콘텐츠 타입 목록 확보
        types = (
            self.all_contents_df["type"].unique().tolist()
            if not self.all_contents_df.empty
            else ["youtube", "blog", "news"]
        )
        candidates: Dict[str, List[Dict[str, Any]]] = {t: [] for t in types}

        if (
            self.all_contents_df.empty
            or "search_query_text" not in self.all_contents_df.columns
        ):
            # 데이터가 없거나 검색 텍스트 컬럼이 없으면 빈 결과 반환
            return candidates

        # 쿼리 필터링 (대소문자 무시, NaN은 빈 문자열로 처리)
        mask = (
            self.all_contents_df["search_query_text"]
            .fillna("")
            .str.contains(query, case=False, na=False)
        )
        filtered = self.all_contents_df[mask]
        if filtered.empty:
            return candidates

        # 타입별로 최대 max_count_by_content개씩 추출
        for t in types:
            df_t = filtered[filtered["type"] == t]
            if df_t.empty:
                continue
            records = df_t.to_dict("records")
            candidates[t] = records[: self.max_count_by_content]

        return candidates
