from typing import Any, Dict, List

import numpy as np

from components.base import BaseCandidateGenerator
from components.registry import register
from components.db_utils import get_contents


@register("query")
class QueryCandidateGenerator(BaseCandidateGenerator):
    """
    각 콘텐츠 타입별로 관심사(Query)에 기반해 max_count_by_content개의 후보를 반환하는 후보군 생성기입니다.
    """

    def __init__(self, max_count_by_content: int):
        """
        Args:
            max_count_by_content (int): 각 타입별 반환할 후보군의 최대 개수
        """
        self.max_count_by_content = max_count_by_content
        self.all_contents_df = get_contents()
        # 성능 개선: 대량 데이터에서는 미리 records로 캐싱하거나, DataFrame을 활용해 nlargest로 최적화하는 방안을 추천
        # self.records_by_type = {...}  # 향후 캐시 구조 등으로 확장 가능

    def get_candidates(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        사용자의 관심사(query)를 바탕으로 각 콘텐츠 타입별 후보를 반환합니다.
        all_contents_df에 있는 search_query_text 필터링합니다.

        Args:
            query (str): 필터링할 검색어 문자열

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

        # 입력된 query를 사용하여 all_contents_df 필터링
        # search_query_text 컬럼이 query 문자열을 포함하는 경우
        # NaN/None 값은 빈 문자열로 처리하여 에러 방지
        # 'search_query_text' 컬럼이 없을 경우를 대비하여 확인
        if "search_query_text" not in self.all_contents_df.columns:
            # 이 경우, search_query_text가 없으므로 필터링 불가. 빈 결과를 반환하거나, 로깅 후 예외처리 가능.
            # 여기서는 모든 타입에 빈 리스트 반환
            for t in types:
                out[t] = []
            # print("Warning: 'search_query_text' column not found in all_contents_df.")
            return out

        # 실제 필터링 로직
        # Series.str.contains는 기본적으로 NaN에 대해 NaN을 반환하므로, na=False로 처리하거나 fillna 필요
        mask = (
            self.all_contents_df["search_query_text"]
            .fillna("")
            .str.contains(query, case=False, na=False)
        )
        filtered_df = self.all_contents_df[mask]

        if filtered_df.empty:
            for t in types:  # 필터링된 결과가 없으면 모든 타입에 빈 리스트 반환
                out[t] = []
            return out

        # 필터링된 DataFrame을 사용하여 타입별로 처리
        for t in types:
            # 현재 타입(t)에 해당하는 콘텐츠만 선택
            type_specific_contents_df = filtered_df[filtered_df["type"] == t]

            if type_specific_contents_df.empty:
                out[t] = []
                continue

            # DataFrame을 레코드(dictionary) 리스트로 변환
            pool = type_specific_contents_df.to_dict("records")

            # 사용자의 요청은 "모두 반환"이므로, max_count_by_content는 최대 개수 제한으로 작용
            # 필터링된 결과 내에서 상위 num_to_select 개를 선택 (현재는 순서 유지)
            num_to_select = min(self.max_count_by_content, len(pool))

            out[t] = pool[:num_to_select]

        return out
