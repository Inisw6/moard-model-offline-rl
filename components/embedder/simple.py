import json
import logging
from typing import Any, Dict, Optional

import numpy as np

from components.core.base import BaseContentEmbedder, BaseUserEmbedder
from components.database.db_utils import get_contents
from components.registry import register


@register("simple_user")
class SimpleUserEmbedder(BaseUserEmbedder):
    """사용자 로그 기반 단순 임베딩 추출기.

    최근 사용자 로그를 요약하여 고정 차원 벡터로 변환합니다.

    Vector 구성:
        [평균 ratio, 평균 time, 콘텐츠 타입별 비율…, 0 패딩]

    Attributes:
        content_types (List[str]): 처리 가능한 콘텐츠 타입 목록.
        type_to_idx_map (Dict[str, int]): 타입 → 인덱스 매핑.
        user_dim (int): 출력 임베딩 벡터 차원.
    """

    def __init__(
        self,
        user_dim: int = 30,
        all_contents_df: Optional[Any] = None,
    ) -> None:
        """SimpleUserEmbedder 생성자.

        Args:
            user_dim (int): 출력할 유저 임베딩 벡터의 최소 차원.
            all_contents_df (Optional[pd.DataFrame]): 테스트용 외부 콘텐츠 DataFrame.
        """
        # 콘텐츠 메타데이터 로드
        self.all_contents_df = (
            all_contents_df if all_contents_df is not None else get_contents()
        )
        if not self.all_contents_df.empty:
            self.content_types = self.all_contents_df["type"].unique().tolist()
        else:
            self.content_types = ["youtube", "blog", "news"]

        self.num_content_types = len(self.content_types)
        self.type_to_idx_map = {t: i for i, t in enumerate(self.content_types)}

        # 최소 차원 보장: [ratio, time] + 타입 수
        min_dim = 2 + self.num_content_types
        if user_dim < min_dim:
            logging.warning(
                "user_dim (%d)가 너무 작습니다. 최소값 %d로 조정합니다.",
                user_dim,
                min_dim,
            )
            self.user_dim = min_dim
        else:
            self.user_dim = user_dim

    def output_dim(self) -> int:
        """유저 임베딩 벡터의 차원을 반환합니다.

        Returns:
            int: 유저 임베딩 벡터 차원.
        """
        return self.user_dim

    def embed_user(self, user: Dict[str, Any]) -> np.ndarray:
        """사용자 데이터를 임베딩 벡터로 변환합니다.

        Args:
            user (Dict[str, Any]):
                {
                    "user_info": ...,
                    "recent_logs": List[Dict],
                    "current_time": datetime
                }

        Returns:
            np.ndarray: (user_dim,) 크기의 임베딩 벡터.
                [0]=평균 ratio, [1]=평균 time,
                [2:2+N]=타입별 비율, [나머지]=0
        """
        logs = user.get("recent_logs", [])
        if not logs:
            return np.zeros(self.user_dim, dtype=np.float32)

        ratio_avg = np.mean([l.get("ratio", 0.0) for l in logs])
        time_avg = np.mean([l.get("time", 0.0) for l in logs])

        counts = {t: 0 for t in self.content_types}
        for log in logs:
            t = str(log.get("content_actual_type", "")).lower()
            if t in counts:
                counts[t] += 1

        total = sum(counts.values())
        type_vec = np.array(
            [(counts[t] / total) if total > 0 else 0.0 for t in self.content_types],
            dtype=np.float32,
        )

        vec = np.concatenate([[ratio_avg, time_avg], type_vec])
        if len(vec) < self.user_dim:
            vec = np.pad(vec, (0, self.user_dim - len(vec)), constant_values=0.0)
        else:
            vec = vec[: self.user_dim]

        return vec.astype(np.float32)


@register("simple_content")
class SimpleContentEmbedder(BaseContentEmbedder):
    """사전 임베딩 + 타입 원핫 기반 단순 콘텐츠 임베더.

    저장된 JSON 임베딩과 콘텐츠 타입을 결합하여 벡터를 생성합니다.

    Vector 구성:
        [사전 임베딩…, 타입 원핫…, 0 패딩]

    Attributes:
        content_types (List[str]): 처리 가능한 콘텐츠 타입 목록.
        type_to_idx_map (Dict[str, int]): 타입 → 인덱스 매핑.
        content_dim (int): 출력 임베딩 벡터 차원.
        pretrained_content_embedding_dim (int): 사전 임베딩 차원.
    """

    def __init__(
        self,
        content_dim: int = 5,
        all_contents_df: Optional[Any] = None,
    ) -> None:
        """SimpleContentEmbedder 생성자.

        Args:
            content_dim (int): 출력할 콘텐츠 임베딩 벡터의 차원 (타입 수 이상).
            all_contents_df (Optional[pd.DataFrame]): 테스트용 외부 콘텐츠 DataFrame.
        """
        # 콘텐츠 메타데이터 로드
        self.all_contents_df = (
            all_contents_df if all_contents_df is not None else get_contents()
        )
        if not self.all_contents_df.empty:
            self.content_types = self.all_contents_df["type"].unique().tolist()
        else:
            self.content_types = ["youtube", "blog", "news"]

        self.num_content_types = len(self.content_types)
        self.type_to_idx_map = {t: i for i, t in enumerate(self.content_types)}

        # 사전 임베딩 차원 설정
        emb_dim = content_dim - self.num_content_types
        if emb_dim < 0:
            logging.warning(
                "content_dim (%d) < 타입 수 (%d). 타입 수로 조정합니다.",
                content_dim,
                self.num_content_types,
            )
            emb_dim = 0
            self.content_dim = self.num_content_types
        else:
            self.content_dim = content_dim

        self.pretrained_content_embedding_dim = emb_dim

    def output_dim(self) -> int:
        """콘텐츠 임베딩 벡터의 차원을 반환합니다.

        Returns:
            int: 콘텐츠 임베딩 벡터 차원.
        """
        return self.content_dim

    def embed_content(self, content: Dict[str, Any]) -> np.ndarray:
        """콘텐츠 데이터를 임베딩 벡터로 변환합니다.

        Args:
            content (Dict[str, Any]):
                {
                    "embedding": str,  # JSON 문자열
                    "type": str
                }

        Returns:
            np.ndarray: (content_dim,) 크기의 임베딩 벡터.
        """
        # 사전 임베딩 로드
        if self.pretrained_content_embedding_dim > 0:
            try:
                s = content.get("embedding") or "[]"
                arr = json.loads(s)
                if not isinstance(arr, list) or not all(
                    isinstance(x, (int, float)) for x in arr
                ):
                    raise ValueError
                emb = np.array(arr, dtype=np.float32)
                if emb.shape[0] != self.pretrained_content_embedding_dim:
                    raise ValueError
            except Exception:
                emb = np.zeros(self.pretrained_content_embedding_dim, dtype=np.float32)
        else:
            emb = np.zeros(0, dtype=np.float32)

        # 타입 원핫 인코딩
        t = content.get("type", "").lower()
        onehot = np.zeros(self.num_content_types, dtype=np.float32)
        idx = self.type_to_idx_map.get(t, -1)
        if idx >= 0:
            onehot[idx] = 1.0

        # 최종 벡터 결합
        vec = np.concatenate([emb, onehot])
        if vec.shape[0] < self.content_dim:
            vec = np.pad(vec, (0, self.content_dim - vec.shape[0]), constant_values=0.0)
        else:
            vec = vec[: self.content_dim]

        return vec.astype(np.float32)
