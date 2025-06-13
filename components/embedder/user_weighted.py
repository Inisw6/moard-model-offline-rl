import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from components.core.base import BaseUserEmbedder
from components.database.db_utils import get_contents, get_user_logs
from components.registry import register


@register("weighted_user")
class WeightedUserEmbedder(BaseUserEmbedder):

    def __init__(
        self,
        user_dim: int = 300,
        time_decay_factor: float = 0.1,
        max_logs: int = 100,
        all_logs_df: Optional[object] = None,
        all_contents_df: Optional[object] = None,
    ) -> None:

        self.all_logs_df = (
            all_logs_df if all_logs_df is not None else get_user_logs()
        )
        # 생성시 인자로 전달받은 DataFrame이 없다면 실제 DB에서 가져옴
        self.all_contents_df = (
            all_contents_df if all_contents_df is not None else get_contents()
        )
        if not self.all_contents_df.empty:
            # 고유한 콘텐츠 타입 목록을 추출
            self.content_types = self.all_contents_df["type"].unique().tolist()
        else:
            # 데이터가 비어 있으면 기본값 사용
            self.content_types = ["youtube", "blog", "news"]

        self.num_content_types = len(self.content_types)

        # 시간 가중치 관련 설정
        self.time_decay_factor = time_decay_factor
        self.max_logs = max_logs
        # 출력 차원 설정
        self.user_dim = user_dim

    def _calculate_weighted_user_embedding(self, user_id: int) -> np.ndarray:
        user_logs = self.all_logs_df
        user_clicks = user_logs[(user_logs["user_id"]==user_id) & (user_logs["event_type"] == "CLICK")].head(self.max_logs)

        if user_clicks.empty:
            return None
        
        now = datetime.now(timezone.utc)
        weighted_embeddings = []
        weights = []
        
        for _, row in user_clicks.iterrows():
            content_id = row["content_id"]
            timestamp = row["timestamp"]
            embeddings = self.all_contents_df[self.all_contents_df["content_id"]==content_id]["embedding"]
            # embedding이 안되어있으면, 조건추가

            hours_diff = (now - timestamp).total_seconds() / 3600
            weight = self.time_decay_factor ** hours_diff

            weighted_embeddings.append(embeddings * weight)

            return np.sum(weighted_embeddings, axis=0)/len(weighted_embeddings)
        
    def _calculate_all_user_embeddings(self) -> dict[int, np.ndarray]:

        user_embeddings = {}
        for user_id in self.all_logs_df["user_id"].unique():
            user_embeddings[user_id] = self._calculate_weighted_user_embedding(user_id)

        return user_embeddings

    def output_dim(self):
        """유저 임베딩 벡터의 차원을 반환합니다.

        Returns:
            int: 유저 임베딩 벡터의 차원.
        """
        return self.user_dim

    def embed_user(self, user: dict) -> np.ndarray:

        uid = user.get("uid", -1)
        
        # 사용자 임베딩 계산
        try:
            user_embedding = self._calculate_weighted_user_embedding(uid)
            if user_embedding is None or len(user_embedding) == 0:
                raise ValueError("Invalud embedding")
        except Exception:
            return np.zeros(self.user_dim, dtype=np.float32)
        
        # 길이 맞춰 패딩 또는 자르기
        if len(user_embedding) < self.user_dim:
            user_embedding = np.pad(user_embedding, (0, self.user_dim - len(user_embedding)), "constant")
        elif len(user_embedding) > self.user_dim:
            user_embedding = user_embedding[:self.user_dim]

        return user_embedding.astype(np.float32)
    
    def estimate_preference(self, state: np.ndarray) -> dict:
        return {ctype: 1.0 / self.num_content_types for ctype in self.content_types}
