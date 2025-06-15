import yaml
from typing import Any, Dict, Optional 
import ast

import numpy as np
import pandas as pd
from components.core.base import BaseUserEmbedder
from components.database.db_utils import get_contents
from components.registry import register, make


@register("weighted_user")
class WeightedUserEmbedder(BaseUserEmbedder):
    """시간 가중 평균을 이용한 사용자 임베딩 클래스."""

    def __init__(
        self,
        user_dim: int = 300,
        time_decay_factor: float = 0.9,
        max_logs: int = 10,
        all_contents_df: Optional[object] = None,
    ) -> None:
        """WeightedUserEmbedder 초기화 함수.

        Args:
            user_dim (int): 사용자 임베딩 차원. 기본값은 300.
            time_decay_factor (float): 시간 가중치 감소 계수. 기본값은 0.9.
            max_logs (int): 최대 로그 수. 기본값은 10.
            all_contents_df (Optional[object]): 콘텐츠 임베딩 데이터프레임 (없을 경우 DB에서 로드).
        """
        
        self.cfg: Dict[str, Any] = yaml.safe_load(open("./config/experiment.yaml"))
        
        # 생성시 인자로 전달받은 DataFrame이 없다면 실제 DB에서 가져옴
        self.all_contents_df = (
            all_contents_df if all_contents_df is not None else get_contents()
        )

        # 시간 가중치 관련 설정
        self.time_decay_factor = time_decay_factor
        self.max_logs = max_logs

        # 출력 차원 설정
        # yaml파일에서 content embedder랑 차원 같게 바꿔주실 수 있나요,,,
        self.user_dim = user_dim

    def output_dim(self) -> int:
        """임베딩 벡터의 출력 차원을 반환합니다.

        Returns:
            int: 사용자 임베딩 벡터의 차원 수.
        """
        return self.user_dim

    def embed_user(self, user: dict) -> np.ndarray:
        """사용자의 로그 기반 시간 가중 임베딩 벡터를 생성합니다.

        Args:
            user (dict): 사용자 정보와 로그를 포함한 딕셔너리. 
                - user_info: {"id": 사용자 ID}
                - recent_logs: [{"user_id", "content_id", "timestamp"}]
                - current_time: datetime 객체

        Returns:
            np.ndarray: 시간 가중치가 적용된 사용자 임베딩 벡터.
        """

        user_id = int(user.get("user_info")["id"])
        logs = user.get("recent_logs", [])
        current_time = user.get("current_time")

        if not logs:
            # 로그가 없으면 전부 0벡터 반환
            return np.zeros(self.user_dim, dtype=np.float32)
        
        logs_df = pd.DataFrame(logs)        
        user_logs = logs_df[logs_df["user_id"]==user_id]
        if user_logs.empty:
            return None
        
        weighted_embedding = []

        for _, row in user_logs.iterrows():
            content_id = row["content_id"]
            timestamp = pd.to_datetime(row["timestamp"])
            embeddings = (self.all_contents_df[self.all_contents_df["id"]==content_id]["embedding"])

            # 콘텐츠 임베딩이 없을 경우 임베딩 생성
            if embeddings is None or not isinstance(embeddings, pd.Series):
                cfg: Dict[str, Any] = self.cfg
                embedder = make(cfg["embedder"]["type"], **cfg["embedder"]["params"])
                embeddings = embedder.embed_content(row) 
            else:
                embeddings = (embeddings.iloc[0])
                embeddings = np.array(ast.literal_eval(embeddings))


            hours_diff = (current_time - timestamp).total_seconds() / 3600
            weight = self.time_decay_factor ** hours_diff

            weighted_embedding.append(embeddings * weight)
        
        weighted_embedding = np.mean(weighted_embedding, axis=0)
        
        # 지정된 차원으로 패딩 또는 자르기
        if len(weighted_embedding) < self.user_dim:
            weighted_embedding = np.pad(weighted_embedding, (0, self.user_dim - len(weighted_embedding)), "constant")
        elif len(weighted_embedding) > self.user_dim:
            weighted_embedding = weighted_embedding[:self.user_dim]

        return weighted_embedding.astype(np.float32)
