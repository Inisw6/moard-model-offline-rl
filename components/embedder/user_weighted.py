import ast
import json
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml
from typing import Any, Dict, Optional

from components.core.base import BaseUserEmbedder
from components.database.db_utils import get_contents
from components.registry import make, register


@register("weighted_user")
class WeightedUserEmbedder(BaseUserEmbedder):
    """시간 가중 평균(time-decayed average)을 이용한 사용자 임베더.

    최근 로그에 시간 가중치를 적용하여 콘텐츠 임베딩을 평균한 벡터를 반환합니다.

    Attributes:
        time_decay_factor (float): 시간 가중치 감소 계수 (0<α≤1).
        max_logs (int): 사용할 최대 로그 수.
        user_dim (int): 출력 임베딩 벡터 차원.
        all_contents_df (pd.DataFrame): 콘텐츠 메타데이터 및 임베딩 DataFrame.
        cfg (Dict[str, Any]): 실험 설정 로드된 YAML 구성.
    """

    def __init__(
        self,
        user_dim: int = 300,
        time_decay_factor: float = 0.9,
        max_logs: int = 10,
        all_contents_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """WeightedUserEmbedder 생성자.

        Args:
            user_dim (int): 출력할 사용자 임베딩 벡터 차원.
            time_decay_factor (float): 시간 가중치 감소 계수.
            max_logs (int): 사용할 최대 로그 개수.
            all_contents_df (Optional[pd.DataFrame]): 외부에서 주입된 콘텐츠 DataFrame.
                제공되지 않으면 DB에서 로드합니다.
        """
        # 설정 파일 로드
        with open("./config/experiment.yaml", "r", encoding="utf-8") as f:
            self.cfg: Dict[str, Any] = yaml.safe_load(f)

        # 콘텐츠 메타데이터 로드
        self.all_contents_df = (
            all_contents_df if all_contents_df is not None else get_contents()
        )

        self.time_decay_factor = time_decay_factor
        self.max_logs = max_logs
        self.user_dim = user_dim

    def output_dim(self) -> int:
        """출력 임베딩 벡터의 차원을 반환합니다.

        Returns:
            int: 사용자 임베딩 벡터 차원.
        """
        return self.user_dim

    def embed_user(self, user: Dict[str, Any]) -> np.ndarray:
        """사용자 로그에 기반한 시간 가중 임베딩 벡터를 생성합니다.

        Args:
            user (Dict[str, Any]):
                {
                    "user_info": {"id": int},
                    "recent_logs": List[Dict[str, Any]],  # 각 로그에는 content_id, timestamp 포함
                    "current_time": datetime
                }

        Returns:
            np.ndarray: (user_dim,) 크기의 임베딩 벡터.
        """
        user_id = int(user["user_info"]["id"])
        logs = user.get("recent_logs", [])
        current_time = user.get("current_time")

        if not logs or current_time is None:
            return np.zeros(self.user_dim, dtype=np.float32)

        # 사용자별 로그 필터링 및 최대 개수 제한
        df = pd.DataFrame(logs)
        df = (
            df[df["user_id"] == user_id]
            .sort_values(by="timestamp", ascending=False)
            .head(self.max_logs)
        )
        if df.empty:
            return np.zeros(self.user_dim, dtype=np.float32)

        weighted_embs = []
        for _, row in df.iterrows():
            cid = row["content_id"]
            ts = pd.to_datetime(row["timestamp"])
            # DB에 임베딩이 있으면 파싱, 없으면 cfg 기반 임베더 사용
            series = self.all_contents_df.loc[
                self.all_contents_df["id"] == cid, "embedding"
            ]
            if not series.empty and isinstance(series.iloc[0], str):
                try:
                    emb = np.array(ast.literal_eval(series.iloc[0]), dtype=np.float32)
                except Exception:
                    emb = np.zeros(self.user_dim, dtype=np.float32)
            else:
                embeder_cfg = self.cfg["embedder"]
                embeder = make(embeder_cfg["type"], **embeder_cfg["params"])
                emb = embeder.embed_content(row)

            # 시간 차이 기반 가중치 계산 (시간 단위: 시간 차)
            hours = (current_time - ts).total_seconds() / 3600.0
            weight = self.time_decay_factor**hours
            weighted_embs.append(emb * weight)

        avg_emb = np.mean(weighted_embs, axis=0)
        # 차원 맞춤 (패딩 또는 자르기)
        if avg_emb.shape[0] < self.user_dim:
            avg_emb = np.pad(
                avg_emb, (0, self.user_dim - avg_emb.shape[0]), constant_values=0.0
            )
        else:
            avg_emb = avg_emb[: self.user_dim]

        return avg_emb.astype(np.float32)
