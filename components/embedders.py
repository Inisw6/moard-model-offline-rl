import numpy as np
import json
import logging

from components.base import BaseEmbedder
from components.registry import register
from .db_utils import get_contents


# 추후 분리 및 임베더 연산 관련 개선
@register("simple_concat")
class SimpleConcatEmbedder(BaseEmbedder):
    """
    사용자의 최근 로그/콘텐츠 속성을 단순 연결하여 벡터로 변환하는 임베더.

    - user 벡터: [평균 ratio, 평균 time, 각 콘텐츠 타입별 비율, (0 padding)] (크기: user_dim)
    - content 벡터: [사전학습 임베딩, 콘텐츠 타입 원핫, (0 padding)] (크기: content_dim)
    """

    def __init__(self, user_dim: int = 30, content_dim: int = 5):
        """
        Args:
            user_dim (int): 반환할 user 임베딩 벡터 차원 (2+타입수 이상 추천)
            content_dim (int): 반환할 content 임베딩 벡터 차원 (타입수 이상 필요)
        """
        self.all_contents_df = get_contents()
        if not self.all_contents_df.empty:
            self.content_types = self.all_contents_df["type"].unique().tolist()
        else:
            self.content_types = ["youtube", "blog", "news"]
        self.num_content_types = len(self.content_types)
        self.type_to_idx_map = {t: i for i, t in enumerate(self.content_types)}

        # content_dim에서 타입 원핫 인코딩 차원 분리
        self.pretrained_content_embedding_dim = content_dim - self.num_content_types
        if self.pretrained_content_embedding_dim < 0:
            logging.warning(
                "content_dim (%d) is too small for %d content types. "
                "Setting pretrained_content_embedding_dim to 0. content_dim will be %d.",
                content_dim,
                self.num_content_types,
                self.num_content_types,
            )
            self.pretrained_content_embedding_dim = 0
            self.content_dim = self.num_content_types
        else:
            self.content_dim = content_dim

        min_user_dim = 2 + self.num_content_types
        if user_dim < min_user_dim:
            print(
                f"Warning: user_dim ({user_dim}) is too small. Adjusting to {min_user_dim}..."
            )
            self.user_dim = min_user_dim
        else:
            self.user_dim = user_dim

    def output_dim(self) -> int:
        """임베딩되는 user 벡터의 차원 반환"""
        return self.user_dim

    def embed_user(self, user: dict) -> np.ndarray:
        """
        사용자의 최근 로그 및 활동을 벡터로 변환.

        Args:
            user (dict):
                - user_info: (dict) 사용자 정보, 예: {"id": 123, ...}
                - recent_logs: (list of dict) [{"ratio": float, "time": float, "content_actual_type": str, ...}, ...]
                - current_time: (datetime 등)

        Returns:
            np.ndarray: (user_dim,)
                [0] = 평균 ratio,
                [1] = 평균 time,
                [2:2+N] = 각 콘텐츠 타입별 비율,
                [이후] = 0 padding
        """
        logs = user.get("recent_logs", [])
        if not logs:
            return np.zeros(self.user_dim, dtype=np.float32)

        ratio_avg = np.mean([l.get("ratio", 0.0) for l in logs])
        time_avg = np.mean([l.get("time", 0.0) for l in logs])

        type_counts = {t: 0 for t in self.content_types}
        for log in logs:
            actual_content_type = str(log.get("content_actual_type", "")).lower()
            if actual_content_type in self.type_to_idx_map:
                type_counts[actual_content_type] += 1

        total_logs_for_known_types = sum(type_counts.values())
        type_vec = np.array(
            [
                (
                    type_counts[t] / total_logs_for_known_types
                    if total_logs_for_known_types > 0
                    else 0
                )
                for t in self.content_types
            ]
        )

        vec = np.concatenate([[ratio_avg, time_avg], type_vec])
        if len(vec) < self.user_dim:
            vec = np.pad(vec, (0, self.user_dim - len(vec)), "constant")
        elif len(vec) > self.user_dim:
            vec = vec[: self.user_dim]
        return vec.astype(np.float32)

    def embed_content(self, content: dict) -> np.ndarray:
        """
        콘텐츠 정보를 임베딩 벡터로 변환.

        Args:
            content (dict):
                - embedding: (str, JSON리스트) 사전학습 임베딩값(str)
                - type: (str) 콘텐츠 타입

        Returns:
            np.ndarray: (content_dim,)
                [0:N] = 사전학습 임베딩 (혹은 0 padding),
                [N:] = 타입 원핫 인코딩,
                부족시 0 padding
        """
        pretrained_emb = np.array([], dtype=np.float32)
        if self.pretrained_content_embedding_dim > 0:
            try:
                embedding_str = content.get("embedding")
                if embedding_str is None or embedding_str == "":
                    pretrained_emb = np.zeros(
                        self.pretrained_content_embedding_dim, dtype=np.float32
                    )
                else:
                    parsed_list = json.loads(embedding_str)
                    if not isinstance(parsed_list, list) or not all(
                        isinstance(x, (int, float)) for x in parsed_list
                    ):
                        pretrained_emb = np.zeros(
                            self.pretrained_content_embedding_dim, dtype=np.float32
                        )
                    else:
                        pretrained_emb_list = np.array(parsed_list, dtype=np.float32)
                        if (
                            len(pretrained_emb_list)
                            != self.pretrained_content_embedding_dim
                        ):
                            pretrained_emb = np.zeros(
                                self.pretrained_content_embedding_dim, dtype=np.float32
                            )
                        else:
                            pretrained_emb = pretrained_emb_list
            except (json.JSONDecodeError, ValueError):
                pretrained_emb = np.zeros(
                    self.pretrained_content_embedding_dim, dtype=np.float32
                )
        else:
            pretrained_emb = np.array([], dtype=np.float32)

        content_type_str = content.get("type", "").lower()
        type_idx = self.type_to_idx_map.get(content_type_str, -1)
        type_onehot = np.zeros(self.num_content_types, dtype=np.float32)
        if type_idx != -1:
            type_onehot[type_idx] = 1.0

        final_vec = np.concatenate([pretrained_emb, type_onehot])
        if len(final_vec) != self.content_dim:
            if len(final_vec) < self.content_dim:
                final_vec = np.pad(
                    final_vec, (0, self.content_dim - len(final_vec)), "constant"
                )
            else:
                final_vec = final_vec[: self.content_dim]

        return final_vec.astype(np.float32)

    def estimate_preference(self, state: np.ndarray) -> dict:
        """
        유저 임베딩 벡터에서 콘텐츠 타입별 선호도를 추정.

        Args:
            state (np.ndarray): (user_dim,)
                [2:2+N] 영역에 각 타입별 비율/선호도가 저장되어 있음

        Returns:
            dict: {타입명: 선호도(float)}
        """
        if len(state) < 2 + self.num_content_types:
            return {t: 0.0 for t in self.content_types}
        type_prefs = state[2 : 2 + self.num_content_types]
        return {
            self.content_types[i]: float(type_prefs[i]) for i in range(len(type_prefs))
        }
