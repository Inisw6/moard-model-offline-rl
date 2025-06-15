import json
import logging
from typing import Optional

import numpy as np

from components.core.base import BaseContentEmbedder, BaseUserEmbedder
from components.database.db_utils import get_contents
from components.registry import register


@register("simple_user")
class SimpleUserEmbedder(BaseUserEmbedder):
    """사용자 로그 기반 단순 임베딩 추출기.

    - 사용자의 최근 로그와 활동을 단순 연결하여 고정 차원 벡터로 변환합니다.
    - 벡터 구성: [평균 ratio, 평균 time, 콘텐츠 타입별 비율, 0 패딩]
    """

    def __init__(
        self,
        user_dim: int = 30,
        all_contents_df: Optional[object] = None,
    ) -> None:
        """SimpleUserEmbedder 생성자.

        Args:
            user_dim (int): 출력할 유저 임베딩 벡터의 차원.
            all_contents_df (Optional[pandas.DataFrame]): 테스트용 외부 콘텐츠 DataFrame.
        """
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
        # 콘텐츠 타입을 인덱스로 매핑
        self.type_to_idx_map = {t: i for i, t in enumerate(self.content_types)}

        # 최소 user_dim = 2 (ratio, time) + 콘텐츠 타입 개수
        min_user_dim = 2 + self.num_content_types
        if user_dim < min_user_dim:
            logging.warning(
                "user_dim (%d)이 너무 작습니다. %d로 조정합니다...",
                user_dim,
                min_user_dim,
            )
            self.user_dim = min_user_dim
        else:
            self.user_dim = user_dim

    def output_dim(self) -> int:
        """유저 임베딩 벡터의 차원을 반환합니다.

        Returns:
            int: 유저 임베딩 벡터의 차원.
        """
        return self.user_dim

    def embed_user(self, user: dict) -> np.ndarray:
        """사용자의 최근 로그 및 활동을 벡터로 변환합니다.

        Args:
            user (dict): {
                "user_info": 사용자 정보,
                "recent_logs": [로그 딕셔너리 목록],
                "current_time": datetime 객체 (선택)
            }

        Returns:
            np.ndarray: (user_dim,) 크기의 벡터.
                [0] = 최근 로그의 평균 ratio
                [1] = 최근 로그의 평균 time
                [2:2+N] = 콘텐츠 타입별 비율
                [나머지] = 0 패딩
        """
        logs = user.get("recent_logs", [])
        if not logs:
            # 로그가 없으면 전부 0 벡터 반환
            return np.zeros(self.user_dim, dtype=np.float32)

        # 로그 내 ratio와 time의 평균 계산
        ratio_avg = np.mean([l.get("ratio", 0.0) for l in logs])
        time_avg = np.mean([l.get("time", 0.0) for l in logs])

        # 콘텐츠 타입별 카운트 초기화
        type_counts = {t: 0 for t in self.content_types}
        for log in logs:
            actual_type = str(log.get("content_actual_type", "")).lower()
            if actual_type in self.type_to_idx_map:
                type_counts[actual_type] += 1

        total_known = sum(type_counts.values())
        # 각 타입별 비율 벡터 생성
        type_vec = np.array(
            [
                (type_counts[t] / total_known) if total_known > 0 else 0
                for t in self.content_types
            ]
        )

        # [평균 ratio, 평균 time]과 타입 비율 벡터를 연결
        vec = np.concatenate([[ratio_avg, time_avg], type_vec])
        if len(vec) < self.user_dim:
            # 벡터가 작으면 0으로 패딩
            vec = np.pad(vec, (0, self.user_dim - len(vec)), "constant")
        elif len(vec) > self.user_dim:
            # 벡터가 크면 잘라냄
            vec = vec[: self.user_dim]

        return vec.astype(np.float32)

@register("simple_content")
class SimpleContentEmbedder(BaseContentEmbedder):
    """사전 임베딩 + 타입 원핫 기반 단순 콘텐츠 임베더.

    - 사전 저장된 임베딩(JSON)과 콘텐츠 타입을 결합해 벡터 생성.
    - 벡터: [사전 임베딩, 타입 원핫, 0 패딩].
    """

    def __init__(
        self,
        content_dim: int = 5,
        all_contents_df: Optional[object] = None,  # 의존성 주입 가능
    ) -> None:
        """SimpleContentEmbedder 생성자.

        Args:
            content_dim (int): 출력 임베딩 차원 (타입 개수 이상이어야 함).
            all_contents_df (Optional[pandas.DataFrame]): 외부 콘텐츠 DataFrame.
        """
        # 의존성 주입된 DataFrame이 없다면 실제 DB에서 가져옴
        self.all_contents_df = (
            all_contents_df if all_contents_df is not None else get_contents()
        )
        if not self.all_contents_df.empty:
            self.content_types = self.all_contents_df["type"].unique().tolist()
        else:
            self.content_types = ["youtube", "blog", "news"]

        self.num_content_types = len(self.content_types)
        self.type_to_idx_map = {t: i for i, t in enumerate(self.content_types)}

        # 사전학습 임베딩을 위한 차원 확보
        self.pretrained_content_embedding_dim = content_dim - self.num_content_types
        if self.pretrained_content_embedding_dim < 0:
            logging.warning(
                "content_dim (%d)이 콘텐츠 타입 수(%d)보다 작습니다. "
                "사전학습 임베딩 차원을 0으로 설정하고 content_dim을 %d로 사용합니다.",
                content_dim,
                self.num_content_types,
                self.num_content_types,
            )
            self.pretrained_content_embedding_dim = 0
            self.content_dim = self.num_content_types
        else:
            self.content_dim = content_dim

    def output_dim(self) -> int:
        """콘텐츠 임베딩 벡터의 차원을 반환합니다.

        Returns:
            int: 임베딩 벡터 차원.
        """
        return self.content_dim

    def embed_content(self, content: dict) -> np.ndarray:
        """콘텐츠의 사전 임베딩 + 타입 정보를 결합한 벡터를 생성합니다.

        Args:
            content (dict): {
                "embedding": str,  # JSON 숫자 리스트
                "type": str,       # 콘텐츠 타입
            }

        Returns:
            np.ndarray: (content_dim,) 크기의 벡터.
        """
        # 사전학습 임베딩 문자열 파싱
        if self.pretrained_content_embedding_dim > 0:
            try:
                embedding_str = content.get("embedding")
                if embedding_str is None or embedding_str == "":
                    # 임베딩이 없으면 0 벡터
                    pretrained_emb = np.zeros(
                        self.pretrained_content_embedding_dim, dtype=np.float32
                    )
                else:
                    parsed_list = json.loads(embedding_str)
                    # 리스트 형태 및 모든 요소가 숫자인지 확인
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
                            # 차원 불일치 시 0 벡터
                            pretrained_emb = np.zeros(
                                self.pretrained_content_embedding_dim, dtype=np.float32
                            )
                        else:
                            pretrained_emb = pretrained_emb_list
            except (json.JSONDecodeError, ValueError):
                # JSON 파싱 오류 시 0 벡터
                pretrained_emb = np.zeros(
                    self.pretrained_content_embedding_dim, dtype=np.float32
                )
        else:
            # 사전학습 임베딩 영역이 없으면 빈 배열
            pretrained_emb = np.array([], dtype=np.float32)

        # 콘텐츠 타입에 대한 원핫 인코딩 생성
        content_type_str = content.get("type", "").lower()
        type_idx = self.type_to_idx_map.get(content_type_str, -1)
        type_onehot = np.zeros(self.num_content_types, dtype=np.float32)
        if type_idx != -1:
            type_onehot[type_idx] = 1.0

        # 사전학습 임베딩 + 원핫 인코딩 연결
        final_vec = np.concatenate([pretrained_emb, type_onehot])
        if len(final_vec) != self.content_dim:
            if len(final_vec) < self.content_dim:
                # 벡터가 작으면 0으로 패딩
                final_vec = np.pad(
                    final_vec, (0, self.content_dim - len(final_vec)), "constant"
                )
            else:
                # 벡터가 크면 잘라냄
                final_vec = final_vec[: self.content_dim]

        return final_vec.astype(np.float32)
