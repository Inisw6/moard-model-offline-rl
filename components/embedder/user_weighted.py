import logging
import time
from typing import Optional

import numpy as np
from components.core.base import BaseUserEmbedder
from components.database.db_utils import get_contents
from components.registry import register


@register("weighted_user")
class WeightedUserEmbedder(BaseUserEmbedder):
    """가중치 기반 사용자 임베딩 추출기.

    - 사용자의 최근 로그에 시간 기반 가중치를 적용
    - 가중 평균을 통해 ratio, time, 콘텐츠 타입별 선호도를 계산
    - 벡터 구성: [가중 평균 ratio, 가중 평균 time, 콘텐츠 타입별 가중 비율, 0 패딩]
    """

    def __init__(
        self,
        user_dim: int = 30,
        time_decay_factor: float = 0.1,
        max_logs: int = 100,
        all_contents_df: Optional[object] = None,
    ) -> None:
        """WeightedUserEmbedder 생성자.

        Args:
            user_dim (int): 출력할 유저 임베딩 벡터의 차원.
            time_decay_factor (float): 시간 감쇠 계수 (0~1, 클수록 최근 로그 가중치 높음).
            max_logs (int): 고려할 최대 로그 개수.
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

        # 시간 가중치 관련 설정
        self.time_decay_factor = time_decay_factor
        self.max_logs = max_logs

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

    def output_dim(self):
        """유저 임베딩 벡터의 차원을 반환합니다.

        Returns:
            int: 유저 임베딩 벡터의 차원.
        """
        return self.user_dim

    def _calculate_time_weight(self, log_time: float, current_time: float) -> float:
        """시간 기반 가중치를 계산합니다.

        Args:
            log_time (float): 로그 시간 (timestamp).
            current_time (float): 현재 시간 (timestamp).

        Returns:
            float: 시간 가중치 (0~1).
        """
        time_diff = current_time - log_time
        # 시간 차이에 따른 지수 감쇠 (1시간 단위로 정규화)
        weight = np.exp(-self.time_decay_factor * time_diff / 3600)
        return max(0.0, min(1.0, weight))

    def _get_content_type_by_id(self, content_id: int) -> str:
        """콘텐츠 ID로 콘텐츠 타입을 조회합니다.

        Args:
            content_id (int): 콘텐츠 ID.

        Returns:
            str: 콘텐츠 타입.
        """
        if self.all_contents_df.empty:
            return "unknown"

        content_row = self.all_contents_df[self.all_contents_df["id"] == content_id]
        if content_row.empty:
            return "unknown"

        return str(content_row.iloc[0].get("type", "unknown")).lower()

    def embed_user(self, user: dict) -> np.ndarray:
        """사용자의 최근 로그 및 활동을 가중치 기반으로 벡터화합니다.

        Args:
            user (dict): {
                "user_info": 사용자 정보,
                "recent_logs": [로그 딕셔너리 목록],
                "current_time": datetime 객체 (선택)
            }

        Returns:
            np.ndarray: (user_dim,) 크기의 벡터.
                [0] = 가중 평균 ratio
                [1] = 가중 평균 time
                [2:2+N] = 콘텐츠 타입별 가중 비율
                [나머지] = 0 패딩
        """
        logs = user.get("recent_logs", [])
        if not logs:
            # 로그가 없으면 전부 0 벡터 반환
            return np.zeros(self.user_dim, dtype=np.float32)

        # 최대 로그 개수 제한 (최신 로그부터)
        if len(logs) > self.max_logs:
            logs = logs[-self.max_logs :]

        # 현재 시간 설정
        current_time = user.get("current_time")
        if current_time is not None:
            current_timestamp = current_time.timestamp()
        else:
            current_timestamp = time.time()

        # 가중치 계산을 위한 변수들
        weighted_ratio_sum = 0.0
        weighted_time_sum = 0.0
        total_weight = 0.0

        # 콘텐츠 타입별 가중 카운트
        weighted_type_counts = {t: 0.0 for t in self.content_types}

        for log in logs:
            # 로그 정보 추출
            log_ratio = log.get("ratio", 0.0)
            log_time = log.get("time", current_timestamp)
            content_id = log.get("content_id")

            # 시간 기반 가중치 계산
            time_weight = self._calculate_time_weight(log_time, current_timestamp)

            # 사용자 행동 기반 가중치 (ratio 활용)
            behavior_weight = max(0.1, log_ratio)  # 최소 가중치 0.1

            # 전체 가중치 = 시간 가중치 × 행동 가중치
            log_weight = time_weight * behavior_weight

            if log_weight > 0:
                # 가중 합계에 추가
                weighted_ratio_sum += log_ratio * log_weight
                weighted_time_sum += log_time * log_weight
                total_weight += log_weight

                # 콘텐츠 타입별 가중 카운트
                if content_id is not None:
                    content_type = self._get_content_type_by_id(content_id)
                    if content_type in self.type_to_idx_map:
                        weighted_type_counts[content_type] += log_weight
                else:
                    # content_id가 없으면 content_actual_type 사용
                    actual_type = str(log.get("content_actual_type", "")).lower()
                    if actual_type in self.type_to_idx_map:
                        weighted_type_counts[actual_type] += log_weight

        if total_weight == 0:
            # 유효한 가중치가 없으면 0 벡터 반환
            return np.zeros(self.user_dim, dtype=np.float32)

        # 가중 평균 계산
        weighted_ratio_avg = weighted_ratio_sum / total_weight
        weighted_time_avg = weighted_time_sum / total_weight

        # 콘텐츠 타입별 가중 비율 계산
        total_weighted_type_count = sum(weighted_type_counts.values())
        type_vec = np.array(
            [
                (
                    (weighted_type_counts[t] / total_weighted_type_count)
                    if total_weighted_type_count > 0
                    else 0.0
                )
                for t in self.content_types
            ]
        )

        # [가중 평균 ratio, 가중 평균 time]과 타입 가중 비율 벡터를 연결
        vec = np.concatenate([[weighted_ratio_avg, weighted_time_avg], type_vec])

        if len(vec) < self.user_dim:
            # 벡터가 작으면 0으로 패딩
            vec = np.pad(vec, (0, self.user_dim - len(vec)), "constant")
        elif len(vec) > self.user_dim:
            # 벡터가 크면 잘라냄
            vec = vec[: self.user_dim]

        return vec.astype(np.float32)

    def estimate_preference(self, state: np.ndarray) -> dict:
        """유저 임베딩 벡터에서 콘텐츠 타입별 선호도를 추정합니다.

        Args:
            state (np.ndarray): 길이가 최소 2+N인 임베딩 벡터.

        Returns:
            dict: {타입명: 선호도(float)}.
        """
        if len(state) < 2 + self.num_content_types:
            return {t: 0.0 for t in self.content_types}

        # 2:2+N 구간이 타입별 가중 비율 정보
        type_prefs = state[2 : 2 + self.num_content_types]

        # 추가적으로 ratio와 time 정보도 활용
        weighted_ratio = state[0] if len(state) > 0 else 0.0

        # ratio가 높을수록 전체적인 선호도 증폭
        amplification_factor = 1.0 + (weighted_ratio * 0.5)

        return {
            self.content_types[i]: float(type_prefs[i] * amplification_factor)
            for i in range(len(type_prefs))
        }
