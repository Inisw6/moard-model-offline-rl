import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random
import logging

from components.base import BaseEnv
from components.registry import register
from .db_utils import get_users, get_user_logs, get_contents
from datetime import datetime, timezone  # timezone 추가


@register("rec_env")
class RecEnv(gym.Env, BaseEnv):
    """
    추천 환경(RecEnv).
    Gymnasium과 사용자 정의 BaseEnv를 함께 상속하여,
    RL 기반 추천 시스템 시뮬레이션을 위한 환경을 제공합니다.
    """

    def __init__(
        self,
        cold_start: int,
        max_steps: int,
        top_k: int,
        embedder,
        candidate_generator,
        reward_fn,
        context,
        user_id: int | None = None,
        click_prob: float = 0.2,
    ) -> None:
        """
        환경을 초기화합니다.

        Args:
            cold_start (int): 콜드스타트 상태 사용 여부.
            max_steps (int): 에피소드 당 최대 추천 횟수.
            top_k (int): 각 타입별 후보군 최대 크기.
            embedder: 사용자/콘텐츠 임베딩 객체.
            candidate_generator: 추천 후보군 생성 객체.
            reward_fn: 보상 함수 객체.
            context: 추천 컨텍스트 관리자.
            user_id (int | None): 환경에 할당할 사용자 ID. None이면 임의 선택.
            click_prob (float): 추천 클릭 확률.
        """

        super().__init__()
        self.context = context
        self.max_steps = max_steps
        self.top_k = top_k
        self.embedder = embedder
        self.candidate_generator = candidate_generator
        self.reward_fn = reward_fn
        self.click_prob = click_prob  # 클릭 확률 파라미터 추가

        self.all_users_df = get_users()
        self.all_user_logs_df = get_user_logs()  # 모든 사용자의 모든 로그 (초기 로드용)
        self.all_contents_df = get_contents()

        self.current_user_id = None
        self.current_user_info = None
        self.current_user_original_logs_df = (
            pd.DataFrame()
        )  # 현재 사용자의 DB 로그 (리셋 시 설정)
        self.current_session_simulated_logs = (
            []
        )  # 현재 에피소드에서 시뮬레이션된 로그 [{dict}, ...]

        if user_id is None:
            if not self.all_users_df.empty:
                self.current_user_id = self.all_users_df.iloc[0]["id"]
            else:
                self.current_user_id = -1  # 더미 ID
                logging.warning(
                    "Warning: No users found in DB. Using dummy user_id = -1."
                )
        else:
            self.current_user_id = user_id

        if self.current_user_id != -1 and not self.all_users_df.empty:
            user_info_series = self.all_users_df[
                self.all_users_df["id"] == self.current_user_id
            ]
            if not user_info_series.empty:
                self.current_user_info = user_info_series.iloc[0].to_dict()
            else:
                logging.warning(
                    f"Warning: User ID {self.current_user_id} not found. Using dummy user_info."
                )
                self.current_user_info = {
                    "id": self.current_user_id,
                    "uuid": "dummy_user_not_found",
                }
        elif self.current_user_id == -1:
            self.current_user_info = {"id": -1, "uuid": "dummy_user"}

        state_dim = embedder.output_dim()
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self._action_space = spaces.Tuple(
            (
                spaces.Discrete(len(self.embedder.content_types)),
                spaces.Discrete(self.top_k),
            )
        )
        self.step_count = 0

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Tuple:
        return self._action_space

    def _set_current_user_info(self, user_id: int | None):
        """
        사용자 ID를 기반으로 환경 내 현재 사용자 정보를 설정합니다.
        사용자 정보가 없으면 더미 사용자를 등록합니다.

        Args:
            user_id (int | None): 환경에 할당할 사용자 ID.
        """
        if user_id is None:
            if not self.all_users_df.empty:
                self.current_user_id = self.all_users_df.iloc[0]["id"]
            else:
                self.current_user_id = -1
                logging.warning("No users found in DB. Using dummy user_id = -1.")
        else:
            self.current_user_id = user_id

        if self.current_user_id != -1 and not self.all_users_df.empty:
            user_info_series = self.all_users_df[
                self.all_users_df["id"] == self.current_user_id
            ]
            if not user_info_series.empty:
                self.current_user_info = user_info_series.iloc[0].to_dict()
            else:
                logging.warning(
                    f"User ID {self.current_user_id} not found. Using dummy user_info."
                )
                self.current_user_info = {
                    "id": self.current_user_id,
                    "uuid": "dummy_user_not_found",
                }
        elif self.current_user_id == -1:
            self.current_user_info = {"id": -1, "uuid": "dummy_user"}

    def _merge_logs_with_content_type(
        self, base_logs_df: pd.DataFrame, simulated_logs_list: list[dict]
    ) -> pd.DataFrame:
        """
        사용자의 실제 로그와 시뮬레이션 로그를 병합한 뒤,
        각 로그에 대해 콘텐츠 타입 정보를 병합(조인)합니다.

        Args:
            base_logs_df (pd.DataFrame): 원본 사용자 로그.
            simulated_logs_list (list[dict]): 현재 에피소드의 시뮬레이션 로그 리스트.

        Returns:
            pd.DataFrame: 콘텐츠 타입 정보가 병합된 전체 로그.
        """
        combined_logs_df = base_logs_df
        if simulated_logs_list:
            sim_logs_df = pd.DataFrame(simulated_logs_list)
            combined_logs_df = pd.concat([base_logs_df, sim_logs_df], ignore_index=True)
        if not combined_logs_df.empty and not self.all_contents_df.empty:
            if "content_actual_type" not in combined_logs_df.columns:
                combined_logs_df["content_actual_type"] = None
            merged_df = pd.merge(
                combined_logs_df,
                self.all_contents_df[["id", "type"]].rename(
                    columns={"id": "content_id", "type": "content_db_type"}
                ),
                on="content_id",
                how="left",
            )
            merged_df["content_actual_type"] = merged_df["content_actual_type"].fillna(
                merged_df["content_db_type"]
            )
            merged_df.drop(columns=["content_db_type"], inplace=True)
            return merged_df
        return combined_logs_df

    def _get_user_data_for_embedding(
        self, base_logs_df: pd.DataFrame, simulated_logs_list: list[dict]
    ) -> dict:
        """
        사용자 임베딩에 필요한 dict 데이터를 생성합니다.

        Args:
            base_logs_df (pd.DataFrame): 원본 사용자 로그.
            simulated_logs_list (list[dict]): 시뮬레이션 로그 리스트.

        Returns:
            dict: embed_user 함수 입력 포맷의 사용자 데이터.
        """
        logs_df = self._merge_logs_with_content_type(base_logs_df, simulated_logs_list)
        processed_logs = logs_df.to_dict("records") if not logs_df.empty else []
        return {
            "user_info": self.current_user_info,
            "recent_logs": processed_logs,
            "current_time": datetime.now(timezone.utc),
        }

    def _select_content_from_action(self, cand_dict: dict, action: tuple[int, int]):
        """
        액션 정보에서 실제 추천할 콘텐츠를 추출합니다.

        Args:
            cand_dict (dict): 추천 후보군 {타입: [콘텐츠, ...]}.
            action (tuple[int, int]): (콘텐츠 타입, 후보 인덱스).

        Returns:
            dict | None: 선택된 콘텐츠, 유효하지 않으면 None.
        """
        ctype, cand_idx = action
        if ctype in cand_dict and len(cand_dict[ctype]) > cand_idx:
            return cand_dict[ctype][cand_idx]
        return None

    def _sample_event_type(self) -> str:
        """
        클릭 확률(click_prob)에 따라 "VIEW" 또는 "CLICK" 이벤트를 샘플링합니다.

        Returns:
            str: "VIEW" 또는 "CLICK".
        """
        return "CLICK" if random.random() < self.click_prob else "VIEW"

    def _create_simulated_log_entry(self, content: dict, event_type: str) -> dict:
        """
        시뮬레이션용 로그 엔트리를 생성합니다.

        Args:
            content (dict): 추천된 콘텐츠 정보.
            event_type (str): 이벤트 타입 ("VIEW" 또는 "CLICK").

        Returns:
            dict: user_logs 포맷의 단일 로그 엔트리.
        """
        return {
            "user_id": self.current_user_id,
            "content_id": content.get("id"),
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content_actual_type": content.get("type"),
            "ratio": 1.0 if event_type == "CLICK" else random.uniform(0.1, 0.9),
            "time": (
                random.randint(60, 600)
                if event_type == "CLICK"
                else random.randint(5, 300)
            ),
        }

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """
        환경을 초기화합니다. (에피소드 시작)

        Args:
            seed (int | None): 랜덤 시드.
            options (dict | None): 추가 옵션.

        Returns:
            tuple[np.ndarray, dict]: 초기 상태 벡터, 기타 info.
        """
        user_initial_data = self._get_user_data_for_embedding(
            self.current_user_original_logs_df, []
        )
        state = self.embedder.embed_user(user_initial_data)
        return state, {}

    def step(
        self, action: tuple[int, int]
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        환경에 액션(추천)을 적용하고, 다음 상태 및 보상 등을 반환합니다.

        Args:
            action (tuple[int, int]): (콘텐츠 타입, 후보 인덱스)

        Returns:
            tuple:
                - 다음 상태 (np.ndarray)
                - 보상 (float)
                - done (bool): 에피소드 종료 여부
                - truncated (bool): 트렁케이트 여부(사용 안함)
                - info (dict): 기타 정보
        """
        self.step_count += 1
        cand_dict = self.candidate_generator.get_candidates(None)
        selected_content = self._select_content_from_action(cand_dict, action)

        if selected_content is None:
            logging.warning(f"Invalid action {action}. Candidate not found.")
            current_data = self._get_user_data_for_embedding(
                self.current_user_original_logs_df, self.current_session_simulated_logs
            )
            state = self.embedder.embed_user(current_data)
            return state, 0.0, True, False, {}

        simulated_event_type = self._sample_event_type()
        reward = self.reward_fn.calculate(
            selected_content, event_type=simulated_event_type
        )
        new_log_entry = self._create_simulated_log_entry(
            selected_content, simulated_event_type
        )
        self.current_session_simulated_logs.append(new_log_entry)

        user_next_data = self._get_user_data_for_embedding(
            self.current_user_original_logs_df, self.current_session_simulated_logs
        )
        next_state = self.embedder.embed_user(user_next_data)

        done = self.step_count >= self.max_steps
        self.context.step()
        return next_state, reward, done, False, {}

    def get_candidates(self, state: np.ndarray) -> dict:
        """
        현 상태에서 추천 후보군을 반환합니다.

        Args:
            state (np.ndarray): 현 상태 벡터

        Returns:
            dict: {콘텐츠 타입: 후보군 리스트}
        """
        return self.candidate_generator.get_candidates(state)
