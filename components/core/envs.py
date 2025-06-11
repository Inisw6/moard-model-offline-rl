import logging
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from components.core.base import BaseEnv
from components.registry import register
from components.database.db_utils import get_contents, get_user_logs, get_users
from components.simulation.simulators import BaseResponseSimulator


@register("rec_env")
class RecEnv(gym.Env, BaseEnv):
    """추천 환경(RecEnv).

    Gymnasium과 사용자 정의 BaseEnv를 함께 상속하여,
    RL 기반 추천 시스템 시뮬레이션을 위한 환경을 제공합니다.

    Attributes:
        context: 추천 컨텍스트 관리자.
        cold_start (int): 콜드스타트 상태 사용 여부.
        max_steps (int): 에피소드 당 최대 추천 횟수.
        top_k (int): 한 스텝에서 추천할 콘텐츠 개수.
        embedder: 사용자/콘텐츠 임베딩 객체.
        candidate_generator: 추천 후보군 생성 객체.
        reward_fn: 보상 함수 객체.
        response_simulator (BaseResponseSimulator): 사용자 반응 시뮬레이터.
        step_count (int): 에피소드 내 현재 스텝 카운트.
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
        response_simulator: BaseResponseSimulator,
        user_id: Optional[int] = None,
        debug: bool = False,
    ) -> None:
        """RecEnv 환경을 초기화합니다.

        Args:
            cold_start (int): 콜드스타트 상태 사용 여부.
            max_steps (int): 에피소드 당 최대 추천 횟수.
            top_k (int): 콘텐츠 추천 수.
            embedder: 사용자/콘텐츠 임베딩 객체.
            candidate_generator: 추천 후보군 생성 객체.
            reward_fn: 보상 함수 객체.
            context: 추천 컨텍스트 관리자.
            response_simulator (BaseResponseSimulator): 사용자 반응 시뮬레이터.
            user_id (Optional[int], optional): 환경에 할당할 사용자 ID. None이면 임의 선택.
            debug (bool, optional): 디버깅 모드 활성화 여부.
        """
        super().__init__()
        # 인자 없으면 예외 발생
        assert embedder is not None, "Embedder must be provided"
        assert candidate_generator is not None, "Candidate generator must be provided"
        assert reward_fn is not None, "Reward function must be provided"
        assert context is not None, "Context manager must be provided"
        assert response_simulator is not None, "Response simulator must be provided"

        # 속성 설정
        self.context = context
        self.cold_start = cold_start
        self.max_steps = max_steps
        self.top_k = top_k
        self.embedder = embedder
        self.candidate_generator = candidate_generator
        self.reward_fn = reward_fn
        self.current_query = None
        self.response_simulator = response_simulator
        self.step_count = 0

        # DB에서 DataFrame 로드
        self._load_dataframes()

        # 사용자 정보 초기화
        self._init_user(user_id)

        # observation / action space 초기화
        self._init_spaces()

    def _load_dataframes(self) -> None:
        """DB에서 사용자 및 콘텐츠, 로그 DataFrame을 불러와 인스턴스 변수로 저장합니다."""
        self.all_users_df = get_users()
        self.all_user_logs_df = get_user_logs()
        self.all_contents_df = get_contents()

    def _init_user(self, user_id: Optional[int]) -> None:
        """에피소드에 사용할 사용자 ID와 사용자 정보를 초기화합니다.

        Args:
            user_id (Optional[int]): 환경에 할당할 사용자 ID.
        """
        self.current_user_original_logs_df = pd.DataFrame()
        self.current_session_simulated_logs = []

        if user_id is None:
            if not self.all_users_df.empty:
                self.current_user_id = self.all_users_df.iloc[0]["id"]
            else:
                self.current_user_id = -1
                logging.warning("No users found in DB. Using dummy user_id = -1.")
        else:
            self.current_user_id = user_id

        if self.current_user_id != -1 and not self.all_users_df.empty:
            series = self.all_users_df[self.all_users_df["id"] == self.current_user_id]
            if not series.empty:
                self.current_user_info = series.iloc[0].to_dict()
            else:
                logging.warning(
                    "User ID %d not found. Using dummy user_info.", self.current_user_id
                )
                self.current_user_info = {
                    "id": self.current_user_id,
                    "uuid": "dummy_user_not_found",
                }
        else:
            self.current_user_info = {"id": -1, "uuid": "dummy_user"}

    def _init_spaces(self) -> None:
        """observation_space와 action_space를 정의합니다."""
        state_dim = self.embedder.output_dim()
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        MAX_CANDIDATE_INDEX = 24
        self._action_space = spaces.Tuple(
            [
                spaces.Tuple(
                    (
                        spaces.Discrete(len(self.embedder.content_types)),
                        spaces.Discrete(MAX_CANDIDATE_INDEX),
                    )
                )
                for _ in range(self.top_k)
            ]
        )

    def _set_current_user_info(self, user_id: Optional[int]) -> None:
        """사용자 ID를 기반으로 환경 내 현재 사용자 정보를 설정합니다.

        Args:
            user_id (Optional[int]): 환경에 할당할 사용자 ID.
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
                    "User ID %d not found. Using dummy user_info.", self.current_user_id
                )
                self.current_user_info = {
                    "id": self.current_user_id,
                    "uuid": "dummy_user_not_found",
                }
        elif self.current_user_id == -1:
            self.current_user_info = {"id": -1, "uuid": "dummy_user"}

    def _merge_logs_with_content_type(
        self, base_logs_df: pd.DataFrame, simulated_logs_list: List[Dict]
    ) -> pd.DataFrame:
        """실제 로그와 시뮬레이션 로그를 병합한 후 콘텐츠 타입 정보를 병합합니다.

        Args:
            base_logs_df (pd.DataFrame): 원본 사용자 로그.
            simulated_logs_list (List[Dict]): 현재 에피소드의 시뮬레이션 로그 리스트.

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
        self, base_logs_df: pd.DataFrame, simulated_logs_list: List[Dict]
    ) -> Dict:
        """사용자 임베딩에 필요한 dict 데이터를 생성합니다.

        Args:
            base_logs_df (pd.DataFrame): 원본 사용자 로그.
            simulated_logs_list (List[Dict]): 시뮬레이션 로그 리스트.

        Returns:
            Dict: embed_user 함수 입력 포맷의 사용자 데이터.
        """
        logs_df = self._merge_logs_with_content_type(base_logs_df, simulated_logs_list)
        processed_logs = logs_df.to_dict("records") if not logs_df.empty else []
        return {
            "user_info": self.current_user_info,
            "recent_logs": processed_logs,
            "current_time": datetime.now(timezone.utc),
        }

    def _select_contents_from_action(
        self, cand_dict: Dict, action_list: List[Tuple[str, int]]
    ) -> List[Dict]:
        """액션 리스트에서 실제 추천할 콘텐츠들을 추출합니다.

        Args:
            cand_dict (Dict): 추천 후보군 {타입: [콘텐츠, ...]}.
            action_list (List[Tuple[str, int]]): [(콘텐츠 타입, 후보 인덱스), ...] 리스트.

        Returns:
            List[Dict]: 선택된 콘텐츠 리스트.
        """
        selected_contents = []
        for ctype, cand_idx in action_list:
            if ctype in cand_dict and len(cand_dict[ctype]) > cand_idx:
                selected_contents.append(cand_dict[ctype][cand_idx])
            else:
                logging.warning(
                    "Invalid action (%s, %d). Candidate not found.", ctype, cand_idx
                )
        return selected_contents

    def _create_simulated_log_entry(
        self, content: Dict, event_type: str, dwell_time: Optional[int] = None
    ) -> Dict:
        """시뮬레이션용 로그 엔트리를 생성합니다.

        Args:
            content (Dict): 추천된 콘텐츠 정보.
            event_type (str): "VIEW" 또는 "CLICK".
            dwell_time (Optional[int], optional): None이면 VIEW→0, CLICK→랜덤(60~600).

        Returns:
            Dict: user_logs 포맷의 단일 로그 엔트리.
        """
        # 1) 콘텐츠 ID/타입 조회 (한 번만)
        content_id = content["id"]
        content_type = content["type"]

        # 2) 클릭 여부 플래그
        is_click = event_type == "CLICK"

        # 3) 체류 시간 결정
        if dwell_time is None:
            time_seconds = random.randint(60, 600) if is_click else 0
        else:
            time_seconds = dwell_time

        # 4) 클릭 확률 비율 산출
        ratio = 1.0 if is_click else 0.1 + 0.8 * random.random()

        # 5) 타임스탬프
        timestamp = datetime.now(timezone.utc).isoformat()

        return {
            "user_id": self.current_user_id,
            "content_id": content_id,
            "event_type": event_type,
            "timestamp": timestamp,
            "content_actual_type": content_type,
            "ratio": ratio,
            "time": time_seconds,
        }

    @property
    def observation_space(self) -> spaces.Box:
        """관찰(observation) 벡터의 공간 분포를 반환합니다.

        Returns:
            spaces.Box: 상태 공간 객체.
        """
        return self._observation_space

    @property
    def action_space(self) -> spaces.Tuple:
        """행동(action) 공간 분포를 반환합니다.

        Returns:
            spaces.Tuple: 액션 공간 객체.
        """
        return self._action_space

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """환경을 초기화합니다. (에피소드 시작)

        Args:
            seed (Optional[int], optional): 랜덤 시드.
            options (Optional[Dict], optional): 추가 옵션.

        Returns:
            Tuple[np.ndarray, Dict]: 초기 상태 벡터와 기타 정보.
        """
        if options and "query" in options:
            self.current_query = options["query"]
        else:
            self.current_query = None

        self.step_count = 0
        self.current_session_simulated_logs.clear()
        self._set_current_user_info(options.get("user_id", None))

        user_initial_data = self._get_user_data_for_embedding(
            self.current_user_original_logs_df, []
        )
        state = self.embedder.embed_user(user_initial_data)
        return state, {}

    def step(
        self, action_list: List[Tuple[str, int]]
    ) -> tuple[np.ndarray, float, bool, bool, Dict]:
        """환경에 액션 리스트(top-k 추천)를 적용하고, 다음 상태 및 보상 등을 반환합니다.

        Args:
            action_list (List[Tuple[str, int]]): [(콘텐츠 타입, 후보 인덱스), ...] top-k 개의 액션.

        Returns:
            Tuple:
                - 다음 상태 (np.ndarray)
                - 보상 (float)
                - done (bool): 에피소드 종료 여부
                - truncated (bool): 트렁케이트 여부(사용 안함)
                - info (Dict): 기타 정보
        """
        self.step_count += 1

        # 1) 현재 사용자 상태(user_state)를 구한다.
        user_current_data = self._get_user_data_for_embedding(
            self.current_user_original_logs_df,
            self.current_session_simulated_logs,
        )
        user_state = self.embedder.embed_user(user_current_data)

        # 2) 후보 생성
        cand_dict = self.candidate_generator.get_candidates(self.current_query)

        # 3) 액션 리스트에 따라 실제 추천 콘텐츠들 선택 (top-k)
        selected_contents = self._select_contents_from_action(cand_dict, action_list)
        if not selected_contents:
            logging.warning("No valid contents selected from actions %s", action_list)
            return user_state, 0.0, True, False, {}

        # 4) 사용자 반응 시뮬레이션
        sim_context = {
            "step_count": self.step_count,
            "session_logs": self.current_session_simulated_logs,
        }
        all_responses = self.response_simulator.simulate_responses(
            selected_contents, sim_context
        )

        # 5) 보상 계산: 모든 응답을 사용하여 보상 계산
        total_reward, individual_reward_dic = (
            self.reward_fn.calculate_from_topk_responses(all_responses=all_responses)
        )

        # 6) 시뮬레이션 로그 생성 및 추가 (클릭한 콘텐츠들만)
        for response in all_responses:
            if response["clicked"]:
                # 클릭한 콘텐츠 찾기
                clicked_content = None
                for content in selected_contents:
                    if content.get("id") == int(response["content_id"]):
                        clicked_content = content
                        break

                if clicked_content:
                    event_type = "CLICK"
                    new_log_entry = self._create_simulated_log_entry(
                        clicked_content, event_type, response["dwell_time"]
                    )
                    self.current_session_simulated_logs.append(new_log_entry)

        # 7) 다음 상태 계산
        user_next_data = self._get_user_data_for_embedding(
            self.current_user_original_logs_df,
            self.current_session_simulated_logs,
        )
        next_state = self.embedder.embed_user(user_next_data)

        # 8) 에피소드 종료 판단
        done = self.step_count >= self.max_steps

        # 9) 컨텍스트 매니저에도 한 스텝 진행 시그널 전달
        self.context.step()

        # 10) 최종 결과 반환
        info = {
            "all_responses": all_responses,
            "selected_contents": selected_contents,
            "total_clicks": sum(1 for r in all_responses if r["clicked"]),
            "individual_rewards": individual_reward_dic,
        }
        return next_state, total_reward, done, False, info

    def get_candidates(self) -> Dict[str, List[Any]]:
        """현 상태에서 추천 후보군을 반환합니다.

        Returns:
            Dict[str, List[Any]]: {타입: 후보 콘텐츠 리스트}
        """
        return self.candidate_generator.get_candidates(self.current_query)
