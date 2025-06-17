import logging
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from components.core.base import BaseEnv
from components.registry import register
from components.simulation.random_simulator import BaseResponseSimulator


@register("rec_env")
class RecEnv(gym.Env, BaseEnv):
    """추천 환경 (RecEnv) 클래스입니다.

    Gymnasium과 BaseEnv를 상속하여 RL 기반 추천 시스템 시뮬레이션을 제공합니다.

    Attributes:
        max_steps (int): 에피소드 당 최대 스텝 수.
        top_k (int): 한 스텝에서 추천할 콘텐츠 개수.
        embedder: 사용자/콘텐츠 임베딩 객체.
        candidate_generator: 추천 후보군 생성 객체.
        reward_fn: 보상 함수 객체.
        response_simulator (BaseResponseSimulator): 사용자 반응 시뮬레이터.
        step_count (int): 현재 스텝 카운터.
    """

    def __init__(
        self,
        contents_df: pd.DataFrame,
        users_df: pd.DataFrame,
        logs_with_type_df: pd.DataFrame,
        max_steps: int,
        top_k: int,
        embedder,
        candidate_generator,
        reward_fn,
        response_simulator: BaseResponseSimulator,
        user_id: Optional[int] = None,
        debug: bool = False,
    ) -> None:
        """RecEnv 환경을 초기화합니다.

        Args:
            contents_df (pd.DataFrame): 사전에 로드된 전체 콘텐츠 데이터프레임.
            users_df (pd.DataFrame): 사전에 로드된 전체 사용자 데이터프레임.
            logs_with_type_df (pd.DataFrame): 콘텐츠 타입이 사전 병합된 전체 로그 데이터프레임.
            max_steps (int): 에피소드 당 최대 추천 횟수.
            top_k (int): 콘텐츠 추천 수.
            embedder: 사용자/콘텐츠 임베딩 객체.
            candidate_generator: 추천 후보군 생성 객체.
            reward_fn: 보상 함수 객체.
            response_simulator (BaseResponseSimulator): 사용자 반응 시뮬레이터.
            user_id (Optional[int], optional): 환경에 할당할 사용자 ID. None이면 임의 선택.
            debug (bool, optional): 디버깅 모드 활성화 여부.
        """
        super().__init__()
        assert embedder is not None, "Embedder must be provided"
        assert candidate_generator is not None, "Candidate generator must be provided"
        assert reward_fn is not None, "Reward function must be provided"
        assert response_simulator is not None, "Response simulator must be provided"

        self.max_steps = max_steps
        self.top_k = top_k
        self.embedder = embedder
        self.candidate_generator = candidate_generator
        self.reward_fn = reward_fn
        self.response_simulator = response_simulator
        self.current_query = None
        self.step_count = 0

        self.all_users_df = users_df
        self.all_user_logs_df = logs_with_type_df
        self.all_contents_df = contents_df

        self._init_user(user_id)
        self._init_spaces()

    def _init_user(self, user_id: Optional[int]) -> None:
        """에피소드에 사용할 사용자 ID와 사용자 정보를 초기화합니다.

        Args:
            user_id (Optional[int]): 환경에 할당할 사용자 ID.

        Returns:
            None
        """
        self.current_user_original_logs_df = pd.DataFrame()
        self.current_session_simulated_logs: List[Dict[str, Any]] = []

        if user_id is None and not self.all_users_df.empty:
            self.current_user_id = int(self.all_users_df.iloc[0]["id"])
        else:
            self.current_user_id = user_id if user_id is not None else -1

        if self.current_user_id != -1 and not self.all_users_df.empty:
            df = self.all_users_df[self.all_users_df["id"] == self.current_user_id]
            if not df.empty:
                self.current_user_info = df.iloc[0].to_dict()
            else:
                logging.warning(
                    "User ID %d not found. Using dummy_user_not_found.",
                    self.current_user_id,
                )
                self.current_user_info = {
                    "id": self.current_user_id,
                    "uuid": "dummy_user_not_found",
                }
        else:
            self.current_user_info = {"id": -1, "uuid": "dummy_user"}

    def _set_current_user_info(self, user_id: Optional[int]) -> None:
        """사용자 ID를 기반으로 current_user_id와 current_user_info를 초기화합니다.

        Args:
            user_id (Optional[int]): 환경에 할당할 사용자 ID.

        Returns:
            None
        """
        if user_id is None and not self.all_users_df.empty:
            self.current_user_id = int(self.all_users_df.iloc[0]["id"])
        else:
            self.current_user_id = user_id if user_id is not None else -1

        if self.current_user_id != -1 and not self.all_users_df.empty:
            df = self.all_users_df[self.all_users_df["id"] == self.current_user_id]
            if not df.empty:
                self.current_user_info = df.iloc[0].to_dict()
            else:
                logging.warning(
                    "User ID %d not found. Using dummy_user_not_found.",
                    self.current_user_id,
                )
                self.current_user_info = {
                    "id": self.current_user_id,
                    "uuid": "dummy_user_not_found",
                }
        else:
            self.current_user_info = {"id": -1, "uuid": "dummy_user"}

    def _init_spaces(self) -> None:
        """observation_space와 action_space를 정의합니다.

        Returns:
            None
        """
        state_dim = self.embedder.output_dim()
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32,
        )
        MAX_INDEX = 24
        single = spaces.Tuple(
            [
                spaces.Discrete(len(self.embedder.content_types)),
                spaces.Discrete(MAX_INDEX),
            ]
        )
        self._action_space = spaces.Tuple([single for _ in range(self.top_k)])

    def _merge_logs_with_content_type(
        self,
        base_logs_df: pd.DataFrame,
        simulated_logs_list: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """원본 로그와 시뮬레이션 로그를 콘텐츠 타입별로 병합하여 반환합니다.

        Args:
            base_logs_df (pd.DataFrame): 원본 로그 데이터프레임.
            simulated_logs_list (List[Dict[str, Any]]): 시뮬레이션 로그 리스트.

        Returns:
            pd.DataFrame: 병합된 로그 데이터프레임.
        """
        if not simulated_logs_list:
            return base_logs_df
        sim_df = pd.DataFrame(simulated_logs_list)
        if sim_df.empty:
            return base_logs_df
        if not self.all_contents_df.empty:
            sim_df = sim_df.merge(
                self.all_contents_df[["id", "type"]].rename(
                    columns={"id": "content_id", "type": "content_db_type"}
                ),
                on="content_id",
                how="left",
            )
            if "content_actual_type" not in sim_df:
                sim_df["content_actual_type"] = np.nan
            sim_df["content_actual_type"] = sim_df["content_actual_type"].fillna(
                sim_df["content_db_type"]
            )
            sim_df.drop(columns=["content_db_type"], inplace=True)
        if base_logs_df.empty:
            return sim_df
        return pd.concat([base_logs_df, sim_df], ignore_index=True)

    def _get_user_data_for_embedding(
        self,
        base_logs_df: pd.DataFrame,
        simulated_logs_list: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """임베딩을 위한 사용자 데이터를 생성합니다.

        Args:
            base_logs_df (pd.DataFrame): 원본 로그 데이터프레임.
            simulated_logs_list (List[Dict[str, Any]]): 시뮬레이션 로그 리스트.

        Returns:
            Dict[str, Any]: 임베딩에 사용할 사용자 데이터 딕셔너리.
        """
        merged_df = self._merge_logs_with_content_type(
            base_logs_df, simulated_logs_list
        )
        records = merged_df.to_dict("records") if not merged_df.empty else []
        return {
            "user_info": self.current_user_info,
            "recent_logs": records,
            "current_time": datetime.now(timezone.utc),
        }

    def _select_contents_from_action(
        self,
        cand_dict: Dict[str, List[Dict[str, Any]]],
        action_list: List[Tuple[str, int]],
    ) -> List[Dict[str, Any]]:
        """행동 리스트에 따라 추천할 콘텐츠를 선택합니다.

        Args:
            cand_dict (Dict[str, List[Dict[str, Any]]]): 타입별 후보 콘텐츠 딕셔너리.
            action_list (List[Tuple[str, int]]): (콘텐츠 타입, 인덱스) 쌍 리스트.

        Returns:
            List[Dict[str, Any]]: 선택된 콘텐츠 리스트.
        """
        selected_contents: List[Dict[str, Any]] = []
        for ctype, idx in action_list:
            items = cand_dict.get(ctype, [])
            if 0 <= idx < len(items):
                selected_contents.append(items[idx])
            else:
                logging.warning(
                    "Invalid action (%s, %d): Candidate not found.", ctype, idx
                )
        return selected_contents

    def _create_simulated_log_entry(
        self,
        content: Dict[str, Any],
        event_type: str,
        dwell_time: Optional[int] = None,
    ) -> Dict[str, Any]:
        """시뮬레이션용 단일 로그 엔트리를 생성합니다.

        Args:
            content (Dict[str, Any]): 콘텐츠 정보 딕셔너리.
            event_type (str): 이벤트 타입 (예: 'CLICK', 'VIEW').
            dwell_time (Optional[int], optional): 체류 시간(초). None이면 랜덤 생성.

        Returns:
            Dict[str, Any]: 생성된 로그 엔트리.
        """
        content_id = content["id"]
        content_type = content["type"]
        is_click = event_type == "CLICK"
        if dwell_time is None:
            time_seconds = random.randint(60, 600) if is_click else 0
        else:
            time_seconds = dwell_time
        ratio = 1.0 if is_click else 0.1 + 0.8 * random.random()
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
            spaces.Box: 관찰 공간(Box) 객체.
        """
        return self._observation_space

    @property
    def action_space(self) -> spaces.Tuple:
        """행동(action) 공간 분포를 반환합니다.

        Returns:
            spaces.Tuple: 행동 공간(Tuple) 객체.
        """
        return self._action_space

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """환경을 초기화합니다. (에피소드 시작)

        Args:
            seed (Optional[int], optional): 랜덤 시드.
            options (Optional[Dict[str, Any]], optional): 추가 옵션 (예: 쿼리, user_id 등).

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 초기 상태 벡터와 추가 정보 딕셔너리.
        """
        opts = options or {}
        if "query" in opts:
            self.current_query = opts["query"]
        else:
            self.current_query = None
        self.step_count = 0
        self.current_session_simulated_logs.clear()
        self._set_current_user_info(opts.get("user_id"))
        self.current_user_original_logs_df = self.all_user_logs_df[
            self.all_user_logs_df["user_id"] == self.current_user_id
        ].copy()
        user_initial = self._get_user_data_for_embedding(
            self.current_user_original_logs_df, []
        )
        state = self.embedder.embed_user(user_initial)
        return state, {}

    def step(
        self,
        action_list: List[Tuple[str, int]],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """한 스텝 진행: 추천 → 시뮬레이션 → 보상 → 다음 상태 반환.

        Args:
            action_list (List[Tuple[str, int]]): (콘텐츠 타입, 인덱스) 쌍 리스트.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: (다음 상태, 보상, 종료 여부, 트렁케이트 여부, 추가 정보)
        """
        self.step_count += 1
        user_data = self._get_user_data_for_embedding(
            self.current_user_original_logs_df,
            self.current_session_simulated_logs,
        )
        user_state = self.embedder.embed_user(user_data)
        cand_dict = self.candidate_generator.get_candidates(self.current_query)
        selected_contents = self._select_contents_from_action(cand_dict, action_list)
        if not selected_contents:
            logging.warning("No valid contents selected from actions %s", action_list)
            return user_state, 0.0, True, False, {}
        sim_context = {
            "step_count": self.step_count,
            "session_logs": self.current_session_simulated_logs,
        }
        all_responses = self.response_simulator.simulate_responses(
            selected_contents, sim_context
        )
        total_reward, individual_reward_dict = (
            self.reward_fn.calculate_from_topk_responses(all_responses=all_responses)
        )
        for resp in all_responses:
            if resp.get("clicked"):
                clicked = next(
                    (
                        c
                        for c in selected_contents
                        if c.get("id") == int(resp["content_id"])
                    ),
                    None,
                )
                if clicked:
                    log_entry = self._create_simulated_log_entry(
                        clicked, event_type="CLICK", dwell_time=resp.get("dwell_time")
                    )
                    self.current_session_simulated_logs.append(log_entry)
        next_user_data = self._get_user_data_for_embedding(
            self.current_user_original_logs_df, self.current_session_simulated_logs
        )
        next_state = self.embedder.embed_user(next_user_data)
        done = self.step_count >= self.max_steps
        info: Dict[str, Any] = {
            "all_responses": all_responses,
            "selected_contents": selected_contents,
            "total_clicks": sum(1 for r in all_responses if r.get("clicked")),
            "individual_rewards": individual_reward_dict,
        }
        return next_state, total_reward, done, False, info

    def get_candidates(self) -> Dict[str, List[Any]]:
        """현재 상태(쿼리)에 기반한 추천 후보군을 반환합니다.

        Returns:
            Dict[str, List[Any]]: 타입별 추천 후보군 딕셔너리.
        """
        return self.candidate_generator.get_candidates(self.current_query)
