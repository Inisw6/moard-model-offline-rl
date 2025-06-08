import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random
import logging
from typing import Any, List, Dict

from components.base import BaseEnv
from components.registry import register
from .db_utils import get_users, get_user_logs, get_contents
from .llm_simu import LLMUserSimulator
from .llm_response_handler import LLMResponseHandler
from datetime import datetime, timezone


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
        llm_simulator: LLMUserSimulator,  # 필수 인자
        user_id: int | None = None,
        persona_id: int | None = None,  # 시뮬레이션용 페르소나 ID
        debug: bool = False,
    ) -> None:
        """
        환경을 초기화합니다.

        Args:
            cold_start (int): 콜드스타트 상태 사용 여부.
            max_steps (int): 에피소드 당 최대 추천 횟수.
            top_k (int): 콘텐츠 추천 수.
            embedder: 사용자/콘텐츠 임베딩 객체.
            candidate_generator: 추천 후보군 생성 객체.
            reward_fn: 보상 함수 객체.
            context: 추천 컨텍스트 관리자.
            llm_simulator (LLMUserSimulator): LLM 기반 사용자 시뮬레이터 (필수).
            user_id (int | None): 환경에 할당할 사용자 ID. None이면 임의 선택.
            persona_id (int | None): 시뮬레이션용 페르소나 ID. None이면 기본값 사용.
            debug (bool): 디버깅 모드 활성화 여부.
        """

        super().__init__()
        self.context = context
        self.max_steps = max_steps
        self.top_k = top_k
        self.embedder = embedder
        self.candidate_generator = candidate_generator
        self.reward_fn = reward_fn
        
        # LLM 시뮬레이터는 필수로 제공되어야 함
        if llm_simulator is None:
            raise ValueError("LLM simulator must be provided")
        
        self.llm_simulator = llm_simulator
        self.response_handler = LLMResponseHandler(debug=debug)
        self.current_query = None
        
        # 페르소나 정보 설정
        from .persona_db import get_persona_db
        persona_db = get_persona_db()
        
        if persona_id is None:
            # 랜덤 페르소나 선택
            persona = persona_db.get_random_persona()
            if debug:
                print(f"🎲 랜덤 페르소나 선택: ID{persona.persona_id} ({persona.mbti}, 레벨{persona.investment_level})")
        else:
            # 지정된 페르소나 사용
            persona = persona_db.get_persona_by_id(persona_id)
            if not persona:
                raise ValueError(f"Persona {persona_id} not found in database")
            if debug:
                print(f"🎭 지정 페르소나: ID{persona.persona_id} ({persona.mbti}, 레벨{persona.investment_level})")
        
        # 페르소나 속성 저장
        self.persona_id = persona.persona_id
        self.persona_mbti = persona.mbti
        self.persona_investment_level = persona.investment_level

        self.all_users_df = get_users()
        self.all_user_logs_df = get_user_logs()
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
        
        # action_space를 top-k 전체 선택으로 변경
        # action = [(content_type, index), (content_type, index), ...]
        self._action_space = spaces.Tuple([
            spaces.Tuple((
                spaces.Discrete(len(self.embedder.content_types)),
                spaces.Discrete(100)  # 충분히 큰 후보 인덱스 범위
            )) for _ in range(top_k)
        ])
        
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

    def _select_contents_from_action(self, cand_dict: dict, action_list: List[tuple]) -> List[dict]:
        """
        액션 리스트에서 실제 추천할 콘텐츠들을 추출합니다.

        Args:
            cand_dict (dict): 추천 후보군 {타입: [콘텐츠, ...]}.
            action_list (List[tuple]): [(콘텐츠 타입, 후보 인덱스), ...] 리스트.

        Returns:
            List[dict]: 선택된 콘텐츠들, 유효하지 않으면 빈 리스트.
        """
        selected_contents = []
        for ctype, cand_idx in action_list:
            if ctype in cand_dict and len(cand_dict[ctype]) > cand_idx:
                selected_contents.append(cand_dict[ctype][cand_idx])
            else:
                logging.warning(f"Invalid action ({ctype}, {cand_idx}). Candidate not found.")
        return selected_contents

    def _simulate_user_response(self, all_candidates: dict) -> List[Dict]:
        """
        LLM 기반으로 사용자 반응을 시뮬레이션합니다.
        
        Args:
            all_candidates (dict): 전체 후보군 {타입: [콘텐츠, ...]}.

        Returns:
            List[Dict]: 각 후보별 반응 정보 리스트
                       [{"content_id": str, "clicked": bool, "dwell_time": int}, ...]
        """
        if self.llm_simulator is None:
            # LLM 시뮬레이터가 없으면 기본 확률 기반으로 폴백
            logging.warning("LLM simulator not available. Falling back to random simulation.")
            return self._create_fallback_responses(all_candidates)
        
        try:
            # 전체 후보군을 flat list로 변환
            all_contents = []
            for content_type, contents in all_candidates.items():
                all_contents.extend(contents)
            
            logging.debug(f"Sending {len(all_contents)} contents to LLM simulator")
            
            # 페르소나 정보 사용
            persona_id = self.persona_id
            mbti = self.persona_mbti
            investment_level = self.persona_investment_level
            
            # LLM 시뮬레이터 호출 - 원본 텍스트 반환
            raw_response = self.llm_simulator.simulate_user_response(
                persona_id=persona_id,
                mbti=mbti,
                investment_level=investment_level,
                recommended_contents=all_contents,
                current_context={
                    "step_count": self.step_count,
                    "session_logs": self.current_session_simulated_logs,
                    "all_candidate_types": list(all_candidates.keys())
                }
            )
            
            # LLMResponseHandler를 사용하여 응답 처리 - 모든 후보에 대한 반응 추출
            return self.response_handler.extract_all_responses(
                llm_raw_text=raw_response,
                all_contents=all_contents
            )
                
        except Exception as e:
            logging.error(f"LLM simulation error: {e}. Falling back to random simulation.")
            return self._create_fallback_responses(all_candidates)
    
    def _create_fallback_responses(self, all_candidates: dict) -> List[Dict]:
        """
        LLM 시뮬레이터가 실패했을 때 사용할 폴백 응답 생성
        """
        all_contents = []
        for content_type, contents in all_candidates.items():
            all_contents.extend(contents)
        
        responses = []
        for content in all_contents:
            # 랜덤하게 일부만 클릭
            clicked = random.random() < 0.3  # 30% 확률로 클릭
            dwell_time = random.randint(60, 300) if clicked else 0
            
            responses.append({
                "content_id": content.get("id"),
                "clicked": clicked,
                "dwell_time": dwell_time
            })
        
        return responses

    def _create_simulated_log_entry(self, content: dict, event_type: str, dwell_time: int = None) -> dict:
        """
        시뮬레이션용 로그 엔트리를 생성합니다.

        Args:
            content (dict): 추천된 콘텐츠 정보.
            event_type (str): 이벤트 타입 ("VIEW" 또는 "CLICK").
            dwell_time (int, optional): LLM에서 계산된 체류시간(초). None이면 VIEW는 0, CLICK은 기본값.

        Returns:
            dict: user_logs 포맷의 단일 로그 엔트리.
        """
        # LLM에서 체류시간을 받았으면 사용, 아니면 이벤트 타입에 따라 처리
        if dwell_time is None:
            if event_type == "VIEW":
                time_seconds = 0  # VIEW면 체류시간 0
            else:  # CLICK
                time_seconds = random.randint(60, 600)  # CLICK인데 체류시간 없으면 기본값
        else:
            time_seconds = dwell_time

        return {
            "user_id": self.current_user_id,
            "content_id": content.get("id"),
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content_actual_type": content.get("type"),
            "ratio": 1.0 if event_type == "CLICK" else random.uniform(0.1, 0.9),
            "time": time_seconds,
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
        if options and "query" in options:
            self.current_query = options["query"]
        else:
            self.current_query = None

        user_initial_data = self._get_user_data_for_embedding(
            self.current_user_original_logs_df, []
        )
        state = self.embedder.embed_user(user_initial_data)
        return state, {}

    def step(
        self, action_list: List[tuple]
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        환경에 액션 리스트(top-k 추천)을 적용하고, 다음 상태 및 보상 등을 반환합니다.

        Args:
            action_list (List[tuple]): [(콘텐츠 타입, 후보 인덱스), ...] top-k 개의 액션

        Returns:
            tuple:
                - 다음 상태 (np.ndarray)
                - 보상 (float)
                - done (bool): 에피소드 종료 여부
                - truncated (bool): 트렁케이트 여부(사용 안함)
                - info (dict): 기타 정보
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
            logging.warning(f"No valid contents selected from actions {action_list}")
            return user_state, 0.0, True, False, {}

        # 4) LLM 시뮬레이션: 선택된 top-k 콘텐츠에 대해
        all_responses = self._simulate_user_response_for_topk(selected_contents)

        # 5) 보상 계산: 모든 응답을 사용하여 보상 계산
        total_reward = self.reward_fn.calculate_from_topk_responses(
            all_responses=all_responses,
            selected_contents=selected_contents
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
            "total_clicks": sum(1 for r in all_responses if r["clicked"])
        }
        return next_state, total_reward, done, False, info

    def _simulate_user_response_for_topk(self, selected_contents: List[dict]) -> List[Dict]:
        """
        선택된 top-k 콘텐츠에 대해 LLM 기반 사용자 반응 시뮬레이션
        
        Args:
            selected_contents (List[dict]): 선택된 top-k 콘텐츠 리스트
        
        Returns:
            List[Dict]: 각 콘텐츠별 반응 정보 리스트
        """
        if self.llm_simulator is None:
            logging.warning("LLM simulator not available. Falling back to random simulation.")
            return self._create_fallback_responses_for_list(selected_contents)
        
        try:
            logging.debug(f"Sending {len(selected_contents)} selected contents to LLM simulator")
            
            # 페르소나 정보 사용
            raw_response = self.llm_simulator.simulate_user_response(
                persona_id=self.persona_id,
                mbti=self.persona_mbti,
                investment_level=self.persona_investment_level,
                recommended_contents=selected_contents,
                current_context={
                    "step_count": self.step_count,
                    "session_logs": self.current_session_simulated_logs
                }
            )
            
            # LLMResponseHandler를 사용하여 응답 처리
            return self.response_handler.extract_all_responses(
                llm_raw_text=raw_response,
                all_contents=selected_contents
            )
                
        except Exception as e:
            logging.error(f"LLM simulation error: {e}. Falling back to random simulation.")
            return self._create_fallback_responses_for_list(selected_contents)

    def _create_fallback_responses_for_list(self, contents: List[dict]) -> List[Dict]:
        """
        콘텐츠 리스트에 대한 폴백 응답 생성
        """
        responses = []
        for content in contents:
            clicked = random.random() < 0.3  # 30% 확률로 클릭
            dwell_time = random.randint(60, 300) if clicked else 0
            
            responses.append({
                "content_id": content.get("id"),
                "clicked": clicked,
                "dwell_time": dwell_time
            })
        
        return responses

    def get_candidates(self) -> dict[str, list[Any]]:
        """
        현 상태에서 추천 후보군을 반환합니다.get_candidatesget_candidates

        Returns:
            dict[str, list[Any]]: {타입: 후보 콘텐츠 리스트}
        """
        return self.candidate_generator.get_candidates(self.current_query)
