import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random # 랜덤 이벤트 시뮬레이션용
from components.base import BaseEnv
from components.registry import register
from .db_utils import get_users, get_user_logs, get_contents
from datetime import datetime, timezone # timezone 추가

@register("rec_env")
class RecEnv(gym.Env, BaseEnv):
    def __init__(self, cold_start, max_steps, top_k,
                 embedder, candidate_generator, reward_fn, context, user_id=None, click_prob=0.2):
        super().__init__()
        self.context = context
        self.max_steps = max_steps
        self.top_k = top_k
        self.embedder = embedder
        self.candidate_generator = candidate_generator
        self.reward_fn = reward_fn
        self.click_prob = click_prob # 클릭 확률 파라미터 추가

        self.all_users_df = get_users()
        self.all_user_logs_df = get_user_logs() # 모든 사용자의 모든 로그 (초기 로드용)
        self.all_contents_df = get_contents()

        self.current_user_id = None
        self.current_user_info = None
        self.current_user_original_logs_df = pd.DataFrame() # 현재 사용자의 DB 로그 (리셋 시 설정)
        self.current_session_simulated_logs = [] # 현재 에피소드에서 시뮬레이션된 로그 [{dict}, ...]

        if user_id is None:
            if not self.all_users_df.empty:
                self.current_user_id = self.all_users_df.iloc[0]['id']
            else:
                self.current_user_id = -1 # 더미 ID
                print("Warning: No users found in DB. Using dummy user_id = -1.")
        else:
            self.current_user_id = user_id
        
        if self.current_user_id != -1 and not self.all_users_df.empty:
            user_info_series = self.all_users_df[self.all_users_df['id'] == self.current_user_id]
            if not user_info_series.empty:
                self.current_user_info = user_info_series.iloc[0].to_dict()
            else:
                print(f"Warning: User ID {self.current_user_id} not found. Using dummy user_info.")
                self.current_user_info = {"id": self.current_user_id, "uuid": "dummy_user_not_found"}
        elif self.current_user_id == -1:
             self.current_user_info = {"id": -1, "uuid": "dummy_user"}

        state_dim = embedder.output_dim()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.step_count = 0

    def _get_user_data_for_embedding(self, base_logs_df, simulated_logs_list):
        """주어진 로그들을 합쳐 embed_user가 기대하는 형태로 만듭니다."""
        combined_logs_df = base_logs_df
        if simulated_logs_list: # 시뮬레이션된 로그가 있다면 DataFrame으로 변환 후 합치기
            sim_logs_df = pd.DataFrame(simulated_logs_list)
            # base_logs_df와 sim_logs_df의 컬럼이 다를 수 있으므로, 필요한 컬럼만 선택하거나 맞춰줘야 함.
            # 여기서는 두 DataFrame이 동일한 필수 컬럼(예: content_id, event_type 등)을 가지고 있다고 가정.
            # content_actual_type은 base_logs_df에만 있을 수 있음 (DB 조인 결과)
            # sim_logs_df에는 직접 content_actual_type을 넣어줬으므로 괜찮음.
            combined_logs_df = pd.concat([base_logs_df, sim_logs_df], ignore_index=True)
        
        # 로그와 콘텐츠 정보 조인 (content_actual_type 추가)
        # 이 작업은 combined_logs_df에 대해 수행되어야 함.
        if not combined_logs_df.empty and not self.all_contents_df.empty:
            # content_actual_type이 이미 있는 로그(sim_logs_df에서 온)와 없는 로그(base_logs_df에서 온)를 구분할 필요는 없음.
            # 조인 후 content_actual_type_x, content_actual_type_y 같은 문제가 생길 수 있으므로, 필요한 컬럼만 선택.
            # 또는, base_logs_df에 content_actual_type이 없다면, 이 단계에서 채워줌.
            # 가장 간단한 방법은 combined_logs_df에서 content_actual_type이 없는 로그에 대해 채우는 것.
            if 'content_actual_type' not in combined_logs_df.columns:
                 combined_logs_df['content_actual_type'] = None # 임시 컬럼
            
            merged_df = pd.merge(
                combined_logs_df,
                self.all_contents_df[['id', 'type']].rename(columns={'id': 'content_id', 'type': 'content_db_type'}),
                on='content_id',
                how='left'
            )
            # 시뮬레이션 로그에서 content_actual_type을 이미 설정했으므로, DB에서 가져온 타입은 content_db_type으로 받음
            # content_actual_type이 없는 경우(초기 DB로그) content_db_type 사용
            merged_df['content_actual_type'] = merged_df['content_actual_type'].fillna(merged_df['content_db_type'])
            merged_df.drop(columns=['content_db_type'], inplace=True) # 임시 컬럼 삭제
            final_logs_df = merged_df
        else:
            final_logs_df = combined_logs_df # 조인할 콘텐츠 정보가 없거나 로그가 비면 그대로 사용

        processed_logs = []
        if not final_logs_df.empty:
            processed_logs = final_logs_df.to_dict('records')

        return {
            "user_info": self.current_user_info,
            "recent_logs": processed_logs,
            "current_time": datetime.now(timezone.utc) # UTC 시간 사용 권장
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.context.reset()
        self.current_session_simulated_logs = [] # 시뮬레이션 로그 초기화

        if self.current_user_id != -1:
            self.current_user_original_logs_df = self.all_user_logs_df[
                self.all_user_logs_df['user_id'] == self.current_user_id
            ].copy() # 현재 사용자의 원본 로그 (복사해서 사용)
        else:
            self.current_user_original_logs_df = pd.DataFrame() # 더미 사용자는 빈 로그
        
        user_initial_data = self._get_user_data_for_embedding(self.current_user_original_logs_df, [])
        state = self.embedder.embed_user(user_initial_data)
        return state, {}

    def step(self, action):
        ctype, cand_idx = action
        self.step_count += 1

        # 현재 상태 기반으로 후보 가져오기 (이전처럼 state를 사용하지 않는다고 가정)
        # cand_user_data = self._get_user_data_for_embedding(self.current_user_original_logs_df, self.current_session_simulated_logs)
        # cand_dict = self.candidate_generator.get_candidates(cand_user_data)
        # 위는 만약 후보 생성이 현재까지의 모든 로그에 의존할 경우. 지금은 state와 무관하므로 이전처럼.
        cand_dict = self.candidate_generator.get_candidates(None) 

        selected_content = None
        if ctype in cand_dict and len(cand_dict[ctype]) > cand_idx:
            selected_content = cand_dict[ctype][cand_idx]
        
        if selected_content is None:
            print(f"Warning: Invalid action {action}. Candidate not found.")
            # 현재 상태를 그대로 반환 (변화 없음)
            current_data = self._get_user_data_for_embedding(self.current_user_original_logs_df, self.current_session_simulated_logs)
            state = self.embedder.embed_user(current_data)
            return state, 0.0, True, False, {} # 에피소드 종료 (오류 상황)

        # 이벤트 시뮬레이션 (VIEW는 기본, CLICK은 확률적)
        simulated_event_type = "VIEW"
        if random.random() < self.click_prob:
            simulated_event_type = "CLICK"

        # 보상 계산 (수정된 reward_fn 호출)
        reward = self.reward_fn.calculate(selected_content, event_type=simulated_event_type)

        # 시뮬레이션된 로그 생성 및 추가
        content_id = selected_content.get('id')
        content_actual_type = selected_content.get('type') # 후보 생성시 content의 type을 사용
        
        new_log_entry = {
            'user_id': self.current_user_id,
            'content_id': content_id,
            'event_type': simulated_event_type,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'content_actual_type': content_actual_type, # embed_user가 사용할 콘텐츠 타입
            # user_logs 스키마에 있는 다른 필드들 (ratio, time 등)도 필요시 추가 가능 (예: 기본값)
            'ratio': 1.0 if simulated_event_type == "CLICK" else random.uniform(0.1, 0.9), # 클릭하면 다 봤다고 가정, view는 랜덤
            'time': random.randint(5, 300) if simulated_event_type == "VIEW" else random.randint(60, 600) # 체류시간 랜덤 설정
        }
        self.current_session_simulated_logs.append(new_log_entry)

        # 다음 상태 생성 (원본 로그 + 현재 세션의 시뮬레이션 로그 사용)
        user_next_data = self._get_user_data_for_embedding(self.current_user_original_logs_df, self.current_session_simulated_logs)
        next_state = self.embedder.embed_user(user_next_data)

        done = self.step_count >= self.max_steps
        self.context.step()
        return next_state, reward, done, False, {}

    def get_candidates(self, state):
        # 현재 candidate_generator는 state를 사용하지 않으므로 그대로 전달
        return self.candidate_generator.get_candidates(state)