import numpy as np
import json
from components.base import BaseEmbedder
from components.registry import register
from .db_utils import get_contents

@register("simple_concat")
class SimpleConcatBuilder(BaseEmbedder):
    def __init__(self, user_dim: int = 30, content_dim: int = 5):
        self.all_contents_df = get_contents()
        if not self.all_contents_df.empty:
            self.content_types = self.all_contents_df['type'].unique().tolist()
        else:
            self.content_types = ["youtube", "blog", "news"]
        self.num_content_types = len(self.content_types)
        self.type_to_idx_map = {t: i for i, t in enumerate(self.content_types)}
        
        # 콘텐츠 임베딩 차원 설정
        # pretrained_content_embedding_dim 은 content_dim 에서 타입 원핫 인코딩 차원을 제외한 것
        self.pretrained_content_embedding_dim = content_dim - self.num_content_types
        if self.pretrained_content_embedding_dim < 0:
            print(f"Warning: content_dim ({content_dim}) is too small for {self.num_content_types} content types. "
                  f"Setting pretrained_content_embedding_dim to 0. content_dim will be {self.num_content_types}.")
            self.pretrained_content_embedding_dim = 0
            self.content_dim = self.num_content_types # content_dim을 타입 원핫 인코딩 차원 합으로 조정
        else:
            self.content_dim = content_dim # 사용자가 지정한 content_dim 사용
        
        min_user_dim = 2 + self.num_content_types
        if user_dim < min_user_dim:
            print(f"Warning: user_dim ({user_dim}) is too small. Adjusting to {min_user_dim}...")
            self.user_dim = min_user_dim
        else:
            self.user_dim = user_dim

    def output_dim(self) -> int:
        return self.user_dim

    def embed_user(self, user: dict) -> np.ndarray:
        logs = user.get("recent_logs", [])
        if not logs:
            return np.zeros(self.user_dim, dtype=np.float32)
        
        ratio_avg = np.mean([l.get("ratio", 0.0) for l in logs])
        time_avg = np.mean([l.get("time", 0.0) for l in logs])
        
        type_counts = {t: 0 for t in self.content_types}
        for log in logs:
            # 'content_actual_type' 필드를 사용하고, 없을 경우 빈 문자열로 처리
            actual_content_type = str(log.get("content_actual_type", "")).lower()
            if actual_content_type in self.type_to_idx_map:
                type_counts[actual_content_type] += 1
            # event_type이 VIEW나 CLICK인 것을 활용하여 다른 피처를 만들 수도 있음 (예: 클릭률 등)
            # event_type = log.get("event_type", "").lower()
        
        total_logs_for_known_types = sum(type_counts.values())
        type_vec = np.array([
            type_counts[t]/total_logs_for_known_types if total_logs_for_known_types > 0 else 0 
            for t in self.content_types
        ])
        
        vec = np.concatenate([[ratio_avg, time_avg], type_vec])
        if len(vec) < self.user_dim:
            vec = np.pad(vec, (0, self.user_dim - len(vec)), 'constant')
        elif len(vec) > self.user_dim: # 필요한 경우 자르기 (user_dim이 min_user_dim 보다 작게 조정된 경우)
            vec = vec[:self.user_dim]
        return vec.astype(np.float32)

    def embed_content(self, content: dict) -> np.ndarray:
        pretrained_emb = np.array([], dtype=np.float32)
        if self.pretrained_content_embedding_dim > 0:
            try:
                embedding_str = content.get("embedding") # 기본값 "[]" 제거, None일 수 있음
                if embedding_str is None or embedding_str == "": # 비어있거나 None이면 빈 임베딩으로 간주
                    # print(f"Warning: Content ID {content.get('id')} has empty or missing embedding string. Using zero vector for pretrained part.")
                    pretrained_emb = np.zeros(self.pretrained_content_embedding_dim, dtype=np.float32)
                else:
                    parsed_list = json.loads(embedding_str)
                    if not isinstance(parsed_list, list) or not all(isinstance(x, (int, float)) for x in parsed_list):
                        # print(f"Warning: Content ID {content.get('id')} embedding is not a list of numbers: '{embedding_str}'. Using zero vector for pretrained part.")
                        pretrained_emb = np.zeros(self.pretrained_content_embedding_dim, dtype=np.float32)
                    else:
                        pretrained_emb_list = np.array(parsed_list, dtype=np.float32)
                        if len(pretrained_emb_list) != self.pretrained_content_embedding_dim:
                            # print(f"Warning: Content ID {content.get('id')} pretrained embedding dimension mismatch. Expected {self.pretrained_content_embedding_dim}, got {len(pretrained_emb_list)}. Using zero vector for pretrained part.")
                            pretrained_emb = np.zeros(self.pretrained_content_embedding_dim, dtype=np.float32)
                        else:
                            pretrained_emb = pretrained_emb_list
            except json.JSONDecodeError:
                # print(f"Warning: Content ID {content.get('id')} failed to parse JSON embedding: '{embedding_str}'. Using zero vector for pretrained part.")
                pretrained_emb = np.zeros(self.pretrained_content_embedding_dim, dtype=np.float32)
            except ValueError as e:
                # print(f"Warning: Content ID {content.get('id')} value error during embedding processing: {e}. Using zero vector for pretrained part.")
                pretrained_emb = np.zeros(self.pretrained_content_embedding_dim, dtype=np.float32)
        else: # pretrained_content_embedding_dim is 0, so no pretrained embedding part
            pretrained_emb = np.array([], dtype=np.float32) # 빈 배열 명시
        
        content_type_str = content.get("type", "").lower()
        type_idx = self.type_to_idx_map.get(content_type_str, -1) # 없는 타입이면 -1
        type_onehot = np.zeros(self.num_content_types, dtype=np.float32)
        if type_idx != -1:
            type_onehot[type_idx] = 1.0
        else:
            # print(f"Warning: Content ID {content.get('id')} has unknown type '{content_type_str}'. Using zero vector for type onehot.")
            pass # 이미 0벡터임
            
        final_vec = np.concatenate([pretrained_emb, type_onehot])
        # 최종 벡터 길이 확인 및 조정 (주로 self.content_dim이 조정된 경우)
        if len(final_vec) != self.content_dim:
             # print(f"Warning: Content ID {content.get('id')} final embedding dimension mismatch. Expected {self.content_dim}, got {len(final_vec)}. Adjusting...")
             # 이 경우는 self.pretrained_content_embedding_dim + self.num_content_types 와 self.content_dim이 다른 경우.
             # __init__에서 self.content_dim을 조정했으므로, 이론적으로는 일치해야 함.
             # 만약 다르다면, 0으로 패딩하거나 잘라내기.
            if len(final_vec) < self.content_dim:
                final_vec = np.pad(final_vec, (0, self.content_dim - len(final_vec)), 'constant')
            else:
                final_vec = final_vec[:self.content_dim]

        return final_vec.astype(np.float32)

    def estimate_preference(self, state: np.ndarray) -> dict:
        if len(state) < 2 + self.num_content_types:
             # print(f"Warning: State vector too short for preference estimation. State len: {len(state)}, expected at least {2 + self.num_content_types}")
             return {t: 0.0 for t in self.content_types} # 기본값 반환
        type_prefs = state[2 : 2 + self.num_content_types]
        return {self.content_types[i]: float(type_prefs[i]) for i in range(len(type_prefs))}