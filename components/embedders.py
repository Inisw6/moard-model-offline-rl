import numpy as np
import json
import logging
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import torch

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
    
    
@register("doc2vec")
class Doc2VecEmbedder(BaseEmbedder):
    """
    Doc2Vec
    - user 벡터: [평균 ratio, 평균 time, 각 콘텐츠 타입별 비율, (0 padding)] (크기: 500개)
    - content 벡터: [Doc2Vec 임베딩(300차원), 콘텐츠 타입 원핫, (0 padding)] (크기: 500개)
    """

    def __init__(self, user_dim: int = 30, content_dim: int = 305, is_one_hot: int = 1):
        """
        Args:
            user_dim (int): 반환할 user 임베딩 벡터 차원
            content_dim (int): 반환할 content 임베딩 벡터 차원 (300 + 타입수 이상 필요)
        """

        ## db에서 contents table -> all_contents_df 불러오기.
        self.all_contents_df = get_contents()
        if not self.all_contents_df.empty:
            self.content_types = self.all_contents_df["type"].unique().tolist()
        else:
            self.content_types = ["youtube", "blog", "news"]
        self.num_content_types = len(self.content_types)
        self.type_to_idx_map = {t: i for i, t in enumerate(self.content_types)}

        # content 차원
        self.content_dim = content_dim - self.num_content_types # 300 = 303 - 3

        ## user 차원 
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

    def create_doc2vec_embedding(self, content: dict) -> str:
        """
        content의 title과 description을 결합하여 Doc2Vec 임베딩을 생성합니다.

        Args:
            content (dict):
                - title: (str) 콘텐츠 제목
                - description: (str) 콘텐츠 설명

        Returns:
            str: JSON 문자열로 변환된 임베딩 벡터
        """
        # title과 description 결합
        text = f"{content.get('title', '')} {content.get('description', '')}"
        
        # 단어 단위로 분리
        words = text.split()
        
        # TaggedDocument 생성
        tagged_doc = TaggedDocument(words=words, tags=['0'])
        
        # Doc2Vec 모델 초기화
        model = Doc2Vec(
            vector_size=self.content_dim,
            window=5,
            min_count=1,
            workers=4,
            epochs=40
        )
        
        # vocabulary 구축
        model.build_vocab([tagged_doc])
        
        # 모델 학습
        model.train([tagged_doc], total_examples=model.corpus_count, epochs=model.epochs)
        
        # 임베딩 벡터 추출
        embedding_vector = model.infer_vector(words)
        
        # JSON 문자열로 변환
        return json.dumps(embedding_vector.tolist())

    def embed_content(self, content: dict) -> np.ndarray:
        """
        콘텐츠 정보를 임베딩 벡터로 변환.

        Args:
            content (dict):
                - title: (str) 콘텐츠 제목
                - description: (str) 콘텐츠 설명
                - type: (str) 콘텐츠 타입

        Returns:

            - if 
            np.ndarray: (content_dim,)
                [0:300] = Doc2Vec 임베딩,
                [300:] = 타입 원핫 인코딩,
                부족시 0 padding
        """
        # Doc2Vec 임베딩 생성
        embedding_str = self.create_doc2vec_embedding(content)
        content['embedding'] = embedding_str

        # Doc2Vec 임베딩 파싱
        try:
            doc2vec_emb = np.array(json.loads(embedding_str), dtype=np.float32) # 300
            
            if len(doc2vec_emb) != (self.content_dim): # 300 != 300
                doc2vec_emb = np.zeros(self.content_dim, dtype=np.float32)

        except (json.JSONDecodeError, ValueError):
            doc2vec_emb = np.zeros(self.content_dim, dtype=np.float32)


        # 타입 원핫 인코딩
        content_type_str = content.get("type", "").lower()
        type_idx = self.type_to_idx_map.get(content_type_str, -1)
        type_onehot = np.zeros(self.num_content_types, dtype=np.float32)
        if type_idx != -1:
            type_onehot[type_idx] = 1.0

        # # [원핫포함] : 최종 벡터 생성 
        final_vec = np.concatenate([doc2vec_emb, type_onehot]) # 303 = 300 + 3
        # if len(final_vec) != self.content_dim+3: # 303
        #     if len(final_vec) < self.content_dim+3:
        #         final_vec = np.pad(
        #             final_vec, (0, (self.content_dim+3) - len(final_vec)), "constant"
        #         )
        #     else:
        #         final_vec = final_vec[: self.content_dim+3]
        
        # [원핫 미포함] 
        final_vec = final_vec[:self.content_dim]

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

@register('bert')
class BERTEmbedder(BaseEmbedder):
    """
    BERT를 사용하여 콘텐츠의 제목과 설명을 임베딩하는 클래스.
    
    - content 벡터: [BERT 임베딩(768차원), 콘텐츠 타입 원핫, (0 padding)] (크기: content_dim)
    """

    def __init__(self, user_dim: int = 30, content_dim: int = 771):
        """
        Args:
            user_dim (int): 반환할 user 임베딩 벡터 차원
            content_dim (int): 반환할 content 임베딩 벡터 차원 (768 + 타입수 이상 필요)
        """
        self.all_contents_df = get_contents()
        if not self.all_contents_df.empty:
            self.content_types = self.all_contents_df["type"].unique().tolist()
        else:
            self.content_types = ["youtube", "blog", "news"]
        self.num_content_types = len(self.content_types)
        self.type_to_idx_map = {t: i for i, t in enumerate(self.content_types)}

        # content 차원 : content_dim[768] = content_dim[771] - 3
        self.content_dim = content_dim - self.num_content_types
        
        # user 차원 :
        min_user_dim = 2 + self.num_content_types
        if user_dim < min_user_dim:
            print(
                f"Warning: user_dim ({user_dim}) is too small. Adjusting to {min_user_dim}..."
            )
            self.user_dim = min_user_dim
        else:
            self.user_dim = user_dim

        # BERT 모델 초기화
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
            self.model = AutoModel.from_pretrained("klue/bert-base")
        except ImportError:
            logging.error("transformers 패키지가 설치되어 있지 않습니다. pip install transformers를 실행해주세요.")
            raise

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

    def create_bert_embedding(self, content: dict) -> str:
        """
        content의 title과 description을 결합하여 BERT 임베딩을 생성합니다.

        Args:
            content (dict):
                - title: (str) 콘텐츠 제목
                - description: (str) 콘텐츠 설명

        Returns:
            str: JSON 문자열로 변환된 임베딩 벡터
        """
        # title과 description 결합
        text = f"{content.get('title', '')} {content.get('description', '')}"
        
        # BERT 토크나이징
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # BERT 임베딩 생성
        with torch.no_grad():
            outputs = self.model(**inputs)
            # [CLS] 토큰의 임베딩을 사용 (문장 전체의 의미를 담고 있음)
            embedding_vector = outputs.last_hidden_state[0, 0, :].numpy()
        
        # JSON 문자열로 변환
        return json.dumps(embedding_vector.tolist())

    def embed_content(self, content: dict) -> np.ndarray:
        """
        콘텐츠 정보를 임베딩 벡터로 변환.

        Args:
            content (dict):
                - title: (str) 콘텐츠 제목
                - description: (str) 콘텐츠 설명
                - type: (str) 콘텐츠 타입

        Returns:
            np.ndarray: (content_dim,)
                [0:768] = BERT 임베딩,
                [768:] = 타입 원핫 인코딩,
                부족시 0 padding
        """
        # BERT 임베딩 생성
        embedding_str = self.create_bert_embedding(content)
        content['embedding'] = embedding_str

        # BERT 임베딩 파싱
        try:
            bert_emb = np.array(json.loads(embedding_str), dtype=np.float32) # 768
            if len(bert_emb) != self.content_dim:
                bert_emb = np.zeros(self.content_dim, dtype=np.float32)
        except (json.JSONDecodeError, ValueError):
            bert_emb = np.zeros(self.content_dim, dtype=np.float32)

        # 타입 원핫 인코딩
        content_type_str = content.get("type", "").lower()
        type_idx = self.type_to_idx_map.get(content_type_str, -1)
        type_onehot = np.zeros(self.num_content_types, dtype=np.float32)
        if type_idx != -1:
            type_onehot[type_idx] = 1.0

        # # [원핫인코딩 포함] 최종 벡터 생성 
        final_vec = np.concatenate([bert_emb, type_onehot]) # 771 = 768 + 3
        # if len(final_vec) != self.content_dim+3: # 768
        #     if len(final_vec) < self.content_dim+3:
        #         final_vec = np.pad(
        #             final_vec, (0, (self.content_dim+3) - len(final_vec)), "constant"
        #         )
        #     else:
        #         final_vec = final_vec[: self.content_dim+3]

        # [원핫인코딩 비포함]
        final_vec = final_vec[:self.content_dim]

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

