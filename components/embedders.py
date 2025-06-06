import numpy as np
import json
import logging

from components.base import BaseUserEmbedder, BaseContentEmbedder, BaseEmbedder
from components.registry import register, make
from .db_utils import get_contents
import logging

# SBert 
from sentence_transformers import SentenceTransformer
import re

# Doc2vec
from gensim.models.doc2vec import Doc2Vec


@register("simple_user")
class SimpleUserEmbedder(BaseUserEmbedder):
    """
    사용자의 최근 로그/콘텐츠 속성을 단순 연결하여 벡터로 변환하는 유저 임베더.

    - user 벡터: [평균 ratio, 평균 time, 각 콘텐츠 타입별 비율, (0 padding)] (크기: user_dim)
    """

    def __init__(self, user_dim: int = 30):
        """
        Args:
            user_dim (int): 반환할 user 임베딩 벡터 차원
        """
        self.all_contents_df = get_contents()
        if not self.all_contents_df.empty:
            self.content_types = self.all_contents_df["type"].unique().tolist()
        else:
            self.content_types = ["youtube", "blog", "news"]
        self.num_content_types = len(self.content_types)
        self.type_to_idx_map = {t: i for i, t in enumerate(self.content_types)}

        min_user_dim = 2 + self.num_content_types
        if user_dim < min_user_dim:
            logging.warning(
                "user_dim (%d) is too small. Adjusting to %d...", user_dim, min_user_dim
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

@register("sbert_content")
class SbertContentEmbedder(BaseContentEmbedder):
    """
    SBERT 기반으로, 콘텐츠(text)만 임베딩.
    - SBERT 임베딩: 768차원
    - 최종 content_dim 차원으로 패딩/자르기
    """

    def __init__(
        self,
        content_dim: int = 768,  # SBERT 출력 차원에 맞춰 기본값을 768으로 설정
        model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    ):
        """
        Args:
            model_name (str): SBERT 모델 이름
            content_dim (int): 반환할 벡터 차원.
        """
        
        # 1) SBERT 모델 로드 
        logging.info("Loading SBERT model '%s' ...", model_name)
        self.sbert_model = SentenceTransformer(model_name)
        self.pretrained_dim = self.sbert_model.get_sentence_embedding_dimension() 

        # 2) content_dim 설정
        if content_dim == self.pretrained_dim:
            self.content_dim = content_dim
        else:
            ## 차원맞추기
            logging.warning(
                "설정된 content_dim (%d)이 SBERT pretrained_dim (%d)과 다릅니다. "
                "이 경우, vector를 pretrained_dim (%d)으로 맞춥니다.",
                content_dim,
                self.pretrained_dim,
                self.pretrained_dim,
            )
            self.content_dim = self.pretrained_dim

        # 3) type처리 (concat에서 필요한 content_types)
        self.all_contents_df = get_contents()
        if not self.all_contents_df.empty:
            self.content_types = self.all_contents_df["type"].unique().tolist()
        else:
            self.content_types = ["youtube", "blog", "news"]

    
    def output_dim(self) -> int:
        """임베딩되는 content 벡터의 차원 반환"""
        return self.content_dim

    def embed_content(self, content: dict) -> np.ndarray:
        """
        Args:
            content (dict):
                - "text": 임베딩할 문자열 (title + description 행)

        Returns:
            np.ndarray: (content_dim,) 크기의 float32 벡터
                [0:pretrained_dim] = SBERT 임베딩 768
                나머지는 0으로 패딩하거나, 자릅니다.
    """
    
        # 1) [전처리단계] 입력 문자열 가져오기 (title + description) 
        raw_text = content.get('title', '') + content.get('description', '')
        raw_text = re.sub(r"<.*?>", "", raw_text)

        # 2) [임베딩단계] SBERT 임베딩
        if raw_text == "" : 
            sbert_emb = np.zeros(self.pretrained_dim, dtype=np.float32)
        else:
            try:
                sbert_emb = self.sbert_model.encode(
                    [raw_text],
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False
                )[0]  ## sbert_emb = array(768차원벡터, dtype=np.float64)
                sbert_emb = sbert_emb.astype(np.float32)
            except Exception as e:
                logging.warning("SBERT inference failed: %s", e)
                sbert_emb = np.zeros(self.pretrained_dim, dtype=np.float32)
        vec = sbert_emb  

        return vec
        

@register("doc2vec_content")
class Doc2VecContentEmbedder(BaseContentEmbedder):
    """
    Doc2Vec 기반으로, 콘텐츠(text)만 임베딩.
    - 사전 학습된 Doc2Vec 모델을 불러와서 임베딩 : 300차원
    - 최종 content_dim 차원으로 패딩/자르기
    """

    def __init__(
        self,
        model_path: str = "models/doc2vec.model",  # train_doc2vec.py에서 저장한 파일 경로
        content_dim: int = 300,                   # Doc2Vec 벡터 차원 (train 시에 사용한 vector_size와 동일하게)
    ):
        """
        Args:
            model_path (str): 디스크에 저장된 Doc2Vec 모델 경로
            content_dim (int): 반환할 벡터 차원 (Doc2Vec vector_size와 맞춰야한다!!)
        """
        # 1) Doc2Vec 모델 로드
        logging.info("Loading Doc2Vec model from '%s' ...", model_path)
        self.doc2vec_model = Doc2Vec.load(model_path)
        self.pretrained_dim = self.doc2vec_model.vector_size  # 300

        # 2) content_dim 설정
        if content_dim == self.pretrained_dim:
            self.content_dim = content_dim
        else:
            # 만약 YAMl에서 잘못된 content_dim을 지정했을 경우 경고
            logging.warning(
                "설정된 content_dim (%d)이 Doc2Vec vector_size (%d)와 다릅니다. "
                "이 경우, vector를 pretrained_dim (%d)으로 맞춥니다.",
                content_dim,
                self.pretrained_dim,
                self.pretrained_dim,
            )
            self.content_dim = self.pretrained_dim

        # 3) type처리 (concat에서 필요한 content_types)
        self.all_contents_df = get_contents()
        if not self.all_contents_df.empty:
            self.content_types = self.all_contents_df["type"].unique().tolist()
        else:
            self.content_types = ["youtube", "blog", "news"]

    def output_dim(self) -> int:
        """임베딩되는 content 벡터의 차원 반환"""
        return self.content_dim

    def embed_content(self, content: dict) -> np.ndarray:
        """
        Args:
            content (dict):
                - "text": 임베딩할 문자열 (title + description 행)


        Returns:
            np.ndarray: (content_dim,) 크기의 float32 벡터
            [0:pretrained_dim] = doc2vec 임베딩 300
            나머지는 0으로 패딩하거나, 자릅니다.
        """
        # 1) [전처리단계] 입력 문자열 가져오기 (title + description) 
        raw_text = content.get("title", "") + " " + content.get("description", "") 
        raw_text = re.sub(r"<.*?>", "", raw_text).strip()

        # 2-1) [임베딩준비] 토크나이즈 (Doc2Vec은 word 리스트를 입력으로 사용한다.)
        #   ** 정교한 토크나이징(konlpy, nltk)을 사용해도된다.
        if raw_text == "":
            tokens = []
        else:
            tokens = raw_text.split()

        # 2-2) [임베딩단계] Doc2Vec inference: vector을 추론
        try:
            inferred_vec = self.doc2vec_model.infer_vector(tokens)
            doc2vec_emb = np.array(inferred_vec, dtype=np.float32)  # pretrained_dim크기
        except Exception as e:
            logging.warning("Doc2Vec inference failed: %s", e)
            doc2vec_emb = np.zeros(self.pretrained_dim, dtype=np.float32)

        vec = doc2vec_emb
        return vec



@register("simple_content")
class SimpleContentEmbedder(BaseContentEmbedder):
    """
    콘텐츠 정보를 단순 연결하여 벡터로 변환하는 콘텐츠 임베더.

    - content 벡터: [사전학습 임베딩, 콘텐츠 타입 원핫, (0 padding)] (크기: content_dim)
    """

    def __init__(self, content_dim: int = 5):
        """
        Args:
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

    def output_dim(self) -> int:
        """임베딩되는 content 벡터의 차원 반환"""
        return self.content_dim

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




@register("simple_concat")
class SimpleConcatEmbedder(BaseEmbedder):
    """
    SimpleUserEmbedder와 SimpleContentEmbedder를 조합한 임베더.
    """

    def __init__(self, user_embedder: dict, content_embedder: dict):
        """
        Args:
            user_embedder (dict): 유저 임베더 설정
                - type: 임베더 타입
                - params: 임베더 파라미터
            content_embedder (dict): 콘텐츠 임베더 설정
                - type: 임베더 타입
                - params: 임베더 파라미터
        """
        from components.registry import make

        self.user_embedder = make(user_embedder["type"], **user_embedder["params"])
        self.content_embedder = make(
            content_embedder["type"], **content_embedder["params"]
        )

        # content_types는 content_embedder에서 가져옴
        self.content_types = self.content_embedder.content_types
        self.num_content_types = len(self.content_types)
        self.type_to_idx_map = {t: i for i, t in enumerate(self.content_types)}

        # user_dim은 user_embedder에서 가져옴
        self.user_dim = self.user_embedder.user_dim

        # content_dim은 content_embedder에서 가져옴
        self.content_dim = self.content_embedder.content_dim

        super().__init__(self.user_embedder, self.content_embedder)
