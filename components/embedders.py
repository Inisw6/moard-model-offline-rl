import numpy as np
import json
import logging

from components.base import BaseUserEmbedder, BaseContentEmbedder, BaseEmbedder
from components.registry import register, make
from .db_utils import get_contents

# SBert 
from sentence_transformers import SentenceTransformer
import re


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
    SBERT 기반으로, 오직 콘텐츠(text)만 임베딩하는 예시.
    - SBERT 임베딩: 768차원 (모델에 따라 다름)
    - 최종 content_dim 차원으로 패딩/자르기
    """

    def __init__(
        self,
        content_dim: int = 768,  # SBERT 출력 차원에 맞춰 기본값을 768으로 설정
        model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    ):
        """
        Args:
            content_dim (int): 최종 반환할 벡터 차원.
                               (SBERT 차원과 동일하거나 더 크게 설정할 수 있음)
            model_name (str): SBERT 모델 이름
        """
        
        # 1) SBERT 모델 로드 (한 번만)
        print(f"Loading SBERT model '{model_name}' ...")
        self.sbert_model = SentenceTransformer(model_name)
        ## content_dim을 잘못입력했을떄를 대비해서
        self.pretrained_dim = self.sbert_model.get_sentence_embedding_dimension() 

        # 2) 최종 차원 설정
        if content_dim == self.pretrained_dim:
            self.content_dim = content_dim
        else: 
            print("[Warning]:yalm에서 content_dim을 잘못입력!")

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
                - "text": SBERT로 임베딩할 문자열 (title + description 행)

        Returns:
            np.ndarray: (content_dim,) 크기의 float32 벡터
                [0:pretrained_dim] = SBERT 임베딩 (예: 768)
                나머지는 0으로 패딩하거나, 자릅니다.
    """
    
        # 1) [전처리단계] SBERT 입력 문자열 가져오기 (title + description) 
        raw_text = content.get('title', '') + content.get('description', '')
        raw_text = re.sub(r"<.*?>", "", raw_text)

        # 2) [임베딩단계] SBERT 임베딩 ** 임베딩단계에서는 pretrained_dim으로]
        if raw_text is None : ## raw_text가 공백일때 -> 0 벡터로 처리
            sbert_emb = np.zeros(self.pretrained_dim, dtype=np.float32)

        ## [??생각해야할 부분] : title, description이 하나라도 공백일때 ###
        # elif raw_text.strip() == "":                               #
        #     sbert_emb =                                            #
        ## 현재는 그냥 없으면 없는대로, 임베딩 진행. ######################

        else:
            # SBERT encode (리스트로 넘기고 [0] 으로 꺼낸다)
            try:
                sbert_emb = self.sbert_model.encode(
                    [raw_text],
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False
                )[0]  ## sbert_emb = array(768차원벡터, dtype=np.float64)
                sbert_emb = sbert_emb.astype(np.float32)
                print(sbert_emb)
            except Exception as e:
                ## embeding실패할 경우 -> 0벡터
                print(f"[Warning] SBERT inference failed: {e}")
                sbert_emb = np.zeros(self.pretrained_dim, dtype=np.float32)

        # 2) SBERT 임베딩만 사용 
        vec = sbert_emb  # vec결과는 무조건 pretrained_dim으로

        # content_dim오류처리할거면, 여기서 !!
        # 3) [**여기는 pretrained_dim이아니라, content_dim으로!! 최종 content_dim에 맞추어 패딩 또는 자르기]
        if len(vec) < self.content_dim:
            pad_len = self.content_dim - len(vec)
            vec = np.pad(vec, (0, pad_len), mode="constant")
        else:
            vec = vec[: self.content_dim]

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
