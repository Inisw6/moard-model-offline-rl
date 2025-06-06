import numpy as np
import json
import logging
import re

from typing import Dict, Any, List

from components.base import BaseUserEmbedder, BaseContentEmbedder, BaseEmbedder
from components.registry import register, make
from .db_utils import get_contents as _original_get_contents

from sentence_transformers import SentenceTransformer
from gensim.models.doc2vec import Doc2Vec


# Module-level cache for contents DataFrame
_cached_contents_df = None


def get_contents_cached():
    """
    get_contents() 호출 결과를 캐싱하여 여러 번 호출 시 DB I/O를 줄입니다.
    """
    global _cached_contents_df
    if _cached_contents_df is None:
        try:
            _cached_contents_df = _original_get_contents()
        except Exception as e:
            logging.error("get_contents 호출 중 예외 발생: %s", e)
            _cached_contents_df = None
    return _cached_contents_df


@register("simple_user")
class SimpleUserEmbedder(BaseUserEmbedder):
    """
    사용자의 최근 로그/콘텐츠 속성을 단순 연결하여 벡터로 변환하는 유저 임베더.

    User 벡터 구성:
        - [0]: 평균 ratio
        - [1]: 평균 time
        - [2:2+N]: 각 콘텐츠 타입별 비율
        - 이후: 0 padding
    """

    def __init__(self, user_dim: int = 30):
        """
        Args:
            user_dim (int): 반환할 user 임베딩 벡터 차원. 최소값은 2 + 콘텐츠 타입 수.

        Raises:
            None
        """
        df = get_contents_cached()
        if df is not None and not df.empty:
            self.content_types = df["type"].unique().tolist()
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
        """
        임베딩되는 user 벡터의 차원을 반환합니다.

        Returns:
            int: user_dim
        """
        return self.user_dim

    def embed_user(self, user: Dict[str, Any]) -> np.ndarray:
        """
        사용자의 최근 로그 및 활동을 벡터로 변환합니다.

        Args:
            user (dict): 사용자 정보 및 최근 로그를 포함하는 딕셔너리.
                - "user_info" (dict): 사용자 관련 메타 정보 (예: {"id": 123, ...})
                - "recent_logs" (list of dict): 최근 활동 로그 리스트.
                  각 로그는 다음 키를 포함할 수 있습니다:
                    - "ratio" (float)
                    - "time" (float)
                    - "content_actual_type" (str)
                - "current_time": 현재 시점 (datetime 등), embedding 계산에는 사용되지 않음.

        Returns:
            np.ndarray: shape (user_dim,)
                - index 0: 평균 ratio
                - index 1: 평균 time
                - index 2~(2+N-1): 각 콘텐츠 타입별 비율 (N = num_content_types)
                - 이후: 0 padding

        Raises:
            None
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
                    (type_counts[t] / total_logs_for_known_types)
                    if total_logs_for_known_types > 0
                    else 0.0
                )
                for t in self.content_types
            ],
            dtype=np.float32,
        )

        vec = np.concatenate([[ratio_avg, time_avg], type_vec])
        if len(vec) < self.user_dim:
            vec = np.pad(vec, (0, self.user_dim - len(vec)), "constant")
        elif len(vec) > self.user_dim:
            vec = vec[: self.user_dim]
        return vec.astype(np.float32)

    def estimate_preference(self, state: np.ndarray) -> Dict[str, float]:
        """
        유저 임베딩 벡터에서 콘텐츠 타입별 선호도를 추정합니다.

        Args:
            state (np.ndarray): (user_dim,)
                - index 2부터 2+num_content_types-1 영역에 각 타입별 비율/선호도가 저장되어 있음.

        Returns:
            dict: 각 콘텐츠 타입별 선호도. 예: {"youtube": 0.5, "blog": 0.3, ...}

        Raises:
            None
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
    SBERT 기반으로 콘텐츠(text)를 임베딩합니다.

    - SBERT 임베딩: 768차원
    - 최종 content_dim 차원으로 맞춰 반환 (기본 768)
    """

    def __init__(
        self,
        content_dim: int = 768,
        model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    ):
        """
        Args:
            model_name (str): SBERT 모델 이름 (Hugging Face 허브).
            content_dim (int): 반환할 벡터 차원. SBERT pretrained_dim과 다른 경우
                               pretrained_dim으로 자동 조정됩니다.

        Raises:
            FileNotFoundError: 모델 로드 시 해당 이름의 SBERT 모델이 로컬/캐시에 존재하지 않을 경우 발생할 수 있음.
        """
        logging.info("Loading SBERT model '%s' ...", model_name)
        try:
            self.sbert_model = SentenceTransformer(model_name)
        except Exception as e:
            logging.error("SBERT 모델 로드 실패: %s", e)
            raise FileNotFoundError(
                f"SBERT model '{model_name}' could not be loaded."
            ) from e

        self.pretrained_dim = self.sbert_model.get_sentence_embedding_dimension()
        if content_dim == self.pretrained_dim:
            self.content_dim = content_dim
        else:
            logging.warning(
                "설정된 content_dim (%d)이 SBERT pretrained_dim (%d)과 다릅니다. "
                "pretrained_dim (%d)으로 맞춥니다.",
                content_dim,
                self.pretrained_dim,
                self.pretrained_dim,
            )
            self.content_dim = self.pretrained_dim

        df = get_contents_cached()
        if df is not None and not df.empty:
            self.content_types = df["type"].unique().tolist()
        else:
            self.content_types = ["youtube", "blog", "news"]

    def output_dim(self) -> int:
        """
        임베딩된 content 벡터의 차원을 반환합니다.

        Returns:
            int: content_dim
        """
        return self.content_dim

    def embed_content(self, content: Dict[str, Any]) -> np.ndarray:
        """
        콘텐츠(text)를 SBERT를 사용해 임베딩합니다.

        Args:
            content (dict):
                - "title" (str): 콘텐츠 제목
                - "description" (str): 콘텐츠 설명

        Returns:
            np.ndarray: shape (content_dim,)
                - index 0~(pretrained_dim-1): SBERT 임베딩 (float32)
                - 이후: 0 padding (필요 시)

        Raises:
            None
        """
        raw_text = content.get("title", "") + " " + content.get("description", "")
        raw_text = re.sub(r"<.*?>", "", raw_text).strip()

        if raw_text == "":
            return np.zeros(self.pretrained_dim, dtype=np.float32)

        try:
            sbert_emb = self.sbert_model.encode(
                [raw_text],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )[0]
            return sbert_emb.astype(np.float32)
        except Exception as e:
            logging.warning("SBERT inference failed: %s", e)
            return np.zeros(self.pretrained_dim, dtype=np.float32)


@register("doc2vec_content")
class Doc2VecContentEmbedder(BaseContentEmbedder):
    """
    Doc2Vec 기반으로 콘텐츠(text)를 임베딩합니다.

    - 사전 학습된 Doc2Vec 모델 로드 (vector_size=300)
    - 최종 content_dim 차원으로 맞춰 반환 (기본 300)
    """

    def __init__(
        self,
        model_path: str = "models/doc2vec.model",
        content_dim: int = 300,
    ):
        """
        Args:
            model_path (str): 디스크에 저장된 Doc2Vec 모델 경로.
            content_dim (int): 반환할 벡터 차원. Doc2Vec vector_size와 다른 경우
                               vector_size로 자동 조정됩니다.

        Raises:
            FileNotFoundError: model_path에 파일이 없을 경우.
        """
        logging.info("Loading Doc2Vec model from '%s' ...", model_path)
        try:
            self.doc2vec_model = Doc2Vec.load(model_path)
        except FileNotFoundError as e:
            logging.error("Doc2Vec 모델 로드 실패: %s", e)
            raise FileNotFoundError(
                f"Doc2Vec model at '{model_path}' not found."
            ) from e
        except Exception as e:
            logging.error("Doc2Vec 모델 로드 중 예외 발생: %s", e)
            raise

        self.pretrained_dim = self.doc2vec_model.vector_size
        if content_dim == self.pretrained_dim:
            self.content_dim = content_dim
        else:
            logging.warning(
                "설정된 content_dim (%d)이 Doc2Vec vector_size (%d)와 다릅니다. "
                "pretrained_dim (%d)으로 맞춥니다.",
                content_dim,
                self.pretrained_dim,
                self.pretrained_dim,
            )
            self.content_dim = self.pretrained_dim

        df = get_contents_cached()
        if df is not None and not df.empty:
            self.content_types = df["type"].unique().tolist()
        else:
            self.content_types = ["youtube", "blog", "news"]

    def output_dim(self) -> int:
        """
        임베딩된 content 벡터의 차원을 반환합니다.

        Returns:
            int: content_dim
        """
        return self.content_dim

    def embed_content(self, content: Dict[str, Any]) -> np.ndarray:
        """
        콘텐츠(text)를 Doc2Vec 모델로 inferencing하여 임베딩합니다.

        Args:
            content (dict):
                - "title" (str): 콘텐츠 제목
                - "description" (str): 콘텐츠 설명

        Returns:
            np.ndarray: shape (content_dim,)
                - index 0~(pretrained_dim-1): Doc2Vec inferencing 결과 벡터 (float32)
                - 이후: 0 padding (필요 시)

        Raises:
            None
        """
        raw_text = content.get("title", "") + " " + content.get("description", "")
        raw_text = re.sub(r"<.*?>", "", raw_text).strip()

        if raw_text == "":
            return np.zeros(self.pretrained_dim, dtype=np.float32)

        tokens: List[str] = raw_text.split()
        try:
            inferred_vec = self.doc2vec_model.infer_vector(tokens)
            return np.array(inferred_vec, dtype=np.float32)
        except Exception as e:
            logging.warning("Doc2Vec inference failed: %s", e)
            return np.zeros(self.pretrained_dim, dtype=np.float32)


@register("simple_content")
class SimpleContentEmbedder(BaseContentEmbedder):
    """
    콘텐츠 정보를 단순 연결하여 벡터로 변환하는 콘텐츠 임베더.

    Content 벡터 구성:
        - [0:N]: 사전학습 임베딩 (JSON 문자열에서 파싱)
        - [N:N+num_content_types]: 콘텐츠 타입 원핫 인코딩
        - 이후: 0 padding
    """

    def __init__(self, content_dim: int = 5):
        """
        Args:
            content_dim (int): 반환할 content 임베딩 벡터 차원.
                               최소값은 num_content_types.

        Raises:
            None
        """
        df = get_contents_cached()
        if df is not None and not df.empty:
            self.content_types = df["type"].unique().tolist()
        else:
            self.content_types = ["youtube", "blog", "news"]
        self.num_content_types = len(self.content_types)
        self.type_to_idx_map = {t: i for i, t in enumerate(self.content_types)}

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
        """
        임베딩된 content 벡터의 차원을 반환합니다.

        Returns:
            int: content_dim
        """
        return self.content_dim

    def embed_content(self, content: Dict[str, Any]) -> np.ndarray:
        """
        콘텐츠 정보를 임베딩 벡터로 변환합니다.

        Args:
            content (dict):
                - "embedding" (str): JSON 리스트 형태의 사전학습 임베딩 문자열.
                - "type" (str): 콘텐츠 타입 (예: "youtube", "blog", "news").

        Returns:
            np.ndarray: shape (content_dim,)
                - index 0~(pretrained_content_embedding_dim-1): 파싱된 사전학습 임베딩 (float32) 또는 0 벡터
                - index pretrained_content_embedding_dim~: 타입 원핫 인코딩
                - 부족 시: 0 padding

        Raises:
            None
        """
        # 사전학습 임베딩 파싱
        if self.pretrained_content_embedding_dim > 0:
            try:
                embedding_str = content.get("embedding", "")
                if not embedding_str:
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
                        arr = np.array(parsed_list, dtype=np.float32)
                        if len(arr) != self.pretrained_content_embedding_dim:
                            pretrained_emb = np.zeros(
                                self.pretrained_content_embedding_dim, dtype=np.float32
                            )
                        else:
                            pretrained_emb = arr
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning("사전학습 임베딩 파싱 실패: %s", e)
                pretrained_emb = np.zeros(
                    self.pretrained_content_embedding_dim, dtype=np.float32
                )
        else:
            pretrained_emb = np.array([], dtype=np.float32)

        # 타입 원핫 인코딩
        content_type_str = str(content.get("type", "")).lower()
        type_idx = self.type_to_idx_map.get(content_type_str, -1)
        type_onehot = np.zeros(self.num_content_types, dtype=np.float32)
        if type_idx != -1:
            type_onehot[type_idx] = 1.0

        final_vec = np.concatenate([pretrained_emb, type_onehot])
        if len(final_vec) < self.content_dim:
            final_vec = np.pad(
                final_vec, (0, self.content_dim - len(final_vec)), "constant"
            )
        elif len(final_vec) > self.content_dim:
            final_vec = final_vec[: self.content_dim]

        return final_vec.astype(np.float32)


@register("simple_concat")
class SimpleConcatEmbedder(BaseEmbedder):
    """
    SimpleUserEmbedder와 SimpleContentEmbedder를 조합한 임베더.
    """

    def __init__(self, user_embedder: Dict[str, Any], content_embedder: Dict[str, Any]):
        """
        Args:
            user_embedder (dict): 유저 임베더 설정
                - "type": 임베더 타입 (등록된 이름)
                - "params": 임베더 생성 시 필요한 파라미터 딕셔너리
            content_embedder (dict): 콘텐츠 임베더 설정
                - "type": 임베더 타입 (등록된 이름)
                - "params": 임베더 생성 시 필요한 파라미터 딕셔너리

        Raises:
            KeyError:
                만약 user_embedder 또는 content_embedder 딕셔너리에 "type" 또는 "params" 키가 없을 경우.
        """
        self.user_embedder = make(user_embedder["type"], **user_embedder["params"])
        self.content_embedder = make(
            content_embedder["type"], **content_embedder["params"]
        )

        self.content_types = self.content_embedder.content_types
        self.num_content_types = len(self.content_types)
        self.type_to_idx_map = {t: i for i, t in enumerate(self.content_types)}

        self.user_dim = self.user_embedder.user_dim
        self.content_dim = self.content_embedder.content_dim

        super().__init__(self.user_embedder, self.content_embedder)
