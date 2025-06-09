import json
import logging
import re
from typing import Optional

import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sentence_transformers import SentenceTransformer

from components.base import BaseContentEmbedder, BaseEmbedder, BaseUserEmbedder
from components.db_utils import get_contents
from components.registry import register


@register("simple_user")
class SimpleUserEmbedder(BaseUserEmbedder):
    """
    SimpleUserEmbedder:
    - 사용자의 최근 로그와 활동을 단순 연결하여 고정 차원 벡터로 변환
    - 벡터 구성: [평균 ratio, 평균 time, 콘텐츠 타입별 비율, 남은 부분 0 패딩]
    """

    def __init__(
        self,
        user_dim: int = 30,
        all_contents_df: Optional[object] = None,  # 의존성 주입 가능
    ):
        """
        Args:
            user_dim (int): 출력할 유저 임베딩 벡터의 차원
            all_contents_df (Optional[pandas.DataFrame]): 테스트용으로 외부에서 전달할 콘텐츠 DataFrame.
                                                         None일 경우 get_contents() 호출.
        """
        # 의존성 주입된 DataFrame이 없다면 실제 DB에서 가져옴
        self.all_contents_df = (
            all_contents_df if all_contents_df is not None else get_contents()
        )
        if not self.all_contents_df.empty:
            # 고유한 콘텐츠 타입 목록을 추출
            self.content_types = self.all_contents_df["type"].unique().tolist()
        else:
            # 데이터가 비어 있으면 기본값 사용
            self.content_types = ["youtube", "blog", "news"]

        self.num_content_types = len(self.content_types)
        # 콘텐츠 타입을 인덱스로 매핑
        self.type_to_idx_map = {t: i for i, t in enumerate(self.content_types)}

        # 최소 user_dim = 2 (ratio, time) + 콘텐츠 타입 개수
        min_user_dim = 2 + self.num_content_types
        if user_dim < min_user_dim:
            logging.warning(
                "user_dim (%d)이 너무 작습니다. %d로 조정합니다...",
                user_dim,
                min_user_dim,
            )
            self.user_dim = min_user_dim
        else:
            self.user_dim = user_dim

    def output_dim(self) -> int:
        """
        Returns:
            int: 유저 임베딩 벡터의 차원
        """
        return self.user_dim

    def embed_user(self, user: dict) -> np.ndarray:
        """
        사용자의 최근 로그 및 활동을 벡터로 변환

        Args:
            user (dict): {
                "user_info": {...},          # 사용자 메타데이터 (예: ID 등)
                "recent_logs": [             # 로그 정보 리스트
                    {"ratio": float, "time": float, "content_actual_type": str, ...}, ...
                ],
                "current_time": datetime 객체 (선택)
            }

        Returns:
            np.ndarray of shape (user_dim,):
                [0] = 최근 로그의 평균 ratio
                [1] = 최근 로그의 평균 time
                [2:2+N] = 콘텐츠 타입별 비율 (N = 콘텐츠 타입 개수)
                [나머지] = 0 패딩
        """
        logs = user.get("recent_logs", [])
        if not logs:
            # 로그가 없으면 전부 0 벡터 반환
            return np.zeros(self.user_dim, dtype=np.float32)

        # 로그 내 ratio와 time의 평균 계산
        ratio_avg = np.mean([l.get("ratio", 0.0) for l in logs])
        time_avg = np.mean([l.get("time", 0.0) for l in logs])

        # 콘텐츠 타입별 카운트 초기화
        type_counts = {t: 0 for t in self.content_types}
        for log in logs:
            actual_type = str(log.get("content_actual_type", "")).lower()
            if actual_type in self.type_to_idx_map:
                type_counts[actual_type] += 1

        total_known = sum(type_counts.values())
        # 각 타입별 비율 벡터 생성
        type_vec = np.array(
            [
                (type_counts[t] / total_known) if total_known > 0 else 0
                for t in self.content_types
            ]
        )

        # [평균 ratio, 평균 time]과 타입 비율 벡터를 연결
        vec = np.concatenate([[ratio_avg, time_avg], type_vec])
        if len(vec) < self.user_dim:
            # 벡터가 작으면 0으로 패딩
            vec = np.pad(vec, (0, self.user_dim - len(vec)), "constant")
        elif len(vec) > self.user_dim:
            # 벡터가 크면 잘라냄
            vec = vec[: self.user_dim]

        return vec.astype(np.float32)

    def estimate_preference(self, state: np.ndarray) -> dict:
        """
        유저 임베딩 벡터에서 콘텐츠 타입별 선호도 추정

        Args:
            state (np.ndarray): 길이가 최소 2 + 콘텐츠 타입 개수인 벡터

        Returns:
            dict: {타입명: 선호도(float)}. 벡터 길이가 부족하면 0 반환
        """
        if len(state) < 2 + self.num_content_types:
            return {t: 0.0 for t in self.content_types}

        # 2:2+N 구간이 타입별 비율 정보
        type_prefs = state[2 : 2 + self.num_content_types]
        return {
            self.content_types[i]: float(type_prefs[i]) for i in range(len(type_prefs))
        }


@register("sbert_content")
class SbertContentEmbedder(BaseContentEmbedder):
    """
    SbertContentEmbedder:
    - 사전 학습된 SBERT(Sentence-BERT) 모델을 사용하여 텍스트 기반 콘텐츠를 임베딩
    - 기본 SBERT 임베딩 차원은 768이며, 지정된 content_dim으로 패딩/자르기
    """

    def __init__(
        self,
        content_dim: int = 768,
        model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        all_contents_df: Optional[object] = None,  # 의존성 주입 가능
    ):
        """
        Args:
            content_dim (int): 출력할 벡터 차원. 사전학습 차원과 다르면 사전학습 차원으로 강제 설정
            model_name (str): HuggingFace SBERT 모델 이름
            all_contents_df (Optional[pandas.DataFrame]): 테스트용으로 외부에서 전달할 콘텐츠 DataFrame.
                                                         None일 경우 get_contents() 호출.
        """
        # SBERT 모델 로드
        logging.info("SBERT 모델 '%s' 로딩 중...", model_name)
        # todo: 모델 로드 부분 싱글톤으로 전환 가능
        self.sbert_model = SentenceTransformer(model_name)
        self.pretrained_dim = self.sbert_model.get_sentence_embedding_dimension()

        # 요청 차원이 사전학습 차원과 다르면 사전학습 차원으로 설정
        if content_dim == self.pretrained_dim:
            self.content_dim = content_dim
        else:
            logging.warning(
                "설정된 content_dim (%d) != SBERT pretrained_dim (%d). "
                "pretrained_dim (%d) 사용합니다.",
                content_dim,
                self.pretrained_dim,
                self.pretrained_dim,
            )
            self.content_dim = self.pretrained_dim

        # 의존성 주입된 DataFrame이 없다면 실제 DB에서 가져옴
        self.all_contents_df = (
            all_contents_df if all_contents_df is not None else get_contents()
        )
        if not self.all_contents_df.empty:
            self.content_types = self.all_contents_df["type"].unique().tolist()
        else:
            self.content_types = ["youtube", "blog", "news"]

    def output_dim(self) -> int:
        """
        Returns:
            int: 콘텐츠 임베딩 벡터의 차원
        """
        return self.content_dim

    def embed_content(self, content: dict) -> np.ndarray:
        """
        SBERT를 이용해 콘텐츠를 임베딩

        Args:
            content (dict):
                - "title": str, 콘텐츠 제목
                - "description": str, 콘텐츠 설명

        Returns:
            np.ndarray of shape (content_dim,):
                [0:pretrained_dim] = SBERT 임베딩(float32),
                필요한 경우 0으로 패딩하거나 잘라냄.
        """
        # 제목과 설명을 합쳐서 HTML 태그 제거
        raw_text = content.get("title", "") + content.get("description", "")
        raw_text = re.sub(r"<.*?>", "", raw_text)

        if raw_text == "":
            # 빈 문자열일 경우 전부 0 벡터 반환
            sbert_emb = np.zeros(self.pretrained_dim, dtype=np.float32)
        else:
            try:
                # SBERT 모델로 텍스트 인코딩
                sbert_emb = self.sbert_model.encode(
                    [raw_text],
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )[0]
                sbert_emb = sbert_emb.astype(np.float32)
            except Exception as e:
                logging.warning("SBERT 추론 실패: %s", e)
                sbert_emb = np.zeros(self.pretrained_dim, dtype=np.float32)

        return sbert_emb


@register("doc2vec_content")
class Doc2VecContentEmbedder(BaseContentEmbedder):
    """
    Doc2VecContentEmbedder:
    - 사전 학습된 Doc2Vec 모델을 사용하여 텍스트 기반 콘텐츠를 임베딩
    - 기본 Doc2Vec 차원은 300이며, 지정된 content_dim으로 패딩/자르기
    """

    def __init__(
        self,
        model_path: str = "models/doc2vec.model",
        content_dim: int = 300,
        all_contents_df: Optional[object] = None,  # 의존성 주입 가능
    ):
        """
        Args:
            model_path (str): 디스크에 저장된 Doc2Vec 모델 경로
            content_dim (int): 출력할 벡터 차원. Doc2Vec vector_size와 일치해야 함
            all_contents_df (Optional[pandas.DataFrame]): 테스트용으로 외부에서 전달할 콘텐츠 DataFrame.
                                                         None일 경우 get_contents() 호출.
        """
        # Doc2Vec 모델 로드
        logging.info("Doc2Vec 모델 '%s' 로딩 중...", model_path)
        # todo: 모델 로드 부분 싱글톤으로 전환 가능
        self.doc2vec_model = Doc2Vec.load(model_path)
        self.pretrained_dim = self.doc2vec_model.vector_size

        # 요청 차원이 사전학습 차원과 다르면 사전학습 차원으로 설정
        if content_dim == self.pretrained_dim:
            self.content_dim = content_dim
        else:
            logging.warning(
                "설정된 content_dim (%d) != Doc2Vec vector_size (%d). "
                "pretrained_dim (%d) 사용합니다.",
                content_dim,
                self.pretrained_dim,
                self.pretrained_dim,
            )
            self.content_dim = self.pretrained_dim

        # 의존성 주입된 DataFrame이 없다면 실제 DB에서 가져옴
        self.all_contents_df = (
            all_contents_df if all_contents_df is not None else get_contents()
        )
        if not self.all_contents_df.empty:
            self.content_types = self.all_contents_df["type"].unique().tolist()
        else:
            self.content_types = ["youtube", "blog", "news"]

    def output_dim(self) -> int:
        """
        Returns:
            int: 콘텐츠 임베딩 벡터의 차원
        """
        return self.content_dim

    def embed_content(self, content: dict) -> np.ndarray:
        """
        Doc2Vec를 이용해 콘텐츠를 임베딩

        Args:
            content (dict):
                - "title": str, 콘텐츠 제목
                - "description": str, 콘텐츠 설명

        Returns:
            np.ndarray of shape (content_dim,):
                [0:pretrained_dim] = Doc2Vec 임베딩(float32),
                필요한 경우 0으로 패딩하거나 잘라냄.
        """
        # 제목과 설명을 합쳐서 HTML 태그 제거 후 토큰화
        raw_text = content.get("title", "") + " " + content.get("description", "")
        raw_text = re.sub(r"<.*?>", "", raw_text).strip()

        if raw_text == "":
            tokens = []
        else:
            # 단순 공백 분할 토큰화. 필요 시 konlpy 등 사용 가능
            tokens = raw_text.split()

        try:
            # Doc2Vec 모델로 토큰 리스트를 벡터로 추론
            inferred_vec = self.doc2vec_model.infer_vector(tokens)
            doc2vec_emb = np.array(inferred_vec, dtype=np.float32)
        except Exception as e:
            logging.warning("Doc2Vec 추론 실패: %s", e)
            doc2vec_emb = np.zeros(self.pretrained_dim, dtype=np.float32)

        return doc2vec_emb


@register("simple_content")
class SimpleContentEmbedder(BaseContentEmbedder):
    """
    SimpleContentEmbedder:
    - 사전 저장된 임베딩(JSON 문자열)과 콘텐츠 타입을 결합하여 벡터 생성
    - 벡터 구성: [사전학습 임베딩, 타입 원핫 인코딩, 남은 부분 0 패딩]
    """

    def __init__(
        self,
        content_dim: int = 5,
        all_contents_df: Optional[object] = None,  # 의존성 주입 가능
    ):
        """
        Args:
            content_dim (int): 출력할 콘텐츠 임베딩 벡터의 차원.
                              콘텐츠 타입 개수 이상이어야 함.
            all_contents_df (Optional[pandas.DataFrame]): 테스트용으로 외부에서 전달할 콘텐츠 DataFrame.
                                                         None일 경우 get_contents() 호출.
        """
        # 의존성 주입된 DataFrame이 없다면 실제 DB에서 가져옴
        self.all_contents_df = (
            all_contents_df if all_contents_df is not None else get_contents()
        )
        if not self.all_contents_df.empty:
            self.content_types = self.all_contents_df["type"].unique().tolist()
        else:
            self.content_types = ["youtube", "blog", "news"]

        self.num_content_types = len(self.content_types)
        self.type_to_idx_map = {t: i for i, t in enumerate(self.content_types)}

        # 사전학습 임베딩을 위한 차원 확보
        self.pretrained_content_embedding_dim = content_dim - self.num_content_types
        if self.pretrained_content_embedding_dim < 0:
            logging.warning(
                "content_dim (%d)이 콘텐츠 타입 수(%d)보다 작습니다. "
                "사전학습 임베딩 차원을 0으로 설정하고 content_dim을 %d로 사용합니다.",
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
        Returns:
            int: 콘텐츠 임베딩 벡터의 차원
        """
        return self.content_dim

    def embed_content(self, content: dict) -> np.ndarray:
        """
        콘텐츠 사전 학습 임베딩과 타입 정보를 결합하여 벡터 생성

        Args:
            content (dict):
                - "embedding": str, JSON으로 인코딩된 숫자 리스트 (사전학습 임베딩)
                - "type": str, 콘텐츠 타입 (예: "youtube", "blog", "news")

        Returns:
            np.ndarray of shape (content_dim,):
                [0:N]   = 사전학습 임베딩(float32) 또는 값이 없으면 0
                [N:N+num_content_types] = 타입 원핫 인코딩
                [나머지] = 0 패딩
        """
        # 사전학습 임베딩 문자열 파싱
        if self.pretrained_content_embedding_dim > 0:
            try:
                embedding_str = content.get("embedding")
                if embedding_str is None or embedding_str == "":
                    # 임베딩이 없으면 0 벡터
                    pretrained_emb = np.zeros(
                        self.pretrained_content_embedding_dim, dtype=np.float32
                    )
                else:
                    parsed_list = json.loads(embedding_str)
                    # 리스트 형태 및 모든 요소가 숫자인지 확인
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
                            # 차원 불일치 시 0 벡터
                            pretrained_emb = np.zeros(
                                self.pretrained_content_embedding_dim, dtype=np.float32
                            )
                        else:
                            pretrained_emb = pretrained_emb_list
            except (json.JSONDecodeError, ValueError):
                # JSON 파싱 오류 시 0 벡터
                pretrained_emb = np.zeros(
                    self.pretrained_content_embedding_dim, dtype=np.float32
                )
        else:
            # 사전학습 임베딩 영역이 없으면 빈 배열
            pretrained_emb = np.array([], dtype=np.float32)

        # 콘텐츠 타입에 대한 원핫 인코딩 생성
        content_type_str = content.get("type", "").lower()
        type_idx = self.type_to_idx_map.get(content_type_str, -1)
        type_onehot = np.zeros(self.num_content_types, dtype=np.float32)
        if type_idx != -1:
            type_onehot[type_idx] = 1.0

        # 사전학습 임베딩 + 원핫 인코딩 연결
        final_vec = np.concatenate([pretrained_emb, type_onehot])
        if len(final_vec) != self.content_dim:
            if len(final_vec) < self.content_dim:
                # 벡터가 작으면 0으로 패딩
                final_vec = np.pad(
                    final_vec, (0, self.content_dim - len(final_vec)), "constant"
                )
            else:
                # 벡터가 크면 잘라냄
                final_vec = final_vec[: self.content_dim]

        return final_vec.astype(np.float32)


@register("simple_concat")
class SimpleConcatEmbedder(BaseEmbedder):
    """
    SimpleConcatEmbedder:
    - SimpleUserEmbedder와 SimpleContentEmbedder를 조합하여 하나의 상위 레벨 임베더를 구성
    - 지정된 user_embedder와 content_embedder에 위임(delegate)하여 임베딩 수행
    """

    def __init__(self, user_embedder: dict, content_embedder: dict):
        """
        Args:
            user_embedder (dict): {
                "type": str, 유저 임베더 타입 이름 (예: "simple_user"),
                "params": dict, 해당 임베더의 인자
            }
            content_embedder (dict): {
                "type": str, 콘텐츠 임베더 타입 이름 (예: "simple_content"),
                "params": dict, 해당 임베더의 인자
            }
        """
        from components.registry import make

        # 레지스트리에서 user 임베더 인스턴스 생성
        self.user_embedder = make(user_embedder["type"], **user_embedder["params"])
        # 레지스트리에서 content 임베더 인스턴스 생성
        self.content_embedder = make(
            content_embedder["type"], **content_embedder["params"]
        )

        # content_embedder로부터 콘텐츠 타입 목록과 매핑 정보 가져오기
        self.content_types = self.content_embedder.content_types
        self.num_content_types = len(self.content_types)
        self.type_to_idx_map = {t: i for i, t in enumerate(self.content_types)}

        # user_dim과 content_dim 정보 저장
        self.user_dim = self.user_embedder.user_dim
        self.content_dim = self.content_embedder.content_dim

        # BaseEmbedder 초기화 (user_embedder, content_embedder를 인자로 전달)
        super().__init__(self.user_embedder, self.content_embedder)
