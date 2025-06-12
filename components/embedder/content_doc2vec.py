import logging
import re
from typing import Optional

import numpy as np
from gensim.models.doc2vec import Doc2Vec

from components.core.base import BaseContentEmbedder
from components.database.db_utils import get_contents
from components.registry import register


@register("doc2vec_content")
class Doc2VecContentEmbedder(BaseContentEmbedder):
    """Doc2Vec 기반 텍스트 콘텐츠 임베더.

    - 사전 학습된 Doc2Vec 모델로 텍스트 기반 콘텐츠 임베딩.
    - 기본 임베딩 차원은 300, 지정된 content_dim으로 패딩/잘림.
    """

    def __init__(
        self,
        model_path: str = "models/doc2vec.model",
        content_dim: int = 300,
        all_contents_df: Optional[object] = None,  # 의존성 주입 가능
    ) -> None:
        """Doc2VecContentEmbedder 생성자.

        Args:
            model_path (str): Doc2Vec 모델 파일 경로.
            content_dim (int): 출력 임베딩 차원 (Doc2Vec vector_size와 다르면 자동 보정).
            all_contents_df (Optional[pandas.DataFrame]): 외부 콘텐츠 DataFrame.
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
        """콘텐츠 임베딩 벡터의 차원을 반환합니다.

        Returns:
            int: 임베딩 벡터 차원.
        """
        return self.content_dim

    def embed_content(self, content: dict) -> np.ndarray:
        """Doc2Vec로 콘텐츠를 임베딩합니다.

        Args:
            content (dict): {"title": str, "description": str}

        Returns:
            np.ndarray: (content_dim,) 크기의 벡터.
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
