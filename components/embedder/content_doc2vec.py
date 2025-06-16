import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec

from components.core.base import BaseContentEmbedder
from components.database.db_utils import get_contents
from components.registry import register

# 모듈-레벨 캐시
_doc2vec_model_singleton: Optional[Doc2Vec] = None


def get_doc2vec_model(model_path: str) -> Doc2Vec:
    """싱글톤 Doc2Vec 모델 로더."""
    global _doc2vec_model_singleton
    if _doc2vec_model_singleton is None:
        logging.info("Doc2Vec 모델 '%s' 최초 로딩...", model_path)
        _doc2vec_model_singleton = Doc2Vec.load(model_path)
    return _doc2vec_model_singleton


@register("doc2vec_content")
class Doc2VecContentEmbedder(BaseContentEmbedder):
    """Doc2Vec 기반 텍스트 콘텐츠 임베더.

    사전 학습된 Doc2Vec 모델을 싱글톤으로 로드하여
    콘텐츠 텍스트를 벡터로 변환합니다.
    출력 차원은 모델의 vector_size로 자동 설정됩니다.
    """

    def __init__(
        self,
        model_path: str = "models/doc2vec.model",
        content_dim: int = 300,
        all_contents_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Doc2VecContentEmbedder 생성자.

        Args:
            model_path (str): Doc2Vec 모델 파일 경로.
            content_dim (int): 요청 임베딩 차원 (모델 vector_size와 일치하지 않으면 무시).
            all_contents_df (Optional[pd.DataFrame]): 외부에서 주입된 전체 콘텐츠 DataFrame.
        """
        self.doc2vec_model = get_doc2vec_model(model_path)
        self.pretrained_dim = self.doc2vec_model.vector_size

        if content_dim != self.pretrained_dim:
            logging.warning(
                "content_dim (%d) != Doc2Vec vector_size (%d). pretrained_dim (%d) 사용합니다.",
                content_dim,
                self.pretrained_dim,
                self.pretrained_dim,
            )
        self.content_dim = self.pretrained_dim

        self.all_contents_df = (
            all_contents_df if all_contents_df is not None else get_contents()
        )
        if not self.all_contents_df.empty:
            self.content_types: List[str] = (
                self.all_contents_df["type"].unique().tolist()
            )
        else:
            self.content_types = ["news", "blog", "youtube"]

    def output_dim(self) -> int:
        """임베딩 벡터의 차원을 반환합니다.

        Returns:
            int: 임베딩 벡터 차원.
        """
        return self.content_dim

    def embed_content(self, content: Dict[str, Any]) -> np.ndarray:
        """단일 콘텐츠를 Doc2Vec으로 임베딩합니다.

        Args:
            content (Dict[str, Any]):
                {
                    "title": str,
                    "description": str
                }

        Returns:
            np.ndarray: 콘텐츠 임베딩 벡터 (shape: [content_dim]).
        """
        raw_text = f"{content.get('title', '')} {content.get('description', '')}"
        text = re.sub(r"<.*?>", "", raw_text).strip()
        tokens: List[str] = text.split() if text else []

        try:
            vector = self.doc2vec_model.infer_vector(tokens)
            emb = np.array(vector, dtype=np.float32)
        except Exception as err:
            logging.warning("Doc2Vec 추론 실패: %s", err)
            emb = np.zeros(self.pretrained_dim, dtype=np.float32)

        return emb
