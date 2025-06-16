import logging
import re
from typing import Any, Dict, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from components.core.base import BaseContentEmbedder
from components.database.db_utils import get_contents
from components.registry import register

# 모듈-레벨 캐시
_sbert_model_singleton: Optional[SentenceTransformer] = None


def get_sbert_model(model_name: str) -> SentenceTransformer:
    """싱글톤 SBERT 모델 로더."""
    global _sbert_model_singleton
    if _sbert_model_singleton is None:
        logging.info("SBERT 모델 '%s' 최초 로딩...", model_name)
        _sbert_model_singleton = SentenceTransformer(model_name)
    return _sbert_model_singleton


@register("sbert_content")
class SbertContentEmbedder(BaseContentEmbedder):
    """SBERT 기반 텍스트 콘텐츠 임베더."""

    def __init__(
        self,
        content_dim: int = 768,
        model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        all_contents_df: Optional[Any] = None,
    ) -> None:
        """SbertContentEmbedder 생성자.

        Args:
            content_dim (int): 요청 임베딩 차원 (모델 차원과 다르면 무시).
            model_name (str): SBERT 모델 이름.
            all_contents_df (Optional[pd.DataFrame]): 외부 콘텐츠 DataFrame 주입.
        """
        self.sbert_model = get_sbert_model(model_name)
        self.pretrained_dim = self.sbert_model.get_sentence_embedding_dimension()

        if content_dim != self.pretrained_dim:
            logging.warning(
                "content_dim (%d) != pretrained_dim (%d). pretrained_dim 사용합니다.",
                content_dim,
                self.pretrained_dim,
            )
        self.content_dim = self.pretrained_dim

        self.all_contents_df = (
            all_contents_df if all_contents_df is not None else get_contents()
        )
        if not self.all_contents_df.empty:
            self.content_types = self.all_contents_df["type"].unique().tolist()
        else:
            self.content_types = ["news", "blog", "youtube"]

    def output_dim(self) -> int:
        """임베딩 벡터 차원 반환."""
        return self.content_dim

    def embed_content(self, content: Dict[str, Any]) -> np.ndarray:
        """단일 콘텐츠를 SBERT로 임베딩."""
        raw = f"{content.get('title','')} {content.get('description','')}"
        text = re.sub(r"<.*?>", "", raw).strip()

        if not text:
            return np.zeros(self.content_dim, dtype=np.float32)
        try:
            emb = self.sbert_model.encode(
                [text],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )[0].astype(np.float32)
        except Exception as err:
            logging.warning("SBERT 추론 실패: %s", err)
            emb = np.zeros(self.content_dim, dtype=np.float32)
        return emb
