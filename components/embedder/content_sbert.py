import logging
import re
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from components.core.base import BaseContentEmbedder
from components.database.db_utils import get_contents
from components.registry import register


@register("sbert_content")
class SbertContentEmbedder(BaseContentEmbedder):
    """SBERT 기반 텍스트 콘텐츠 임베더.

    - 사전 학습된 SBERT(Sentence-BERT) 모델로 텍스트 기반 콘텐츠 임베딩.
    - 기본 임베딩 차원은 768, 지정된 content_dim으로 패딩/잘림.
    """

    def __init__(
        self,
        content_dim: int = 768,
        model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        all_contents_df: Optional[object] = None,  # 의존성 주입 가능
    ) -> None:
        """SbertContentEmbedder 생성자.

        Args:
            content_dim (int): 출력 임베딩 차원 (SBERT 모델 차원과 다르면 자동 보정).
            model_name (str): 사용할 SBERT 모델명 (HuggingFace 호환).
            all_contents_df (Optional[pandas.DataFrame]): 외부 콘텐츠 DataFrame.
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
        """콘텐츠 임베딩 벡터의 차원을 반환합니다.

        Returns:
            int: 임베딩 벡터 차원.
        """
        return self.content_dim

    def embed_content(self, content: dict) -> np.ndarray:
        """SBERT로 콘텐츠를 임베딩합니다.

        Args:
            content (dict): {"title": str, "description": str}

        Returns:
            np.ndarray: (content_dim,) 크기의 벡터.
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
