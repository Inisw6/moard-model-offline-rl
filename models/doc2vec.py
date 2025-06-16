import logging
import os
import re
from typing import List

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from components.database.db_utils import get_contents


def preprocess_text(text: str) -> List[str]:
    """텍스트에서 HTML 태그를 제거하고 공백 기준으로 토큰화합니다.

    Args:
        text (str): 원본 텍스트 (HTML 태그 포함 가능).

    Returns:
        List[str]: 전처리된 토큰 리스트.
    """
    cleaned = re.sub(r"<.*?>", "", text).strip()
    return cleaned.split()


def build_tagged_documents(df: pd.DataFrame) -> List[TaggedDocument]:
    """DataFrame의 'title' 및 'description' 컬럼을 TaggedDocument로 변환합니다.

    Args:
        df (pd.DataFrame): 'title' 및 'description' 컬럼을 포함하는 콘텐츠 DataFrame.

    Returns:
        List[TaggedDocument]: 각 행마다 토큰과 태그(문서 인덱스)를 가진 리스트.

    Raises:
        KeyError: 'title' 또는 'description' 컬럼이 없을 경우.
    """
    if "title" not in df.columns or "description" not in df.columns:
        missing = {"title", "description"} - set(df.columns)
        raise KeyError(f"필수 컬럼 누락: {missing}")

    documents: List[TaggedDocument] = []
    for idx, row in df.iterrows():
        title = str(row["title"] or "")
        description = str(row["description"] or "")
        tokens = preprocess_text(f"{title} {description}")
        documents.append(TaggedDocument(words=tokens, tags=[str(idx)]))
    return documents


def train_and_save(
    documents: List[TaggedDocument],
    vector_size: int = 300,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 40,
    save_path: str = "models/doc2vec.model",
) -> None:
    """Doc2Vec 모델을 학습하고 지정 경로에 저장합니다.

    Args:
        documents (List[TaggedDocument]): 학습에 사용할 TaggedDocument 리스트.
        vector_size (int): 임베딩 벡터 차원. 기본값 300.
        window (int): 컨텍스트 윈도우 크기. 기본값 5.
        min_count (int): 단어 최소 빈도. 기본값 2.
        epochs (int): 학습 반복 횟수. 기본값 40.
        save_path (str): 모델 파일 저장 경로.

    Raises:
        ValueError: documents가 비어 있거나 파라미터가 비정상적일 경우.
        OSError: 디렉토리 생성 또는 파일 저장 실패 시.
    """
    if not documents:
        raise ValueError("학습에 사용할 TaggedDocument 리스트가 비어 있습니다.")

    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=os.cpu_count() or 4,
        epochs=epochs,
        dm=1,
        seed=42,
    )

    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

    dirname = os.path.dirname(save_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    model.save(save_path)
    logging.info("Doc2Vec 모델을 '%s'에 저장했습니다.", save_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    try:
        df = get_contents()
    except Exception as e:
        logging.error("get_contents() 호출 중 오류: %s", e)
        raise

    if df.empty:
        logging.warning("불러온 콘텐츠 DataFrame이 비어 있습니다.")
    else:
        try:
            docs = build_tagged_documents(df)
            train_and_save(
                documents=docs,
                vector_size=300,
                window=5,
                min_count=2,
                epochs=40,
                save_path="models/doc2vec.model",
            )
        except Exception as e:
            logging.error("모델 학습 또는 저장 중 오류: %s", e)
            raise
