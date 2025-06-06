import re
import os
import logging
import pandas as pd
from typing import List

from components.db_utils import get_contents

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# 추후 성능 최적화: 병렬처리 및 메모리 관리 필요


def preprocess_text(text: str) -> List[str]:
    """
    텍스트를 전처리하여 토큰 리스트로 반환합니다.

    Args:
        text (str): 원본 텍스트 (HTML 태그가 포함될 수 있음).

    Returns:
        List[str]:
            HTML 태그 제거 및 공백 기준 분리된 토큰 리스트.

    Raises:
        None
    """
    text = re.sub(r"<.*?>", "", text)
    text = text.strip()
    tokens = text.split()
    return tokens


def build_tagged_documents(df: pd.DataFrame) -> List[TaggedDocument]:
    """
    DataFrame의 각 행을 TaggedDocument 객체로 변환하여 리스트로 반환합니다.

    Args:
        df (pd.DataFrame): 'title' 및 'description' 컬럼을 포함하는 콘텐츠 데이터프레임.

    Returns:
        List[TaggedDocument]:
            각 행마다 전처리된 토큰과 태그(문서 식별자)를 가진 TaggedDocument 객체 리스트.

    Raises:
        KeyError:
            만약 DataFrame에 'title' 또는 'description' 컬럼이 없을 경우.
    """
    documents: List[TaggedDocument] = []
    for idx, row in df.iterrows():
        title = str(row.get("title") or "")
        description = str(row.get("description") or "")
        raw_text = f"{title} {description}"
        tokens = preprocess_text(raw_text)
        tag = [str(idx)]
        documents.append(TaggedDocument(words=tokens, tags=tag))
    return documents


def train_and_save(
    documents: List[TaggedDocument],
    vector_size: int = 300,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 40,
    save_path: str = "models/doc2vec.model",
) -> None:
    """
    Doc2Vec 모델을 학습하고 파일로 저장합니다.

    Args:
        documents (List[TaggedDocument]): 학습에 사용할 TaggedDocument 객체 리스트.
        vector_size (int): 학습할 임베딩 벡터의 차원 수. 기본값은 300.
        window (int): 컨텍스트 윈도우 크기. 기본값은 5.
        min_count (int): 단어 최소 빈도 수 (빈도 미만 단어 제외). 기본값은 2.
        epochs (int): 학습 반복 수. 기본값은 40.
        save_path (str): 모델을 저장할 파일 경로. 디렉토리가 없으면 생성합니다. 기본값은 "models/doc2vec.model".

    Returns:
        None: 학습된 모델을 지정된 경로에 저장하고 로그를 출력합니다.

    Raises:
        OSError:
            설정된 save_path의 디렉토리를 생성하거나 모델을 저장할 때 파일 시스템 오류가 발생할 경우.
        ValueError:
            documents 리스트가 비어 있거나, Doc2Vec 빌드/학습 중 파라미터가 잘못된 경우.
    """
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=os.cpu_count() or 4,
        epochs=epochs,
        dm=1,  # 1 = PV-DM, 0 = PV-DBOW
        seed=42,
    )

    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    model.save(save_path)
    logging.info(f"Doc2Vec 모델이 '{save_path}'에 저장되었습니다.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # 1) 콘텐츠 데이터 로드
    try:
        df = get_contents()
    except Exception as e:
        logging.error(f"get_contents 호출 중 예외 발생: {e}")
        raise

    if df.empty:
        logging.warning(
            "get_contents()로 불러온 데이터프레임이 비어 있습니다. 데이터 확인하세요."
        )
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
            logging.error(f"모델 학습 또는 저장 중 예외 발생: {e}")
            raise
