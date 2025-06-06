import re
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from components.db_utils import get_contents  

def preprocess_text(text: str) -> list[str]:
    """
    전처리
    """
    text = re.sub(r"<.*?>", "", text)
    text = text.strip()
    # 공백 기준으로 단어 분리
    tokens = text.split()
    return tokens

def build_tagged_documents(df: pd.DataFrame) -> list[TaggedDocument]:
   
    documents = []
    for idx, row in df.iterrows():
        # words
        title = row.get("title","")
        description = row.get("description","")
        raw_text = f"{title} {description}"
        tokens = preprocess_text(raw_text)
        # tag : 문서식별자
        tag = [str(idx)]
        # TaggedDocument진행하기.
        documents.append(TaggedDocument(words=tokens, tags=tag))
    return documents

def train_and_save(documents: list[TaggedDocument],
                   vector_size: int = 300,
                   window: int = 5,
                   min_count: int = 2,
                   epochs: int = 40,
                   save_path: str = "models/doc2vec.model"):
    """
    Doc2Vec 모델 학습 후, "models/doc2vec.model" 저장
    """
    # 모델 초기화
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
        dm=1, # 1 = PV-DM, 0 = PV-DBOW
        seed=42
    )

    # 단어 사전(vocabulary) 생성
    model.build_vocab(documents)

    # 학습
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

    # 저장 디렉토리가 없으면 생성
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Doc2Vec 모델이 '{save_path}'에 저장되었습니다.")

if __name__ == "__main__":
    # 1) 콘텐츠 데이터 로드
    df = get_contents()
    if df.empty:
        print("get_contents()로 불러온 데이터프레임이 비어 있습니다. 데이터 확인하세요.")
    else:
        docs = build_tagged_documents(df)
        # 2) 모델 학습 및 저장 (원하는 파라미터로 조정 가능)
        train_and_save(
            documents=docs,
            vector_size=300, # 임베딩 차원
            window=5,
            min_count=2,
            epochs=40,
            save_path="models/doc2vec.model"
        )
