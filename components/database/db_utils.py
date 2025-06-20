import uuid
from typing import List

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
from sqlalchemy.types import Enum as SAEnum

# SQLite 데이터베이스 설정
DB_PATH: str = "./data/sample_recsys.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# 데이터베이스 테이블 정의 (DDL 기반)
class User(Base):
    """사용자(users) 테이블 ORM 모델."""

    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    uuid = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True)

    recommendations = relationship("Recommendation", back_populates="user")
    stock_logs = relationship("StockLog", back_populates="user")
    user_logs = relationship("UserLog", back_populates="user")


class StockInfo(Base):
    """종목(stock_info) 테이블 ORM 모델."""

    __tablename__ = "stock_info"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    code = Column(String(255))
    industry_detail = Column(String(255))
    industry_type = Column(String(255))
    market_type = Column(String(255))
    name = Column(String(255))

    search_queries = relationship("SearchQuery", back_populates="stock_info")


class SearchQuery(Base):
    """검색 쿼리(search_queries) 테이블 ORM 모델."""

    __tablename__ = "search_queries"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    searched_at = Column(DateTime)
    stock_info_id = Column(BigInteger, ForeignKey("stock_info.id"))
    query = Column(String(255))

    stock_info = relationship("StockInfo", back_populates="search_queries")
    contents = relationship("Content", back_populates="search_query")


class Content(Base):
    """콘텐츠(contents) 테이블 ORM 모델."""

    __tablename__ = "contents"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    published_at = Column(DateTime)
    query_at = Column(DateTime)
    search_query_id = Column(BigInteger, ForeignKey("search_queries.id"))
    description = Column(Text)
    embedding = Column(Text)
    image_url = Column(Text)
    title = Column(Text)
    url = Column(Text, unique=True)
    type = Column(Text)

    search_query = relationship("SearchQuery", back_populates="contents")
    user_logs = relationship("UserLog", back_populates="content")

    # recommendations 테이블을 통해 Recommendation과 다대다 관계 설정
    recommendations = relationship(
        "Recommendation",
        secondary="recommendation_contents",
        back_populates="contents",
        viewonly=True,
    )
    # RecommendationContent 연관 객체에 대한 일대다 관계
    recommendation_links = relationship(
        "RecommendationContent", back_populates="content", cascade="all, delete-orphan"
    )


class Recommendation(Base):
    """추천(recommendations) 테이블 ORM 모델."""

    __tablename__ = "recommendations"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    recommended_at = Column(DateTime)
    user_id = Column(BigInteger, ForeignKey("users.id"))
    model_version = Column(String(255))
    query = Column(String(255))

    user = relationship("User", back_populates="recommendations")
    # recommendation_contents 테이블을 통해 Content와 다대다 관계 설정
    contents = relationship(
        "Content",
        secondary="recommendation_contents",
        back_populates="recommendations",
        viewonly=True,
    )
    # RecommendationContent 연관 객체에 대한 일대다 관계
    content_links = relationship(
        "RecommendationContent",
        back_populates="recommendation",
        cascade="all, delete-orphan",
    )
    user_logs = relationship("UserLog", back_populates="recommendation")


class RecommendationContent(Base):
    """추천-콘텐츠(recommendation_contents) 중간 테이블 ORM 모델.

    추천과 콘텐츠 간의 다대다 연결, 랭킹 정보(rank) 포함.
    """

    __tablename__ = "recommendation_contents"

    content_id = Column(BigInteger, ForeignKey("contents.id"), primary_key=True)
    recommendation_id = Column(
        BigInteger, ForeignKey("recommendations.id"), primary_key=True
    )
    rank = Column(Integer)

    content = relationship("Content", back_populates="recommendation_links")
    recommendation = relationship("Recommendation", back_populates="content_links")


class StockLog(Base):
    """종목 조회 이력(stock_logs) 테이블 ORM 모델."""

    __tablename__ = "stock_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id"), nullable=False)
    viewed_at = Column(DateTime, nullable=False)
    stock_name = Column(String(255), nullable=False)

    user = relationship("User", back_populates="stock_logs")


class UserLog(Base):
    """사용자 로그(user_logs) 테이블 ORM 모델."""

    __tablename__ = "user_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    content_id = Column(BigInteger, ForeignKey("contents.id"), nullable=False)
    recommendation_id = Column(BigInteger, ForeignKey("recommendations.id"))
    user_id = Column(BigInteger, ForeignKey("users.id"), nullable=False)
    event_type = Column(
        SAEnum("CLICK", "VIEW", name="user_log_event_enum"), nullable=False
    )
    timestamp = Column(DateTime(timezone=True))
    ratio = Column(Float)
    time = Column(Integer)

    content = relationship("Content", back_populates="user_logs")
    recommendation = relationship("Recommendation", back_populates="user_logs")
    user = relationship("User", back_populates="user_logs")


def get_db_session() -> Session:
    """SQLAlchemy 세션을 생성하고 반환합니다.

    Returns:
        Session: SQLAlchemy 데이터베이스 세션 객체.
    """
    db = SessionLocal()
    return db


# 테이블 생성 (애플리케이션 시작 시 한 번 호출)
def create_tables():
    """모든 테이블을 데이터베이스에 생성합니다.

    애플리케이션 최초 구동 시 1회 호출 권장.
    """
    Base.metadata.create_all(bind=engine)


# CRUD 함수들
# 각 테이블에서 데이터를 DataFrame으로 반환하는 함수들
# 이 함수들은 데이터베이스 세션을 열고, 쿼리를 실행한 후 DataFrame으로 변환하여 반환합니다.
def get_users() -> pd.DataFrame:
    """users 테이블의 모든 레코드를 DataFrame으로 반환합니다.

    Returns:
        pd.DataFrame: users 테이블 전체 데이터.
    """
    db = get_db_session()
    try:
        query = db.query(User)
        df = pd.read_sql(query.statement, db.bind)
    finally:
        db.close()
    return df


def get_stock_names() -> List[str]:
    """stock_info 테이블의 모든 주식 이름 리스트를 반환합니다.

    Returns:
        List[str]: 주식명 리스트.
    """
    db = get_db_session()
    try:
        query = db.query(StockInfo.name)
        stock_names = [row[0] for row in query.all()]
    finally:
        db.close()
    return stock_names


def get_stock_info() -> pd.DataFrame:
    """stock_info 테이블의 모든 레코드를 DataFrame으로 반환합니다.

    Returns:
        pd.DataFrame: stock_info 테이블 전체 데이터.
    """
    db = get_db_session()
    try:
        query = db.query(StockInfo)
        df = pd.read_sql(query.statement, db.bind)
    finally:
        db.close()
    return df


def get_search_queries() -> pd.DataFrame:
    """search_queries 테이블의 모든 레코드를 DataFrame으로 반환합니다.

    Returns:
        pd.DataFrame: search_queries 테이블 전체 데이터.
    """
    db = get_db_session()
    try:
        query = db.query(SearchQuery)
        df = pd.read_sql(query.statement, db.bind)
    finally:
        db.close()
    return df


def get_contents() -> pd.DataFrame:
    """contents 테이블의 모든 레코드를 DataFrame으로 반환합니다.

    SearchQuery 테이블과 outer join하여 search_query 텍스트도 포함합니다.

    Returns:
        pd.DataFrame: contents + search_query_text 데이터.
    """
    db = get_db_session()
    try:
        query = db.query(
            Content, SearchQuery.query.label("search_query_text")
        ).outerjoin(SearchQuery, Content.search_query_id == SearchQuery.id)

        results = query.all()

        if not results:
            content_columns = [c.name for c in Content.__table__.columns]
            df_columns = content_columns + ["search_query_text"]
            return pd.DataFrame(columns=df_columns)

        contents_data = []
        for row_content, row_search_query_text in results:
            content_dict = {
                c.name: getattr(row_content, c.name) for c in Content.__table__.columns
            }
            content_dict["search_query_text"] = row_search_query_text
            contents_data.append(content_dict)

        df = pd.DataFrame(contents_data)

    finally:
        db.close()
    return df


def get_recommendations() -> pd.DataFrame:
    """recommendations 테이블의 모든 레코드를 DataFrame으로 반환합니다.

    Returns:
        pd.DataFrame: recommendations 테이블 전체 데이터.
    """
    db = get_db_session()
    try:
        query = db.query(Recommendation)
        df = pd.read_sql(query.statement, db.bind)
    finally:
        db.close()
    return df


def get_recommendation_contents() -> pd.DataFrame:
    """recommendation_contents 테이블의 모든 레코드를 DataFrame으로 반환합니다.

    Returns:
        pd.DataFrame: recommendation_contents 테이블 전체 데이터.
    """
    db = get_db_session()
    try:
        query = db.query(RecommendationContent)
        df = pd.read_sql(query.statement, db.bind)
    finally:
        db.close()
    return df


def get_user_logs() -> pd.DataFrame:
    """user_logs 테이블의 모든 레코드를 DataFrame으로 반환합니다.

    Returns:
        pd.DataFrame: user_logs 테이블 전체 데이터.
    """
    db = get_db_session()
    try:
        query = db.query(UserLog)
        df = pd.read_sql(query.statement, db.bind)
    finally:
        db.close()
    return df


def get_stock_logs() -> pd.DataFrame:
    """stock_logs 테이블의 모든 레코드를 DataFrame으로 반환합니다.

    Returns:
        pd.DataFrame: stock_logs 테이블 전체 데이터.
    """
    db = get_db_session()
    try:
        query = db.query(StockLog)
        df = pd.read_sql(query.statement, db.bind)
    finally:
        db.close()
    return df
