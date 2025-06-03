import sqlite3
import pandas as pd

DB_PATH: str = "sample_recsys.db"  # 데이터베이스 파일 경로


def get_db_connection() -> sqlite3.Connection:
    """
    SQLite 데이터베이스에 연결하고 Connection 객체를 반환합니다.

    Returns:
        sqlite3.Connection: 데이터베이스 연결 객체
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 컬럼명으로 접근 가능하도록 설정
    return conn


def get_users() -> pd.DataFrame:
    """
    users 테이블에서 모든 데이터를 DataFrame으로 반환합니다.

    Returns:
        pd.DataFrame: users 테이블 전체 데이터
    """
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM users", conn)
    conn.close()
    return df


def get_stock_info() -> pd.DataFrame:
    """
    stock_info 테이블에서 모든 데이터를 DataFrame으로 반환합니다.

    Returns:
        pd.DataFrame: stock_info 테이블 전체 데이터
    """
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM stock_info", conn)
    conn.close()
    return df


def get_search_queries() -> pd.DataFrame:
    """
    search_queries 테이블에서 모든 데이터를 DataFrame으로 반환합니다.

    Returns:
        pd.DataFrame: search_queries 테이블 전체 데이터
    """
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM search_queries", conn)
    conn.close()
    return df


def get_contents() -> pd.DataFrame:
    """
    contents 테이블에서 모든 데이터를 DataFrame으로 반환합니다.

    Returns:
        pd.DataFrame: contents 테이블 전체 데이터
    """
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM contents", conn)
    conn.close()
    return df


def get_recommendations() -> pd.DataFrame:
    """
    recommendations 테이블에서 모든 데이터를 DataFrame으로 반환합니다.

    Returns:
        pd.DataFrame: recommendations 테이블 전체 데이터
    """
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM recommendations", conn)
    conn.close()
    return df


def get_recommendation_contents() -> pd.DataFrame:
    """
    recommendation_contents 테이블에서 모든 데이터를 DataFrame으로 반환합니다.

    Returns:
        pd.DataFrame: recommendation_contents 테이블 전체 데이터
    """
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM recommendation_contents", conn)
    conn.close()
    return df


def get_user_logs() -> pd.DataFrame:
    """
    user_logs 테이블에서 모든 데이터를 DataFrame으로 반환합니다.

    Returns:
        pd.DataFrame: user_logs 테이블 전체 데이터
    """
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM user_logs", conn)
    conn.close()
    return df


def get_stock_logs() -> pd.DataFrame:
    """
    stock_logs 테이블에서 모든 데이터를 DataFrame으로 반환합니다.

    Returns:
        pd.DataFrame: stock_logs 테이블 전체 데이터
    """
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM stock_logs", conn)
    conn.close()
    return df
