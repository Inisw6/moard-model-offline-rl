import sqlite3
import pandas as pd

DB_PATH = "sample_recsys.db"  # 데이터베이스 파일 경로


def get_db_connection():
    """SQLite 데이터베이스에 연결합니다."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 컬럼명으로 접근 가능하도록 설정
    return conn


def get_users():
    """users 테이블에서 모든 데이터를 가져옵니다."""
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM users", conn)
    conn.close()
    return df


def get_stock_info():
    """stock_info 테이블에서 모든 데이터를 가져옵니다."""
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM stock_info", conn)
    conn.close()
    return df


def get_search_queries():
    """search_queries 테이블에서 모든 데이터를 가져옵니다."""
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM search_queries", conn)
    conn.close()
    return df


def get_contents():
    """contents 테이블에서 모든 데이터를 가져옵니다."""
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM contents", conn)
    conn.close()
    return df


def get_recommendations():
    """recommendations 테이블에서 모든 데이터를 가져옵니다."""
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM recommendations", conn)
    conn.close()
    return df


def get_recommendation_contents():
    """recommendation_contents 테이블에서 모든 데이터를 가져옵니다."""
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM recommendation_contents", conn)
    conn.close()
    return df


def get_user_logs():
    """user_logs 테이블에서 모든 데이터를 가져옵니다."""
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM user_logs", conn)
    conn.close()
    return df


def get_stock_logs():
    """stock_logs 테이블에서 모든 데이터를 가져옵니다."""
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM stock_logs", conn)
    conn.close()
    return df 