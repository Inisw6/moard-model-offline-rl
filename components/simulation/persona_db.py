import os
import random
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class SimulationPersona:
    """시뮬레이션용 페르소나 정보"""

    persona_id: int
    mbti: str
    investment_level: int  # 1: 초보, 2: 중급, 3: 고급


class PersonaDB:
    """SQLite 기반 페르소나 데이터베이스 관리자"""

    def __init__(self, db_path: str = "data/personas.db"):
        self.db_path = db_path
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
        self._populate_default_personas()

    def _init_database(self):
        """데이터베이스 테이블 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS personas (
                    persona_id INTEGER PRIMARY KEY,
                    mbti TEXT NOT NULL,
                    investment_level INTEGER NOT NULL
                )
            """
            )
            conn.commit()

    def _populate_default_personas(self):
        """기본 페르소나들을 데이터베이스에 추가"""
        # 이미 데이터가 있는지 확인
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM personas")
            count = cursor.fetchone()[0]

            if count > 0:
                return  # 이미 데이터가 있으면 스킵

        import json

        with open("data/default_personas.json", "r", encoding="utf-8") as f:
            default_personas = json.load(f)

        for persona in default_personas:
            # 각 페르소나를 SimulationPersona 객체로 변환
            persona_ = SimulationPersona(
                persona_id=persona["persona_id"],
                mbti=persona["mbti"],
                investment_level=persona["investment_level"],
            )
            self._insert_persona(persona_)

        with sqlite3.connect(self.db_path) as conn:
            for persona in default_personas:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO personas 
                    (persona_id, mbti, investment_level)
                    VALUES (?, ?, ?)
                """,
                    (persona.persona_id, persona.mbti, persona.investment_level),
                )
            conn.commit()

    def get_persona_by_id(self, persona_id: int) -> Optional[SimulationPersona]:
        """ID로 페르소나 조회"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM personas WHERE persona_id = ?", (persona_id,)
            )
            row = cursor.fetchone()

            if row:
                return SimulationPersona(**dict(row))
            return None

    def get_all_personas(self) -> List[SimulationPersona]:
        """모든 페르소나 조회"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM personas")
            rows = cursor.fetchall()

            return [SimulationPersona(**dict(row)) for row in rows]

    def get_personas_by_level(self, investment_level: int) -> List[SimulationPersona]:
        """투자 레벨별 페르소나 조회"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM personas WHERE investment_level = ?", (investment_level,)
            )
            rows = cursor.fetchall()

            return [SimulationPersona(**dict(row)) for row in rows]

    def get_personas_by_mbti(self, mbti: str) -> List[SimulationPersona]:
        """MBTI별 페르소나 조회"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM personas WHERE mbti = ?", (mbti,))
            rows = cursor.fetchall()

            return [SimulationPersona(**dict(row)) for row in rows]

    def get_random_persona(self) -> SimulationPersona:
        """랜덤 페르소나 선택"""
        personas = self.get_all_personas()
        return random.choice(personas)

    def create_user_db_func(self, persona_id: int) -> Callable[[int], Dict[str, Any]]:
        """특정 페르소나에 대한 user_db_func 생성"""
        persona = self.get_persona_by_id(persona_id)
        if not persona:
            raise ValueError(f"Persona {persona_id} not found")

        def user_db_func(user_id: int) -> Dict[str, Any]:
            return {
                "id": user_id,
                "uuid": f"persona_{persona.persona_id}",
                "persona_id": persona.persona_id,
                "mbti": persona.mbti,
                "investment_level": persona.investment_level,
            }

        return user_db_func


# 전역 인스턴스
_persona_db = None


def get_persona_db() -> PersonaDB:
    """페르소나 DB 인스턴스 반환"""
    global _persona_db
    if _persona_db is None:
        _persona_db = PersonaDB()
    return _persona_db


def get_random_persona_func() -> Callable[[int], Dict[str, Any]]:
    """랜덤 페르소나용 user_db_func 반환"""
    db = get_persona_db()
    persona = db.get_random_persona()
    return db.create_user_db_func(persona.persona_id)


def get_persona_manager():
    """기존 호환성을 위한 별칭"""
    return get_persona_db()
