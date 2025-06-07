import sqlite3
import random
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable
import os


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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS personas (
                    persona_id INTEGER PRIMARY KEY,
                    mbti TEXT NOT NULL,
                    investment_level INTEGER NOT NULL
                )
            """)
            conn.commit()
    
    def _populate_default_personas(self):
        """기본 페르소나들을 데이터베이스에 추가"""
        # 이미 데이터가 있는지 확인
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM personas")
            count = cursor.fetchone()[0]
            
            if count > 0:
                return  # 이미 데이터가 있으면 스킵
        
        default_personas = [
    # ─── INFP (13명) ───
    SimulationPersona(persona_id=1,  mbti="INFP", investment_level=1),
    SimulationPersona(persona_id=2,  mbti="INFP", investment_level=1),
    SimulationPersona(persona_id=3,  mbti="INFP", investment_level=1),
    SimulationPersona(persona_id=4,  mbti="INFP", investment_level=2),
    SimulationPersona(persona_id=5,  mbti="INFP", investment_level=2),
    SimulationPersona(persona_id=6,  mbti="INFP", investment_level=2),
    SimulationPersona(persona_id=7,  mbti="INFP", investment_level=3),
    SimulationPersona(persona_id=8,  mbti="INFP", investment_level=3),
    SimulationPersona(persona_id=9,  mbti="INFP", investment_level=3),
    SimulationPersona(persona_id=10, mbti="INFP", investment_level=4),
    SimulationPersona(persona_id=11, mbti="INFP", investment_level=4),
    SimulationPersona(persona_id=12, mbti="INFP", investment_level=5),
    SimulationPersona(persona_id=13, mbti="INFP", investment_level=5),

    # ─── ENFP (13명) ───
    SimulationPersona(persona_id=14, mbti="ENFP", investment_level=1),
    SimulationPersona(persona_id=15, mbti="ENFP", investment_level=1),
    SimulationPersona(persona_id=16, mbti="ENFP", investment_level=1),
    SimulationPersona(persona_id=17, mbti="ENFP", investment_level=2),
    SimulationPersona(persona_id=18, mbti="ENFP", investment_level=2),
    SimulationPersona(persona_id=19, mbti="ENFP", investment_level=2),
    SimulationPersona(persona_id=20, mbti="ENFP", investment_level=3),
    SimulationPersona(persona_id=21, mbti="ENFP", investment_level=3),
    SimulationPersona(persona_id=22, mbti="ENFP", investment_level=3),
    SimulationPersona(persona_id=23, mbti="ENFP", investment_level=4),
    SimulationPersona(persona_id=24, mbti="ENFP", investment_level=4),
    SimulationPersona(persona_id=25, mbti="ENFP", investment_level=5),
    SimulationPersona(persona_id=26, mbti="ENFP", investment_level=5),

    # ─── ESFJ (8명) ───
    SimulationPersona(persona_id=27, mbti="ESFJ", investment_level=1),
    SimulationPersona(persona_id=28, mbti="ESFJ", investment_level=1),
    SimulationPersona(persona_id=29, mbti="ESFJ", investment_level=2),
    SimulationPersona(persona_id=30, mbti="ESFJ", investment_level=2),
    SimulationPersona(persona_id=31, mbti="ESFJ", investment_level=3),
    SimulationPersona(persona_id=32, mbti="ESFJ", investment_level=3),
    SimulationPersona(persona_id=33, mbti="ESFJ", investment_level=4),
    SimulationPersona(persona_id=34, mbti="ESFJ", investment_level=5),

    # ─── ISFJ (8명) ───
    SimulationPersona(persona_id=35, mbti="ISFJ", investment_level=1),
    SimulationPersona(persona_id=36, mbti="ISFJ", investment_level=1),
    SimulationPersona(persona_id=37, mbti="ISFJ", investment_level=2),
    SimulationPersona(persona_id=38, mbti="ISFJ", investment_level=2),
    SimulationPersona(persona_id=39, mbti="ISFJ", investment_level=3),
    SimulationPersona(persona_id=40, mbti="ISFJ", investment_level=3),
    SimulationPersona(persona_id=41, mbti="ISFJ", investment_level=4),
    SimulationPersona(persona_id=42, mbti="ISFJ", investment_level=5),

    # ─── ISFP (7명) ───
    SimulationPersona(persona_id=43, mbti="ISFP", investment_level=1),
    SimulationPersona(persona_id=44, mbti="ISFP", investment_level=1),
    SimulationPersona(persona_id=45, mbti="ISFP", investment_level=2),
    SimulationPersona(persona_id=46, mbti="ISFP", investment_level=2),
    SimulationPersona(persona_id=47, mbti="ISFP", investment_level=3),
    SimulationPersona(persona_id=48, mbti="ISFP", investment_level=4),
    SimulationPersona(persona_id=49, mbti="ISFP", investment_level=5),

    # ─── ESFP (6명) ───
    SimulationPersona(persona_id=50, mbti="ESFP", investment_level=1),
    SimulationPersona(persona_id=51, mbti="ESFP", investment_level=1),
    SimulationPersona(persona_id=52, mbti="ESFP", investment_level=2),
    SimulationPersona(persona_id=53, mbti="ESFP", investment_level=3),
    SimulationPersona(persona_id=54, mbti="ESFP", investment_level=4),
    SimulationPersona(persona_id=55, mbti="ESFP", investment_level=5),

    # ─── INTP (6명) ───
    SimulationPersona(persona_id=56, mbti="INTP", investment_level=1),
    SimulationPersona(persona_id=57, mbti="INTP", investment_level=1),
    SimulationPersona(persona_id=58, mbti="INTP", investment_level=2),
    SimulationPersona(persona_id=59, mbti="INTP", investment_level=3),
    SimulationPersona(persona_id=60, mbti="INTP", investment_level=4),
    SimulationPersona(persona_id=61, mbti="INTP", investment_level=5),

    # ─── INFJ (6명) ───
    SimulationPersona(persona_id=62, mbti="INFJ", investment_level=1),
    SimulationPersona(persona_id=63, mbti="INFJ", investment_level=1),
    SimulationPersona(persona_id=64, mbti="INFJ", investment_level=2),
    SimulationPersona(persona_id=65, mbti="INFJ", investment_level=3),
    SimulationPersona(persona_id=66, mbti="INFJ", investment_level=4),
    SimulationPersona(persona_id=67, mbti="INFJ", investment_level=5),

    # ─── ENFJ (6명) ───
    SimulationPersona(persona_id=68, mbti="ENFJ", investment_level=1),
    SimulationPersona(persona_id=69, mbti="ENFJ", investment_level=1),
    SimulationPersona(persona_id=70, mbti="ENFJ", investment_level=2),
    SimulationPersona(persona_id=71, mbti="ENFJ", investment_level=3),
    SimulationPersona(persona_id=72, mbti="ENFJ", investment_level=4),
    SimulationPersona(persona_id=73, mbti="ENFJ", investment_level=5),

    # ─── ENTP (5명) ───
    SimulationPersona(persona_id=74, mbti="ENTP", investment_level=1),
    SimulationPersona(persona_id=75, mbti="ENTP", investment_level=2),
    SimulationPersona(persona_id=76, mbti="ENTP", investment_level=3),
    SimulationPersona(persona_id=77, mbti="ENTP", investment_level=4),
    SimulationPersona(persona_id=78, mbti="ENTP", investment_level=5),

    # ─── ESTJ (5명) ───
    SimulationPersona(persona_id=79, mbti="ESTJ", investment_level=1),
    SimulationPersona(persona_id=80, mbti="ESTJ", investment_level=2),
    SimulationPersona(persona_id=81, mbti="ESTJ", investment_level=3),
    SimulationPersona(persona_id=82, mbti="ESTJ", investment_level=4),
    SimulationPersona(persona_id=83, mbti="ESTJ", investment_level=5),

    # ─── ISTJ (4명) ───
    SimulationPersona(persona_id=84, mbti="ISTJ", investment_level=1),
    SimulationPersona(persona_id=85, mbti="ISTJ", investment_level=2),
    SimulationPersona(persona_id=86, mbti="ISTJ", investment_level=4),
    SimulationPersona(persona_id=87, mbti="ISTJ", investment_level=5),

    # ─── INTJ (4명) ───
    SimulationPersona(persona_id=88, mbti="INTJ", investment_level=1),
    SimulationPersona(persona_id=89, mbti="INTJ", investment_level=2),
    SimulationPersona(persona_id=90, mbti="INTJ", investment_level=4),
    SimulationPersona(persona_id=91, mbti="INTJ", investment_level=5),

    # ─── ISTP (3명) ───
    SimulationPersona(persona_id=92, mbti="ISTP", investment_level=1),
    SimulationPersona(persona_id=93, mbti="ISTP", investment_level=3),
    SimulationPersona(persona_id=94, mbti="ISTP", investment_level=5),

    # ─── ESTP (3명) ───
    SimulationPersona(persona_id=95, mbti="ESTP", investment_level=1),
    SimulationPersona(persona_id=96, mbti="ESTP", investment_level=3),
    SimulationPersona(persona_id=97, mbti="ESTP", investment_level=5),

    # ─── ENTJ (3명) ───
    SimulationPersona(persona_id=98,  mbti="ENTJ", investment_level=1),
    SimulationPersona(persona_id=99,  mbti="ENTJ", investment_level=3),
    SimulationPersona(persona_id=100, mbti="ENTJ", investment_level=5),
]

        
        
        with sqlite3.connect(self.db_path) as conn:
            for persona in default_personas:
                conn.execute("""
                    INSERT OR REPLACE INTO personas 
                    (persona_id, mbti, investment_level)
                    VALUES (?, ?, ?)
                """, (
                    persona.persona_id,
                    persona.mbti,
                    persona.investment_level
                ))
            conn.commit()
    
    def get_persona_by_id(self, persona_id: int) -> Optional[SimulationPersona]:
        """ID로 페르소나 조회"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM personas WHERE persona_id = ?", 
                (persona_id,)
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
                "SELECT * FROM personas WHERE investment_level = ?", 
                (investment_level,)
            )
            rows = cursor.fetchall()
            
            return [SimulationPersona(**dict(row)) for row in rows]
    
    def get_personas_by_mbti(self, mbti: str) -> List[SimulationPersona]:
        """MBTI별 페르소나 조회"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM personas WHERE mbti = ?", 
                (mbti,)
            )
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
                "investment_level": persona.investment_level
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