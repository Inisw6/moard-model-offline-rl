import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# SQLite 데이터베이스 설정
DB_PATH: str = "./data/personas.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@dataclass
class SimulationPersona:
    """시뮬레이션용 페르소나 정보"""

    persona_id: int
    mbti: str
    investment_level: int  # 1: 초보, 2: 중급, 3: 고급


class Persona(Base):
    """페르소나 테이블 정의"""

    __tablename__ = "personas"

    persona_id = Column(Integer, primary_key=True)
    mbti = Column(String(4), nullable=False)
    investment_level = Column(Integer, nullable=False)  # 1: 초보, 2: 중급, 3: 고급


class PersonaDB:
    """SQLAlchemy 기반 페르소나 데이터베이스 관리자"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
        self._populate_default_personas()

    def _init_database(self):
        """데이터베이스 테이블 초기화"""
        Base.metadata.create_all(bind=engine)

    def _populate_default_personas(self):
        """기본 페르소나들을 데이터베이스에 추가"""
        db = get_db_session()
        try:
            # 이미 데이터가 있는지 확인
            count = db.query(Persona).count()
            if count > 0:
                return  # 이미 데이터가 있으면 스킵

            import json

            with open("data/default_personas.json", "r", encoding="utf-8") as f:
                default_personas = json.load(f)

            for persona in default_personas:
                db_persona = Persona(
                    persona_id=persona["persona_id"],
                    mbti=persona["mbti"],
                    investment_level=persona["investment_level"],
                )
                db.add(db_persona)
            db.commit()
        finally:
            db.close()

    def get_persona_by_id(self, persona_id: int) -> Optional[SimulationPersona]:
        """ID로 페르소나 조회"""
        db = get_db_session()
        try:
            persona = db.query(Persona).filter(Persona.persona_id == persona_id).first()
            if persona:
                return SimulationPersona(
                    persona_id=persona.persona_id,
                    mbti=persona.mbti,
                    investment_level=persona.investment_level,
                )
            return None
        finally:
            db.close()

    def get_all_personas(self) -> List[SimulationPersona]:
        """모든 페르소나 조회"""
        db = get_db_session()
        try:
            personas = db.query(Persona).all()
            return [
                SimulationPersona(
                    persona_id=p.persona_id,
                    mbti=p.mbti,
                    investment_level=p.investment_level,
                )
                for p in personas
            ]
        finally:
            db.close()

    def get_personas_by_level(self, investment_level: int) -> List[SimulationPersona]:
        """투자 레벨별 페르소나 조회"""
        db = get_db_session()
        try:
            personas = (
                db.query(Persona)
                .filter(Persona.investment_level == investment_level)
                .all()
            )
            return [
                SimulationPersona(
                    persona_id=p.persona_id,
                    mbti=p.mbti,
                    investment_level=p.investment_level,
                )
                for p in personas
            ]
        finally:
            db.close()

    def get_personas_by_mbti(self, mbti: str) -> List[SimulationPersona]:
        """MBTI별 페르소나 조회"""
        db = get_db_session()
        try:
            personas = db.query(Persona).filter(Persona.mbti == mbti).all()
            return [
                SimulationPersona(
                    persona_id=p.persona_id,
                    mbti=p.mbti,
                    investment_level=p.investment_level,
                )
                for p in personas
            ]
        finally:
            db.close()

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


def get_db_session():
    """
    SQLAlchemy 세션을 생성하고 반환합니다.
    호출자는 세션 사용 후 db.close()를 호출해야 합니다.
    """
    db = SessionLocal()
    return db


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
