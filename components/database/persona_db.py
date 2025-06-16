import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# SQLite 데이터베이스 설정
DB_PATH: str = "./data/personas.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# 기본 ORM 베이스 및 세션
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db_session() -> Session:
    """새로운 SQLAlchemy 세션을 생성하여 반환합니다.

    Returns:
        Session: 데이터베이스 세션 객체
    """
    return SessionLocal()


@dataclass
class SimulationPersona:
    """시뮬레이션용 페르소나 정보 데이터 클래스.

    Attributes:
        persona_id (int): 페르소나 고유번호 (기본키)
        mbti (str): MBTI 유형 (예: "INTJ")
        investment_level (int): 투자 성향 (1=초보, 2=중급, 3=고급)
    """

    persona_id: int
    mbti: str
    investment_level: int


class Persona(Base):
    """페르소나(personas) 테이블 ORM 모델."""

    __tablename__ = "personas"

    persona_id = Column(Integer, primary_key=True)
    mbti = Column(String(4), nullable=False)
    investment_level = Column(Integer, nullable=False)


class PersonaDB:
    """SQLAlchemy 기반 페르소나 데이터베이스 관리자."""

    def __init__(self, db_path: str = DB_PATH):
        """PersonaDB 인스턴스를 초기화하고 기본 데이터를 삽입합니다.

        Args:
            db_path (str, optional): 데이터베이스 파일 경로. Defaults to DB_PATH.
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
        self._populate_default_personas()

    def _init_database(self) -> None:
        """테이블이 없으면 SQLAlchemy로 테이블을 생성합니다."""
        Base.metadata.create_all(bind=engine)

    def _populate_default_personas(self) -> None:
        """기본 페르소나 데이터를 JSON 파일에서 로드하여 삽입합니다."""
        db = get_db_session()
        try:
            if db.query(Persona).count() > 0:
                return
            import json

            with open("data/default_personas.json", "r", encoding="utf-8") as f:
                default_list = json.load(f)

            for entry in default_list:
                db.add(
                    Persona(
                        persona_id=entry["persona_id"],
                        mbti=entry["mbti"],
                        investment_level=entry["investment_level"],
                    )
                )
            db.commit()
        finally:
            db.close()

    def get_persona_by_id(self, persona_id: int) -> Optional[SimulationPersona]:
        """ID로 페르소나를 조회합니다.

        Args:
            persona_id (int): 조회할 페르소나 ID.

        Returns:
            Optional[SimulationPersona]: 해당 페르소나 정보 또는 None
        """
        db = get_db_session()
        try:
            p = db.query(Persona).filter(Persona.persona_id == persona_id).first()
            if p:
                return SimulationPersona(
                    persona_id=p.persona_id,
                    mbti=p.mbti,
                    investment_level=p.investment_level,
                )
            return None
        finally:
            db.close()

    def get_all_personas(self) -> List[SimulationPersona]:
        """전체 페르소나 목록을 조회합니다.

        Returns:
            List[SimulationPersona]: 등록된 모든 페르소나
        """
        db = get_db_session()
        try:
            return [
                SimulationPersona(
                    persona_id=p.persona_id,
                    mbti=p.mbti,
                    investment_level=p.investment_level,
                )
                for p in db.query(Persona).all()
            ]
        finally:
            db.close()

    def get_personas_by_level(self, level: int) -> List[SimulationPersona]:
        """투자 성향 레벨로 페르소나를 필터링합니다.

        Args:
            level (int): 투자 성향 (1, 2, 또는 3)

        Returns:
            List[SimulationPersona]: 해당 레벨의 페르소나 리스트
        """
        db = get_db_session()
        try:
            return [
                SimulationPersona(
                    persona_id=p.persona_id,
                    mbti=p.mbti,
                    investment_level=p.investment_level,
                )
                for p in db.query(Persona)
                .filter(Persona.investment_level == level)
                .all()
            ]
        finally:
            db.close()

    def get_personas_by_mbti(self, mbti: str) -> List[SimulationPersona]:
        """MBTI 유형으로 페르소나를 필터링합니다.

        Args:
            mbti (str): 조회할 MBTI 문자열

        Returns:
            List[SimulationPersona]: 해당 MBTI의 페르소나 리스트
        """
        db = get_db_session()
        try:
            return [
                SimulationPersona(
                    persona_id=p.persona_id,
                    mbti=p.mbti,
                    investment_level=p.investment_level,
                )
                for p in db.query(Persona).filter(Persona.mbti == mbti).all()
            ]
        finally:
            db.close()

    def get_random_persona(self) -> SimulationPersona:
        """랜덤으로 하나의 페르소나를 반환합니다."""
        candidates = self.get_all_personas()
        return random.choice(candidates)

    def create_user_db_func(self, persona_id: int) -> Callable[[int], Dict[str, Any]]:
        """주어진 페르소나 ID로 user_db_func 콜백을 생성합니다.

        Args:
            persona_id (int): 사용할 페르소나 ID

        Returns:
            Callable[[int], Dict[str, Any]]: user_id를 받아 페르소나 기반 dict 반환

        Raises:
            ValueError: 페르소나가 존재하지 않을 때
        """
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


# 전역 PersonaDB 인스턴스
_persona_db: Optional[PersonaDB] = None


def get_persona_db() -> PersonaDB:
    """글로벌 PersonaDB 싱글톤 인스턴스를 반환합니다."""
    global _persona_db
    if _persona_db is None:
        _persona_db = PersonaDB()
    return _persona_db


def get_random_persona_func() -> Callable[[int], Dict[str, Any]]:
    """랜덤 페르소나 기반 user_db_func를 반환합니다."""
    db = get_persona_db()
    return db.create_user_db_func(db.get_random_persona().persona_id)


def get_persona_manager() -> PersonaDB:
    """기존 호환성을 위한 PersonaDB 별칭입니다."""
    return get_persona_db()
