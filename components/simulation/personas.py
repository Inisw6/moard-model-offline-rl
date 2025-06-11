from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict

# 1. Enum 정의


class Horizon(str, Enum):
    """투자 기간(Enum).

    Values:
        SHORT: 단기 투자
        MEDIUM: 중기 투자
        LONG: 장기 투자
    """

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class AnalysisStyle(str, Enum):
    """투자 분석 스타일(Enum).

    Values:
        FUNDAMENTAL: 펀더멘털 분석
        TECHNICAL: 기술적 분석
        MIXED: 혼합 분석
    """

    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    MIXED = "mixed"


# 2. MBTI별 기본값

MBTI_DEFAULTS: Dict[str, Dict] = {
    # 각 MBTI-타입별 투자 페르소나 (0~1 스케일)
    # Horizon  ➜  SHORT / MEDIUM / LONG
    # AnalysisStyle ➜ TECHNICAL / FUNDAMENTAL / MIXED
    "ISTJ": {
        "preferences": {"youtube": 0.35, "blog": 0.80, "news": 0.65},
        "risk_tolerance": 0.25,
        "decision_speed": 0.40,
        "social_influence": 0.25,
        "investment_horizon": Horizon.LONG,
        "analysis_preference": AnalysisStyle.FUNDAMENTAL,
        "volatility_tolerance": 0.30,
        "dividend_preference": 0.70,
        "expert_reliance": 0.55,
    },
    "ISFJ": {
        "preferences": {"youtube": 0.30, "blog": 0.70, "news": 0.60},
        "risk_tolerance": 0.30,
        "decision_speed": 0.35,
        "social_influence": 0.40,
        "investment_horizon": Horizon.LONG,
        "analysis_preference": AnalysisStyle.FUNDAMENTAL,
        "volatility_tolerance": 0.30,
        "dividend_preference": 0.75,
        "expert_reliance": 0.70,
    },
    "INFJ": {
        "preferences": {"youtube": 0.45, "blog": 0.70, "news": 0.60},
        "risk_tolerance": 0.40,
        "decision_speed": 0.45,
        "social_influence": 0.35,
        "investment_horizon": Horizon.LONG,
        "analysis_preference": AnalysisStyle.FUNDAMENTAL,
        "volatility_tolerance": 0.40,
        "dividend_preference": 0.50,
        "expert_reliance": 0.50,
    },
    "INTJ": {
        "preferences": {"youtube": 0.50, "blog": 0.85, "news": 0.75},
        "risk_tolerance": 0.60,
        "decision_speed": 0.65,
        "social_influence": 0.20,
        "investment_horizon": Horizon.LONG,
        "analysis_preference": AnalysisStyle.MIXED,
        "volatility_tolerance": 0.55,
        "dividend_preference": 0.25,
        "expert_reliance": 0.35,
    },
    "ISTP": {
        "preferences": {"youtube": 0.50, "blog": 0.75, "news": 0.80},
        "risk_tolerance": 0.65,
        "decision_speed": 0.70,
        "social_influence": 0.30,
        "investment_horizon": Horizon.SHORT,
        "analysis_preference": AnalysisStyle.TECHNICAL,
        "volatility_tolerance": 0.70,
        "dividend_preference": 0.10,
        "expert_reliance": 0.30,
    },
    "ISFP": {
        "preferences": {"youtube": 0.45, "blog": 0.65, "news": 0.55},
        "risk_tolerance": 0.35,
        "decision_speed": 0.40,
        "social_influence": 0.45,
        "investment_horizon": Horizon.MEDIUM,
        "analysis_preference": AnalysisStyle.FUNDAMENTAL,
        "volatility_tolerance": 0.35,
        "dividend_preference": 0.60,
        "expert_reliance": 0.60,
    },
    "INFP": {
        "preferences": {"youtube": 0.45, "blog": 0.70, "news": 0.55},
        "risk_tolerance": 0.40,
        "decision_speed": 0.45,
        "social_influence": 0.50,
        "investment_horizon": Horizon.LONG,
        "analysis_preference": AnalysisStyle.FUNDAMENTAL,
        "volatility_tolerance": 0.40,
        "dividend_preference": 0.50,
        "expert_reliance": 0.55,
    },
    "INTP": {
        "preferences": {"youtube": 0.55, "blog": 0.80, "news": 0.70},
        "risk_tolerance": 0.50,
        "decision_speed": 0.55,
        "social_influence": 0.20,
        "investment_horizon": Horizon.MEDIUM,
        "analysis_preference": AnalysisStyle.MIXED,
        "volatility_tolerance": 0.50,
        "dividend_preference": 0.30,
        "expert_reliance": 0.35,
    },
    "ESTP": {
        "preferences": {"youtube": 0.70, "blog": 0.55, "news": 0.75},
        "risk_tolerance": 0.80,
        "decision_speed": 0.85,
        "social_influence": 0.70,
        "investment_horizon": Horizon.SHORT,
        "analysis_preference": AnalysisStyle.TECHNICAL,
        "volatility_tolerance": 0.85,
        "dividend_preference": 0.10,
        "expert_reliance": 0.30,
    },
    "ESFP": {
        "preferences": {"youtube": 0.70, "blog": 0.60, "news": 0.65},
        "risk_tolerance": 0.65,
        "decision_speed": 0.70,
        "social_influence": 0.75,
        "investment_horizon": Horizon.MEDIUM,
        "analysis_preference": AnalysisStyle.MIXED,
        "volatility_tolerance": 0.65,
        "dividend_preference": 0.30,
        "expert_reliance": 0.50,
    },
    "ENFP": {
        "preferences": {"youtube": 0.65, "blog": 0.60, "news": 0.70},
        "risk_tolerance": 0.70,
        "decision_speed": 0.65,
        "social_influence": 0.75,
        "investment_horizon": Horizon.LONG,
        "analysis_preference": AnalysisStyle.MIXED,
        "volatility_tolerance": 0.70,
        "dividend_preference": 0.25,
        "expert_reliance": 0.40,
    },
    "ENTP": {
        "preferences": {"youtube": 0.85, "blog": 0.60, "news": 0.70},
        "risk_tolerance": 0.75,
        "decision_speed": 0.85,
        "social_influence": 0.65,
        "investment_horizon": Horizon.SHORT,
        "analysis_preference": AnalysisStyle.MIXED,
        "volatility_tolerance": 0.80,
        "dividend_preference": 0.20,
        "expert_reliance": 0.50,
    },
    "ESTJ": {
        "preferences": {"youtube": 0.40, "blog": 0.65, "news": 0.75},
        "risk_tolerance": 0.45,
        "decision_speed": 0.60,
        "social_influence": 0.55,
        "investment_horizon": Horizon.SHORT,
        "analysis_preference": AnalysisStyle.FUNDAMENTAL,
        "volatility_tolerance": 0.45,
        "dividend_preference": 0.60,
        "expert_reliance": 0.60,
    },
    "ESFJ": {
        "preferences": {"youtube": 0.45, "blog": 0.60, "news": 0.65},
        "risk_tolerance": 0.35,
        "decision_speed": 0.50,
        "social_influence": 0.70,
        "investment_horizon": Horizon.LONG,
        "analysis_preference": AnalysisStyle.FUNDAMENTAL,
        "volatility_tolerance": 0.35,
        "dividend_preference": 0.65,
        "expert_reliance": 0.70,
    },
    "ENFJ": {
        "preferences": {"youtube": 0.55, "blog": 0.65, "news": 0.70},
        "risk_tolerance": 0.50,
        "decision_speed": 0.55,
        "social_influence": 0.75,
        "investment_horizon": Horizon.LONG,
        "analysis_preference": AnalysisStyle.FUNDAMENTAL,
        "volatility_tolerance": 0.50,
        "dividend_preference": 0.45,
        "expert_reliance": 0.65,
    },
    "ENTJ": {
        "preferences": {"youtube": 0.60, "blog": 0.75, "news": 0.80},
        "risk_tolerance": 0.80,
        "decision_speed": 0.80,
        "social_influence": 0.45,
        "investment_horizon": Horizon.LONG,
        "analysis_preference": AnalysisStyle.MIXED,
        "volatility_tolerance": 0.75,
        "dividend_preference": 0.20,
        "expert_reliance": 0.55,
    },
}


# 안전장치 (존재하지 않는 MBTI 타입용)
MBTI_FALLBACK = {
    "preferences": {"youtube": 0.5, "blog": 0.5, "news": 0.5},
    "risk_tolerance": 0.5,
    "decision_speed": 0.5,
    "social_influence": 0.5,
    "investment_horizon": Horizon.MEDIUM,
    "analysis_preference": AnalysisStyle.MIXED,
    "volatility_tolerance": 0.5,
    "dividend_preference": 0.5,
    "expert_reliance": 0.5,
}


# 3. PersonaConfig 데이터 클래스


@dataclass
class PersonaConfig:
    """투자자 페르소나 구성 정보.

    Attributes:
        mbti (str): MBTI 유형.
        investment_level (int): 투자 레벨(1~5).
        user_id (int): 사용자 고유 ID.
        description (str): 설명.
        preferences (Dict[str, float]): 채널별 선호도.
        risk_tolerance (float): 위험 허용도 (0~1).
        decision_speed (float): 의사결정 속도 (0~1).
        social_influence (float): 사회적 영향력 민감도 (0~1).
        investment_horizon (Horizon): 투자 기간.
        analysis_preference (AnalysisStyle): 분석 성향.
        volatility_tolerance (float): 변동성 허용도 (0~1).
        dividend_preference (float): 배당 선호도 (0~1).
        expert_reliance (float): 전문가 의존도 (0~1).
        created_at (str): 생성 시각(UTC ISO8601).
    """

    mbti: str
    investment_level: int  # 1–5
    user_id: int  # 사용자 고유 ID
    description: str
    preferences: Dict[str, float]
    risk_tolerance: float
    decision_speed: float
    social_influence: float

    investment_horizon: Horizon
    analysis_preference: AnalysisStyle
    volatility_tolerance: float
    dividend_preference: float
    expert_reliance: float

    created_at: str = field(
        default_factory=lambda: __import__("datetime").datetime.utcnow().isoformat()
    )

    # ------------------------ Validation ---------------------------------- #
    def __post_init__(self) -> None:
        """입력값 유효성 검사. 0~1 범위 값/레벨 체크."""
        # 0~1 범위 체크
        for f_name in [
            "risk_tolerance",
            "decision_speed",
            "social_influence",
            "volatility_tolerance",
            "dividend_preference",
            "expert_reliance",
        ]:
            value = getattr(self, f_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{f_name} must be 0–1 (got {value})")

        if not 1 <= self.investment_level <= 5:
            raise ValueError("investment_level must be 1–5")

        if any(not 0.0 <= v <= 1.0 for v in self.preferences.values()):
            raise ValueError("preferences values must be 0–1")

    # ----------------------- Helper methods ------------------------------- #
    def get_persona_id(self) -> str:
        """페르소나 고유 ID 문자열을 반환합니다.

        Returns:
            str: "{mbti}_{investment_level}_{user_id}" 형식의 문자열.
        """
        return f"{self.mbti}_{self.investment_level}_{self.user_id}"

    def to_dict(self) -> Dict:
        """PersonaConfig 객체를 dict로 변환합니다.

        Returns:
            dict: 직렬화된 딕셔너리 (Enum은 값으로 변환).
        """
        data = asdict(self)
        data["investment_horizon"] = self.investment_horizon.value
        data["analysis_preference"] = self.analysis_preference.value
        return data

    @classmethod
    def from_dict(cls, d: Dict) -> PersonaConfig:
        """dict로부터 PersonaConfig 인스턴스를 생성합니다.

        Args:
            d (dict): 직렬화된 데이터.

        Returns:
            PersonaConfig: 생성된 인스턴스.
        """
        d = deepcopy(d)
        d["investment_horizon"] = Horizon(d["investment_horizon"])
        d["analysis_preference"] = AnalysisStyle(d["analysis_preference"])
        return cls(**d)


# 4. 팩토리 함수


def create_persona_from_user_data(
    user_id: int, mbti: str, investment_level: int
) -> PersonaConfig:
    """사용자 정보로부터 PersonaConfig 객체를 생성합니다.

    Args:
        user_id (int): 사용자 ID.
        mbti (str): MBTI 유형.
        investment_level (int): 투자 수준 (1~5).

    Returns:
        PersonaConfig: 구성된 페르소나 인스턴스.
    """
    # MBTI 기본값 가져오기 (overrides 없음)
    defaults = deepcopy(MBTI_DEFAULTS.get(mbti, MBTI_FALLBACK))

    # 기본 정보 설정
    data = {
        **defaults,
        "mbti": mbti,
        "investment_level": investment_level,
        "user_id": user_id,
        "description": f"{mbti} 유형 투자자",
    }

    # Enum 타입 변환
    if not isinstance(data["investment_horizon"], Horizon):
        data["investment_horizon"] = Horizon(data["investment_horizon"])

    if not isinstance(data["analysis_preference"], AnalysisStyle):
        data["analysis_preference"] = AnalysisStyle(data["analysis_preference"])

    return PersonaConfig(**data)
