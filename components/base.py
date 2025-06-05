from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from gymnasium import spaces


class BaseEnv(ABC):
    @property
    @abstractmethod
    def observation_space(self) -> spaces.Space:
        """관찰(observation) 벡터의 공간 분포 정의 (예: Box)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def action_space(self) -> spaces.Space:
        """행동(action) 공간 분포 정의 (예: Discrete, Tuple)."""
        raise NotImplementedError

    @abstractmethod
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        에피소드 시작 시 호출.
        return: (초기 상태 벡터, info 딕셔너리)
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        한 스텝 진행.
        action: 행동 인덱스 또는 튜플 형태 등 (하위 클래스별 규약).
        return: (다음 상태 벡터, 보상, done(종료 여부), truncated(시간 초과 여부), info)
        """
        raise NotImplementedError


class BaseAgent(ABC):
    @abstractmethod
    def select_action(
        self, user_state: np.ndarray, candidate_embs: List[np.ndarray]
    ) -> int:
        """
        현재 정책(policy)에 따라 행동 선택.
        return: 선택한 후보 인덱스(int)
        """
        raise NotImplementedError

    @abstractmethod
    def store(
        self,
        user_state: np.ndarray,
        content_emb: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        next_cands_embs: Dict[str, List[np.ndarray]],
        done: bool,
    ) -> None:
        """
        (s, a_embedding, r, s', next_candidate_embeddings, done) 튜플을 리플레이 버퍼에 저장.
        """
        raise NotImplementedError

    @abstractmethod
    def learn(self) -> None:
        """
        리플레이 버퍼에서 샘플을 뽑아 Q-네트워크(혹은 정책)를 업데이트.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """현재 모델 파라미터 및 필요한 상태(epsilon 등)를 저장."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """저장된 모델 파라미터 및 상태를 불러옴."""
        raise NotImplementedError


class BaseUserEmbedder(ABC):
    @abstractmethod
    def embed_user(self, user: Dict[str, Any]) -> np.ndarray:
        """
        사용자 정보 딕셔너리를 받아서 user_dim 길이 벡터로 변환.
        예: {"user_info":..., "recent_logs":[...], "current_time":...}
        return: np.ndarray (shape: [user_dim])
        """
        raise NotImplementedError

    @abstractmethod
    def estimate_preference(self, state: np.ndarray) -> Dict[str, float]:
        """
        state 벡터(유저 임베딩)에서 콘텐츠 타입별 선호도 딕셔너리 반환.
        예: {"youtube":0.2, "blog":0.5, "news":0.3}
        """
        raise NotImplementedError

    @abstractmethod
    def output_dim(self) -> int:
        """embed_user가 반환하는 벡터의 차원(user_dim)을 반환."""
        raise NotImplementedError


class BaseContentEmbedder(ABC):
    @abstractmethod
    def embed_content(self, content: Dict[str, Any]) -> np.ndarray:
        """
        콘텐츠 딕셔너리를 받아서 content_dim 길이 벡터로 변환.
        예: {"id":..., "type":..., "embedding": "[...]", ...}
        return: np.ndarray (shape: [content_dim])
        """
        raise NotImplementedError

    @abstractmethod
    def output_dim(self) -> int:
        """embed_content가 반환하는 벡터의 차원(content_dim)을 반환."""
        raise NotImplementedError


class BaseEmbedder(ABC):
    """유저와 콘텐츠 임베더를 조합하는 인터페이스"""

    def __init__(
        self, user_embedder: BaseUserEmbedder, content_embedder: BaseContentEmbedder
    ):
        self.user_embedder = user_embedder
        self.content_embedder = content_embedder

    def embed_user(self, user: Dict[str, Any]) -> np.ndarray:
        return self.user_embedder.embed_user(user)

    def embed_content(self, content: Dict[str, Any]) -> np.ndarray:
        return self.content_embedder.embed_content(content)

    def estimate_preference(self, state: np.ndarray) -> Dict[str, float]:
        return self.user_embedder.estimate_preference(state)

    def output_dim(self) -> int:
        return self.user_embedder.output_dim()


class BaseCandidateGenerator(ABC):
    @abstractmethod
    def get_candidates(
        self, state: Optional[np.ndarray]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        주어진 사용자 상태(state)를 바탕으로 각 콘텐츠 타입별 후보 목록을 반환.
        return: {type: [content_dict, ...], ...}
        """
        raise NotImplementedError


class BaseRewardFn(ABC):
    @abstractmethod
    def calculate(self, content: Dict[str, Any], event_type: str = "VIEW") -> float:
        """
        콘텐츠 정보와 이벤트 타입("VIEW" or "CLICK")을 받아 보상값 반환.
        return: 보상(float)
        """
        raise NotImplementedError
