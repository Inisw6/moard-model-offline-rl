from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces


class BaseEnv(ABC):
    """
    강화학습 환경의 기본 인터페이스.

    Attributes:
        observation_space (spaces.Space): 관찰 벡터의 공간 분포.
        action_space (spaces.Space): 행동 공간 분포.
    """

    @property
    @abstractmethod
    def observation_space(self) -> spaces.Space:
        """
        관찰(observation) 벡터의 공간 분포를 반환합니다.

        Returns:
            spaces.Space: 상태 공간 객체 (예: Box).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def action_space(self) -> spaces.Space:
        """
        행동(action) 공간 분포를 반환합니다.

        Returns:
            spaces.Space: 행동 공간 객체 (예: Discrete, Tuple 등).
        """
        raise NotImplementedError

    @abstractmethod
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        에피소드 초기화 및 시작 상태를 반환합니다.

        Args:
            seed (Optional[int]): 랜덤 시드.
            options (Optional[Dict[str, Any]]): 추가 옵션.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (초기 상태 벡터, info 딕셔너리)
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        하나의 환경 스텝을 실행하고 결과를 반환합니다.

        Args:
            action (Any): 수행할 행동.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
                next_state: 다음 상태 벡터
                reward: 즉시 보상
                done: 에피소드 종료 여부
                truncated: 시간 제한 등으로 인한 중단 여부
                info: 추가 정보 딕셔너리
        """
        raise NotImplementedError


class BaseAgent(ABC):
    """
    강화학습 에이전트의 기본 인터페이스.
    """

    @abstractmethod
    def select_action(
        self, user_state: np.ndarray, candidate_embs: List[np.ndarray]
    ) -> int:
        """
        현재 정책에 따라 행동을 선택합니다.

        Args:
            user_state (np.ndarray): 상태 벡터.
            candidate_embs (List[np.ndarray]): 후보 콘텐츠 임베딩 리스트.

        Returns:
            int: 선택한 후보 인덱스.
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
        경험을 버퍼에 저장합니다.

        Args:
            user_state (np.ndarray): 상태 벡터.
            content_emb (np.ndarray): 행동에 해당하는 콘텐츠 임베딩.
            reward (float): 보상값.
            next_state (np.ndarray): 다음 상태 벡터.
            next_cands_embs (Dict[str, List[np.ndarray]]): 다음 후보군 임베딩.
            done (bool): 에피소드 종료 여부.
        """
        raise NotImplementedError

    @abstractmethod
    def learn(self) -> None:
        """
        버퍼에서 샘플을 뽑아 정책 또는 Q-네트워크를 업데이트합니다.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """
        모델 상태를 저장합니다.

        Args:
            path (str): 저장 파일 경로.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """
         저장된 모델 상태를 불러옵니다.

        Args:
            path (str): 로드할 파일 경로.
        """
        raise NotImplementedError


class BaseUserEmbedder(ABC):
    """
    사용자 임베딩 추상 클래스.
    """

    @abstractmethod
    def embed_user(self, user: Dict[str, Any]) -> np.ndarray:
        """
        사용자 정보를 벡터로 임베딩합니다.

        Args:
            user (Dict[str, Any]): 사용자 정보 딕셔너리.

        Returns:
            np.ndarray: user_dim 크기의 임베딩 벡터.
        """
        raise NotImplementedError

    @abstractmethod
    def output_dim(self) -> int:
        """
        임베딩 벡터의 차원을 반환합니다.

        Returns:
            int: user_dim 값.
        """
        raise NotImplementedError


class BaseContentEmbedder(ABC):
    """
    콘텐츠 임베딩 추상 클래스.
    """

    @abstractmethod
    def embed_content(self, content: Dict[str, Any]) -> np.ndarray:
        """
        콘텐츠 정보를 벡터로 임베딩합니다.

        Args:
            content (Dict[str, Any]): 콘텐츠 정보 딕셔너리.

        Returns:
            np.ndarray: content_dim 크기의 임베딩 벡터.
        """
        raise NotImplementedError

    @abstractmethod
    def output_dim(self) -> int:
        """
        임베딩 벡터의 차원을 반환합니다.

        Returns:
            int: content_dim 값.
        """
        raise NotImplementedError


class BaseEmbedder(ABC):
    """
    유저 임베더와 콘텐츠 임베더를 조합하는 기본 클래스.
    """

    def __init__(
        self, user_embedder: BaseUserEmbedder, content_embedder: BaseContentEmbedder
    ):
        """
        BaseEmbedder 생성자.

        Args:
            user_embedder (BaseUserEmbedder): 사용자 임베더.
            content_embedder (BaseContentEmbedder): 콘텐츠 임베더.
        """
        self.user_embedder = user_embedder
        self.content_embedder = content_embedder

    def embed_user(self, user: Dict[str, Any]) -> np.ndarray:
        """
        사용자 정보를 임베딩합니다.

        Args:
            user (Dict[str, Any]): 사용자 정보 딕셔너리.

        Returns:
            np.ndarray: 임베딩 벡터.
        """
        return self.user_embedder.embed_user(user)

    def embed_content(self, content: Dict[str, Any]) -> np.ndarray:
        """
        콘텐츠 정보를 임베딩합니다.

        Args:
            content (Dict[str, Any]): 콘텐츠 정보 딕셔너리.

        Returns:
            np.ndarray: 임베딩 벡터.
        """
        return self.content_embedder.embed_content(content)

    def output_dim(self) -> int:
        """
        유저 임베더 출력 차원을 반환합니다.

        Returns:
            int: user_dim 값.
        """
        return self.user_embedder.output_dim()


class BaseCandidateGenerator(ABC):
    """
    후보군 생성기 추상 클래스.
    """

    @abstractmethod
    def get_candidates(
        self, state: Optional[np.ndarray]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        주어진 상태에서 후보 콘텐츠 목록을 반환합니다.

        Args:
            state (Optional[np.ndarray]): 사용자 상태 임베딩.

        Returns:
            Dict[str, List[Dict[str, Any]]]: 타입별 콘텐츠 딕셔너리 목록.
        """
        raise NotImplementedError


class BaseRewardFn(ABC):
    """
    보상 함수 추상 클래스.
    """

    @abstractmethod
    def calculate(self, content: Dict[str, Any], event_type: str = "VIEW") -> float:
        """
        보상을 계산합니다.

        Args:
            content (Dict[str, Any]): 콘텐츠 정보.
            event_type (str): 이벤트 타입 ('VIEW' 또는 'CLICK').

        Returns:
            float: 보상값.
        """
        raise NotImplementedError


class BaseResponseSimulator(ABC):
    """
    사용자 반응 시뮬레이터의 기본 클래스.
    """

    @abstractmethod
    def simulate_responses(
        self, selected_contents: List[Dict], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        추천 콘텐츠에 대한 사용자 반응을 시뮬레이션합니다.

        Args:
            selected_contents (List[Dict[str, Any]]): 추천된 콘텐츠 리스트.
            context (Dict[str, Any]): 시뮬레이션 컨텍스트.

        Returns:
            List[Dict[str, Any]]: 콘텐츠별 사용자 반응 리스트.
        """
        raise NotImplementedError
