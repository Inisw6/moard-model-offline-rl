from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces


class BaseEnv(ABC):
    """강화학습 환경의 기본 인터페이스."""

    @property
    @abstractmethod
    def observation_space(self) -> spaces.Space:
        """관찰(observation) 벡터의 공간 분포를 반환합니다.

        Returns:
            spaces.Space: 상태 공간 객체 (예: Box).
        """
        pass

    @property
    @abstractmethod
    def action_space(self) -> spaces.Space:
        """행동(action) 공간 분포를 반환합니다.

        Returns:
            spaces.Space: 행동 공간 객체 (예: Discrete, Tuple 등).
        """
        pass

    @abstractmethod
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """에피소드 초기화 및 시작 상태를 반환합니다.

        Args:
            seed (Optional[int], optional): 랜덤 시드. Defaults to None.
            options (Optional[Dict[str, Any]], optional): 추가 옵션. Defaults to None.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 초기 상태 벡터와 info 딕셔너리.
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """환경에 액션을 적용하고 다음 상태를 반환합니다.

        Args:
            action (Any): 에이전트의 행동.

        Returns:
            Tuple[
                np.ndarray,  # next_state
                float,       # reward
                bool,        # terminated
                bool,        # truncated
                Dict[str, Any]  # info
            ]: 스텝 결과 정보.
        """
        pass


class BaseAgent(ABC):
    """강화학습 에이전트의 기본 인터페이스."""

    @abstractmethod
    def select_action(
        self, user_state: np.ndarray, candidate_embs: List[np.ndarray]
    ) -> int:
        """현재 정책(policy)에 따라 행동을 선택합니다.

        Args:
            user_state (np.ndarray): 사용자 상태 벡터.
            candidate_embs (List[np.ndarray]): 후보 콘텐츠 임베딩 리스트.

        Returns:
            int: 선택한 후보 인덱스.
        """
        pass

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
        """경험을 리플레이 버퍼에 저장합니다.

        Args:
            user_state (np.ndarray): 상태 벡터.
            content_emb (np.ndarray): 콘텐츠 임베딩.
            reward (float): 보상값.
            next_state (np.ndarray): 다음 상태 벡터.
            next_cands_embs (Dict[str, List[np.ndarray]]): 다음 후보군 임베딩.
            done (bool): 에피소드 종료 여부.
        """
        pass

    @abstractmethod
    def learn(self) -> Optional[float]:
        """버퍼에서 샘플을 추출해 정책 혹은 Q-네트워크를 업데이트합니다.

        Returns:
            Optional[float]: 업데이트 손실값, 또는 업데이트가 수행되지 않으면 None.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """모델 파라미터 및 상태를 저장합니다.

        Args:
            path (str): 저장 파일 경로.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """저장된 모델 파라미터 및 상태를 불러옵니다.

        Args:
            path (str): 저장 파일 경로.
        """
        pass


class BaseUserEmbedder(ABC):
    """사용자 임베딩 추상 클래스."""

    @abstractmethod
    def embed_user(self, user: Dict[str, Any]) -> np.ndarray:
        """사용자 정보를 임베딩 벡터로 변환합니다.

        Args:
            user (Dict[str, Any]): 사용자 정보 딕셔너리.

        Returns:
            np.ndarray: 길이 output_dim 벡터.
        """
        pass

    @abstractmethod
    def output_dim(self) -> int:
        """임베딩 벡터의 차원을 반환합니다.

        Returns:
            int: 임베딩 벡터 차원.
        """
        pass


class BaseContentEmbedder(ABC):
    """콘텐츠 임베딩 추상 클래스."""

    @abstractmethod
    def embed_content(self, content: Dict[str, Any]) -> np.ndarray:
        """콘텐츠 정보를 임베딩합니다.

        Args:
            content (Dict[str, Any]): 콘텐츠 정보 딕셔너리.

        Returns:
            np.ndarray: 길이 output_dim 벡터.
        """
        pass

    @abstractmethod
    def output_dim(self) -> int:
        """임베딩 벡터의 차원을 반환합니다.

        Returns:
            int: 임베딩 벡터 차원.
        """
        pass


class BaseEmbedder:
    """유저 및 콘텐츠 임베더를 조합하는 기본 클래스."""

    def __init__(
        self, user_embedder: BaseUserEmbedder, content_embedder: BaseContentEmbedder
    ):
        """생성자.

        Args:
            user_embedder (BaseUserEmbedder): 사용자 임베더 인스턴스.
            content_embedder (BaseContentEmbedder): 콘텐츠 임베더 인스턴스.
        """
        self.user_embedder = user_embedder
        self.content_embedder = content_embedder

    def embed_user(self, user: Dict[str, Any]) -> np.ndarray:
        """사용자 정보를 임베딩합니다."""
        return self.user_embedder.embed_user(user)

    def embed_content(self, content: Dict[str, Any]) -> np.ndarray:
        """콘텐츠 정보를 임베딩합니다."""
        return self.content_embedder.embed_content(content)

    def output_dim(self) -> int:
        """임베딩 차원을 반환합니다."""
        return self.user_embedder.output_dim()


class BaseCandidateGenerator(ABC):
    """후보군 생성기 추상 클래스."""

    @abstractmethod
    def get_candidates(
        self, state: Optional[np.ndarray]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """주어진 상태에서 후보 콘텐츠를 생성합니다.

        Args:
            state (Optional[np.ndarray]): 사용자 상태 벡터.

        Returns:
            Dict[str, List[Dict[str, Any]]]: {타입: [콘텐츠 딕셔너리, ...]}."""
        pass


class BaseRewardFn(ABC):
    """보상 함수 추상 클래스."""

    @abstractmethod
    def calculate(self, content: Dict[str, Any], event_type: str = "VIEW") -> float:
        """보상값을 계산합니다.

        Args:
            content (Dict[str, Any]): 콘텐츠 정보.
            event_type (str, optional): 이벤트 타입. Defaults to "VIEW".

        Returns:
            float: 보상값.
        """
        pass


class BaseResponseSimulator(ABC):
    """사용자 반응 시뮬레이터 기본 클래스."""

    @abstractmethod
    def simulate_responses(
        self, selected_contents: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """사용자 반응을 시뮬레이션합니다.

        Args:
            selected_contents (List[Dict[str, Any]]): 추천된 콘텐츠 리스트.
            context (Dict[str, Any]): 시뮬레이션 컨텍스트.

        Returns:
            List[Dict[str, Any]]: 반응 리스트.
        """
        pass
