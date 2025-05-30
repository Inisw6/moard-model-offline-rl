from abc import ABC, abstractmethod
import numpy as np

class BaseEnv(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, *args, **kwargs):
        pass

    @abstractmethod
    def store(self, *args, **kwargs):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass

class BaseEmbedder(ABC):
    @abstractmethod
    def embed_user(self, user: dict) -> np.ndarray:
        pass

    @abstractmethod
    def embed_content(self, content: dict) -> np.ndarray:
        pass

    @abstractmethod
    def estimate_preference(self, state: np.ndarray) -> dict:
        pass

    @abstractmethod
    def output_dim(self) -> int:
        pass

class BaseCandidateGenerator(ABC):
    @abstractmethod
    def get_candidates(self, state: np.ndarray) -> dict:
        pass

class BaseRewardFn(ABC):
    @abstractmethod
    def calculate(self, content: dict) -> float:
        pass