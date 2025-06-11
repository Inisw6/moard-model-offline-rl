import random
from typing import Any, List, Tuple


class ReplayBuffer:
    """간단한 경험 리플레이 버퍼 클래스.

    Args:
        capacity (int): 최대 저장 가능한 transition 개수.
    """

    def __init__(self, capacity: int = 10000) -> None:
        """ReplayBuffer 생성자.

        Args:
            capacity (int, optional): 최대 transition 개수. 기본값은 10000.
        """
        self.capacity = capacity
        self.buffer: List[Any] = []

    def push(
        self,
        state_cont_pair: Tuple[Any, Any],
        reward: float,
        next_info: Tuple[Any, Any],
        done: bool,
    ) -> None:
        """transition(상태, 행동 결과 등)을 버퍼에 추가합니다.

        Args:
            state_cont_pair (Tuple[Any, Any]): (user_state, content_emb)
            reward (float): 보상 값
            next_info (Tuple[Any, Any]): (next_state, next_cands_embs)
            done (bool): 에피소드 종료 여부
        """
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state_cont_pair, reward, next_info, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[
        List[Any], List[Any], List[float], Tuple[List[Any], List[Any]], List[bool]
    ]:
        """랜덤하게 batch_size 개수만큼 transition을 샘플링합니다.

        Args:
            batch_size (int): 샘플링할 transition 개수

        Returns:
            Tuple:
                - user_states (List[Any])
                - content_embs (List[Any])
                - rewards (List[float])
                - (next_states, next_cands_embs) (Tuple[List[Any], List[Any]])
                - dones (List[bool])

        Raises:
            ValueError: buffer 크기보다 batch_size가 더 클 때 발생
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Sample size {batch_size} greater than buffer size {len(self.buffer)}"
            )
        batch = random.sample(self.buffer, batch_size)
        sc, r, ni, d = zip(*batch)
        s, ce = zip(*sc)
        ns, next_embs = zip(*ni)
        return list(s), list(ce), list(r), (list(ns), list(next_embs)), list(d)

    def __len__(self) -> int:
        """현재 버퍼에 저장된 transition 개수를 반환합니다.

        Returns:
            int: 저장된 transition의 개수
        """
        return len(self.buffer)
