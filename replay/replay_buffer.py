import random
from typing import Any, List, Tuple


class ReplayBuffer:
    """
    간단한 경험 리플레이 버퍼.

    Args:
        capacity (int): 최대 저장 가능한 transition 개수.
    """

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self.buffer: List[Any] = []

    def push(
        self,
        state_cont_pair: Tuple[Any, Any],
        reward: float,
        next_info: Tuple[Any, Any],
        done: bool,
    ) -> None:
        """
        transition을 버퍼에 추가합니다.

        Args:
            state_cont_pair (tuple): (user_state, content_emb)
            reward (float): 보상
            next_info (tuple): (next_state, next_cands_embs)
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
        """
        무작위로 배치 크기만큼 transition을 샘플링합니다.

        Returns:
            (user_states, content_embs, rewards, (next_states, next_cands_embs), dones)
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
        """현재 버퍼에 저장된 transition 개수를 반환합니다."""
        return len(self.buffer)
