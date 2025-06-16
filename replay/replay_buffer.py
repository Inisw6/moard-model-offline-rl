import random
from collections import deque
from typing import Any, Deque, List, Tuple


class ReplayBuffer:
    """경험을 저장하고 배치 샘플링을 지원하는 리플레이 버퍼 클래스.

    Attributes:
        capacity (int): 최대 저장 가능 transition 개수.
        buffer (Deque): transition 저장소.
    """

    def __init__(self, capacity: int = 10000) -> None:
        """ReplayBuffer 생성자.

        Args:
            capacity (int, optional): 최대 transition 개수. 기본값은 10000.
        """
        self.capacity: int = capacity
        self.buffer: Deque[
            Tuple[
                Tuple[Any, Any],  # (user_state, content_emb)
                float,  # reward
                Tuple[Any, Any],  # (next_state, next_cands_embs)
                bool,  # done
            ]
        ] = deque(maxlen=capacity)

    def push(
        self,
        state_action_pair: Tuple[Any, Any],
        reward: float,
        next_transition: Tuple[Any, Any],
        done: bool,
    ) -> None:
        """새 transition을 버퍼에 추가합니다.

        Args:
            state_action_pair (Tuple[Any, Any]): (user_state, content_emb)
            reward (float): 보상 값
            next_transition (Tuple[Any, Any]): (next_state, next_cands_embs)
            done (bool): 에피소드 종료 여부
        """
        self.buffer.append((state_action_pair, reward, next_transition, done))

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
        state_action_pairs, rewards, next_transitions, dones = zip(*batch)
        user_states, content_embs = zip(*state_action_pairs)
        next_states, next_cands_embs = zip(*next_transitions)

        return (
            list(user_states),
            list(content_embs),
            list(rewards),
            (list(next_states), list(next_cands_embs)),
            list(dones),
        )

    def __len__(self) -> int:
        """현재 버퍼에 저장된 transition 개수를 반환합니다.

        Returns:
            int: 저장된 transition의 개수
        """
        return len(self.buffer)
