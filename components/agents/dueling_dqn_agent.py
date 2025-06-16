import logging
import random
from itertools import chain
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from components.core.base import BaseAgent
from components.registry import register
from models.dueling_q_network import DuelingQNetwork
from replay.replay_buffer import ReplayBuffer


@register("dueling_dqn")
class DuelingDQNAgent(BaseAgent):
    """
    Dueling DQN 기반 추천 에이전트.

    Attributes:
        device (torch.device): 사용할 디바이스 (CPU 또는 GPU).
        q_net (DuelingQNetwork): 학습용 Q 네트워크.
        target_q_net (DuelingQNetwork): 타겟 Q 네트워크.
        optimizer (torch.optim.Optimizer): 옵티마이저.
        buffer (ReplayBuffer): 리플레이 버퍼.
        gamma (float): 할인 계수.
        batch_size (int): 배치 크기.
        epsilon (float): 탐험률.
        epsilon_min (float): 최소 탐험률.
        epsilon_dec (float): 탐험률 감소 계수.
        update_freq (int): 타겟 네트워크 업데이트 빈도.
        step_count (int): 학습 단계 카운터.
        loss_type (str): 손실 함수 종류.
    """

    def __init__(
        self,
        user_dim: int,
        content_dim: int,
        lr: float,
        batch_size: int,
        eps_start: float,
        eps_min: float,
        eps_decay: float,
        gamma: float,
        update_freq: int,
        capacity: int,
        loss_type: str = "smooth_l1",
        device: str = "cpu",
    ) -> None:
        """
        DuelingDQNAgent 생성자.

        Args:
            user_dim (int): 사용자 상태 임베딩 차원.
            content_dim (int): 콘텐츠 임베딩 차원.
            lr (float): 학습률.
            batch_size (int): 배치 크기.
            eps_start (float): 초기 탐험률.
            eps_min (float): 최소 탐험률.
            eps_decay (float): 탐험률 감소 계수.
            gamma (float): 할인 계수.
            update_freq (int): 타겟 네트워크 업데이트 주기.
            capacity (int): 리플레이 버퍼 용량.
            loss_type (str): 손실 함수 종류 ('mse' 또는 'smooth_l1').
            device (str): 사용할 디바이스 ('cpu' 또는 'cuda').
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else device)
        self.q_net = DuelingQNetwork(user_dim, content_dim).to(self.device)
        self.target_q_net = DuelingQNetwork(user_dim, content_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=capacity)

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = eps_start
        self.epsilon_min = eps_min
        self.epsilon_dec = eps_decay
        self.update_freq = update_freq
        self.step_count = 0
        self.loss_type = loss_type

    def select_action(
        self,
        user_state: List[float],
        candidate_embs: List[List[float]],
    ) -> int:
        """
        ε-greedy를 통해 행동을 선택합니다.

        Args:
            user_state (List[float]): 사용자 상태 벡터.
            candidate_embs (List[List[float]]): 후보 콘텐츠 임베딩 리스트.

        Returns:
            int: 선택된 행동(콘텐츠 인덱스).
        """
        if random.random() < self.epsilon:
            return random.randrange(len(candidate_embs))

        state_tensor = torch.tensor(user_state, dtype=torch.float32, device=self.device)
        state_rep = state_tensor.unsqueeze(0).repeat(len(candidate_embs), 1)
        cand_tensor = torch.tensor(
            candidate_embs, dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            q_vals = self.q_net(state_rep, cand_tensor).squeeze(1)

        return int(q_vals.argmax().item())

    def store(
        self,
        user_state: List[float],
        content_emb: List[float],
        reward: float,
        next_state: List[float],
        next_cands_embs: Dict[str, List[List[float]]],
        done: bool,
    ) -> None:
        """
        경험을 리플레이 버퍼에 저장합니다.

        Args:
            user_state (List[float]): 현재 상태 임베딩.
            content_emb (List[float]): 행동에 해당하는 콘텐츠 임베딩.
            reward (float): 보상.
            next_state (List[float]): 다음 상태 임베딩.
            next_cands_embs (Dict[str, List[List[float]]]): 다음 상태 후보 임베딩.
            done (bool): 에피소드 종료 여부.
        """
        self.buffer.push(
            (user_state, content_emb), reward, (next_state, next_cands_embs), done
        )

    def learn(self) -> float:
        """
        미니배치를 학습하여 Q 네트워크를 업데이트합니다.

        Returns:
            float: 현재 학습 손실 값.
        """
        if len(self.buffer) < self.batch_size:
            return 0.0

        self.step_count += 1
        us_batch, ce_batch, rewards, next_info, dones = self.buffer.sample(
            self.batch_size
        )
        next_states, next_cands_embs = next_info

        us = torch.tensor(us_batch, dtype=torch.float32, device=self.device)
        ce = torch.tensor(ce_batch, dtype=torch.float32, device=self.device)
        rs = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        ds = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q_net(us, ce)

        # 다음 상태의 최대 Q 값 계산
        flat_states, flat_cands, batch_indices = [], [], []
        for i, (ns, nxt) in enumerate(zip(next_states, next_cands_embs)):
            all_embs = list(chain.from_iterable(nxt.values()))
            for emb in all_embs:
                flat_states.append(ns)
                flat_cands.append(emb)
                batch_indices.append(i)

        if flat_states:
            fs = torch.tensor(flat_states, dtype=torch.float32, device=self.device)
            fc = torch.tensor(flat_cands, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                q_flat = self.target_q_net(fs, fc).squeeze(1).cpu().numpy()
            batch_indices = np.array(batch_indices)
            max_q = [
                q_flat[batch_indices == i].max() if np.any(batch_indices == i) else 0.0
                for i in range(len(next_states))
            ]
            max_nq = torch.tensor(
                max_q, dtype=torch.float32, device=self.device
            ).unsqueeze(1)
        else:
            max_nq = torch.zeros((len(next_states), 1), device=self.device)

        target = rs + self.gamma * max_nq * (1 - ds)

        if self.loss_type == "mse":
            loss = F.mse_loss(q_sa, target)
        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(q_sa, target)
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.step_count % self.update_freq == 0:
            logging.info(
                f"Step {self.step_count}: loss={loss.item():.4f}, epsilon={self.epsilon:.4f}"
            )
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def select_slate(
        self,
        state: List[float],
        candidate_embs: Dict[str, List[List[float]]],
        max_recs: int,
    ) -> List[Tuple[str, int]]:
        """
        슬레이트 추천을 수행합니다 (ε-greedy).

        Args:
            state (List[float]): 현재 상태 벡터.
            candidate_embs (Dict[str, List[List[float]]]): 타입별 후보 임베딩.
            max_recs (int): 추천할 아이템 최대 개수.

        Returns:
            List[Tuple[str, int]]: (콘텐츠 타입, 인덱스) 리스트.
        """
        all_cands = [
            (t, i) for t, embs in candidate_embs.items() for i in range(len(embs))
        ]
        if not all_cands:
            return []

        if random.random() < self.epsilon:
            return random.sample(all_cands, min(max_recs, len(all_cands)))

        slate_scores = []
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self.q_net.eval()
        for ctype, embs in candidate_embs.items():
            if not embs:
                continue
            cand_tensor = torch.tensor(embs, dtype=torch.float32, device=self.device)
            reps = state_tensor.repeat(len(embs), 1)
            with torch.no_grad():
                q_vals = self.q_net(reps, cand_tensor).squeeze(1)
            slate_scores.extend(
                [((ctype, idx), q.item()) for idx, q in enumerate(q_vals)]
            )
        self.q_net.train()

        slate_scores.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in slate_scores[:max_recs]]

    def decay_epsilon(self) -> None:
        """
        탐험률을 감소시킵니다.
        """
        self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)

    def save(self, path: str) -> None:
        """
        모델 상태를 파일로 저장합니다.

        Args:
            path (str): 저장할 체크포인트 파일 경로.
        """
        checkpoint = {
            "q_net_state": self.q_net.state_dict(),
            "target_net_state": self.target_q_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "epsilon": self.epsilon,
        }
        torch.save(checkpoint, path)
        logging.info(f"Dueling DQN checkpoint saved to {path}")

    def load(self, path: str) -> None:
        """
        체크포인트에서 모델 상태를 복원합니다.

        Args:
            path (str): 로드할 체크포인트 파일 경로.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net_state"])
        self.target_q_net.load_state_dict(ckpt["target_net_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.step_count = ckpt.get("step_count", 0)
        self.epsilon = ckpt.get("epsilon", self.epsilon)
        self.q_net.train()
        self.target_q_net.eval()
        logging.info(f"Dueling DQN checkpoint loaded from {path}")
