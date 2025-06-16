from itertools import chain
import logging
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from components.core.base import BaseAgent
from components.registry import register
from models.q_network import QNetwork
from replay.replay_buffer import ReplayBuffer


@register("dqn")
class DQNAgent(BaseAgent):
    """
    DQN 기반 추천 에이전트.

    Attributes:
        device (torch.device): 사용할 디바이스(CPU 또는 GPU).
        q_net (QNetwork): Q 네트워크.
        target_q_net (QNetwork): 타겟 Q 네트워크.
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
        DQNAgent 생성자.

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
        self.q_net = QNetwork(user_dim, content_dim).to(self.device)
        self.target_q_net = QNetwork(user_dim, content_dim).to(self.device)
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
        self, user_state: List[float], candidate_embs: List[List[float]]
    ) -> int:
        """
        현재 상태에서 행동을 선택합니다 (ε-greedy).

        Args:
            user_state (List[float]): 사용자 상태 벡터.
            candidate_embs (List[List[float]]): 후보 콘텐츠 임베딩 리스트.

        Returns:
            int: 선택된 행동(콘텐츠 인덱스).
        """
        if random.random() < self.epsilon:
            return random.randrange(len(candidate_embs))
        us = torch.FloatTensor(user_state).unsqueeze(0).to(self.device)
        us_rep = us.repeat(len(candidate_embs), 1)
        ce = torch.FloatTensor(candidate_embs).to(self.device)
        with torch.no_grad():
            q_vals = self.q_net(us_rep, ce).squeeze(1)
        return int(torch.argmax(q_vals).item())

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
            next_cands_embs (Dict[str, List[List[float]]]): 다음 상태의 후보군 임베딩.
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
            return

        self.step_count += 1

        # 미니배치 샘플 추출
        user_states, content_embs, rewards, next_info, dones = self.buffer.sample(
            self.batch_size
        )
        next_states, next_cands_embs = next_info

        # 텐서 변환
        us = torch.tensor(user_states, dtype=torch.float32, device=self.device)
        ce = torch.tensor(content_embs, dtype=torch.float32, device=self.device)
        rs = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        ds = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # 현재 Q 값
        q_sa = self.q_net(us, ce)

        # 다음 상태에서의 최대 Q값을 벡터화로 계산
        flat_states, flat_cands, batch_indices = [], [], []
        for i, (ns, nxt) in enumerate(zip(next_states, next_cands_embs)):
            # nxt: Dict[str, List[List[float]]] - 모든 타입의 후보 임베딩 합치기
            all_embs = list(chain.from_iterable(nxt.values()))
            for cand in all_embs:
                flat_states.append(ns)
                flat_cands.append(cand)
                batch_indices.append(i)

        if flat_states:
            # flat_states: (총 후보수, state_dim), flat_cands: (총 후보수, cand_dim)
            flat_states_tensor = torch.tensor(
                flat_states, dtype=torch.float32, device=self.device
            )
            flat_cands_tensor = torch.tensor(
                flat_cands, dtype=torch.float32, device=self.device
            )
            with torch.no_grad():
                q_flat = (
                    self.target_q_net(flat_states_tensor, flat_cands_tensor)
                    .squeeze(1)
                    .cpu()
                    .numpy()
                )
            batch_indices = np.array(batch_indices)
            max_q_per_sample = []
            for i in range(len(next_states)):
                sample_qs = q_flat[batch_indices == i]
                # 후보가 없으면 Q=0
                max_q_per_sample.append(sample_qs.max() if len(sample_qs) > 0 else 0.0)
            max_nq = torch.tensor(
                max_q_per_sample, dtype=torch.float32, device=self.device
            ).unsqueeze(1)
        else:
            max_nq = torch.zeros((len(next_states), 1), device=self.device)

        # 타겟 계산
        target = rs + self.gamma * max_nq * (1 - ds)

        # 손실 계산
        if self.loss_type == "mse":
            loss = F.mse_loss(q_sa, target)
        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(q_sa, target)
        else:
            raise ValueError(f"지원하지 않는 loss_type입니다: {self.loss_type}")

        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 타겟 네트워크 업데이트
        if self.step_count % self.update_freq == 0:
            logging.info(
                f"Step {self.step_count}: Loss = {loss.item()}, Epsilon = {self.epsilon:.4f}"
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

    def decay_epsilon(self):
        """
        탐험률(epsilon)을 감소시킵니다.
        """
        self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)

    def save(self, path: str) -> None:
        """
        모델 상태를 체크포인트 파일로 저장합니다.

        Args:
            path (str): 저장할 파일 경로.
        """
        checkpoint = {
            "q_net_state": self.q_net.state_dict(),
            "target_net_state": self.target_q_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_dec": self.epsilon_dec,
        }
        torch.save(checkpoint, path)
        logging.info(f"[DQNAgent] Checkpoint saved to {path}")

    def load(self, path: str) -> None:
        """
        체크포인트에서 모델 상태를 복원합니다.

        Args:
            path (str): 체크포인트 파일 경로.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net_state"])
        self.target_q_net.load_state_dict(checkpoint["target_net_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.step_count = checkpoint.get("step_count", 0)
        self.env_step_count = checkpoint.get("env_step_count", 0)
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.epsilon_min = checkpoint.get("epsilon_min", self.epsilon_min)
        self.epsilon_dec = checkpoint.get("epsilon_dec", self.epsilon_dec)
        self.q_net.train()
        self.target_q_net.eval()
        logging.info(f"[DQNAgent] Checkpoint loaded from {path}")
