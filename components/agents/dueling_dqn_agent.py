import random
from itertools import chain
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import logging

from components.core.base import BaseAgent
from components.registry import register
from models.q_network import DuelingQNetwork
from replay.replay_buffer import ReplayBuffer


<<<<<<< HEAD
@register("dqn")
class DQNAgent(BaseAgent):
    """DQN 기반 추천 에이전트.

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
        """DQNAgent 생성자.

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
            loss_type (str, optional): 손실 함수 종류 ('mse' 또는 'smooth_l1'). 기본값 'smooth_l1'.
            device (str, optional): 사용할 디바이스 ('cpu' 또는 'cuda'). 기본값 'cpu'.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else device)
        self.q_net = QNetwork(user_dim, content_dim).to(self.device)
        self.target_q_net = QNetwork(user_dim, content_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=float(lr))
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
        """현재 상태에서 추천 콘텐츠(액션)를 선택합니다.

        Args:
            user_state (List[float]): 현재 사용자 상태 임베딩 벡터.
            candidate_embs (List[List[float]]): 후보 콘텐츠 임베딩 리스트.

        Returns:
            int: 선택한 콘텐츠 인덱스.
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
        """경험을 리플레이 버퍼에 저장합니다.

        Args:
            user_state (List[float]): 현재 상태 임베딩.
            content_emb (List[float]): 액션에 해당하는 콘텐츠 임베딩.
            reward (float): 보상.
            next_state (List[float]): 다음 상태 임베딩.
            next_cands_embs (Dict[str, List[List[float]]]): 다음 상태에서의 후보군 임베딩 (타입별).
            done (bool): 에피소드 종료 여부.
        """
        self.buffer.push(
            (user_state, content_emb), reward, (next_state, next_cands_embs), done
        )

    def learn(self) -> float:
        """리플레이 버퍼에서 샘플을 추출해 Q 네트워크를 한 번 업데이트합니다.

        Returns:
            float: 미니배치의 손실 값(loss). (버퍼 데이터가 부족한 경우 None 반환)
        """
        # 1. 충분한 경험이 쌓이지 않았다면 학습하지 않음
        if len(self.buffer) < self.batch_size:
            return

        self.step_count += 1

        # 2. 미니배치 샘플 추출
        user_states, content_embs, rewards, next_info, dones = self.buffer.sample(
            self.batch_size
        )
        next_states, next_cands_embs = next_info

        # 3. 텐서 변환
        us = torch.FloatTensor(np.array(user_states)).to(self.device)
        ce = torch.FloatTensor(np.array(content_embs)).to(self.device)
        rs = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        ds = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 4. Q(s, a) 계산
        q_sa = self.q_net(us, ce)

        # 5. 다음 상태에서의 최대 Q값을 벡터화로 계산
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
            flat_states_tensor = torch.FloatTensor(np.array(flat_states)).to(
                self.device
            )
            flat_cands_tensor = torch.FloatTensor(np.array(flat_cands)).to(self.device)
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

        # 6. 타겟 계산
        target = rs + self.gamma * max_nq * (1 - ds)

        # 7. 손실 계산
        if self.loss_type == "mse":
            loss = F.mse_loss(q_sa, target)
        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(q_sa, target)
        else:
            raise ValueError(f"지원하지 않는 loss_type입니다: {self.loss_type}")

        # 8. 역전파 및 파라미터 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 9. 일정 주기로 타겟 네트워크 동기화
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
        Epsilon-greedy를 통해 슬레이트 추천을 수행합니다.

        Args:
            state: 현재 상태 벡터 (List[float]).
            candidate_embs: 콘텐츠 타입별 후보 임베딩, ex: {'news': [[...], ...], 'video': [[...], ...]}
            max_recs: 추천할 전체 아이템 개수

        Returns:
            추천 슬레이트: List of (content_type, candidate_index)
        """
        all_candidates = []
        for ctype, embs in candidate_embs.items():
            for idx, _ in enumerate(embs):
                all_candidates.append((ctype, idx))

        if not all_candidates:
            return []

        # 탐험(Exploration)
        if random.random() < self.epsilon:
            sample_count = min(max_recs, len(all_candidates))
            return random.sample(all_candidates, sample_count)

        # 활용(Exploitation): 각 타입별로 Q값 계산
        q_values_with_pos = []
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        for ctype, embs in candidate_embs.items():
            if not embs:
                continue

            cand_tensor = torch.tensor(embs, dtype=torch.float32, device=self.device)
            state_rep = state_tensor.repeat(len(embs), 1)

            with torch.no_grad():
                q_vals = self.q_net(state_rep, cand_tensor).squeeze(1)

            for i, q_val in enumerate(q_vals):
                q_values_with_pos.append(((ctype, i), q_val.item()))

        # Q값 기준으로 정렬하여 상위 max_recs개 선택
        q_values_with_pos.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [item[0] for item in q_values_with_pos[:max_recs]]
        return top_candidates

    def decay_epsilon(self):
        """탐험률(epsilon)을 감소시킵니다."""
        self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)

    def save(self, path: str) -> None:
        """에이전트의 상태를 파일로 저장합니다.

        Args:
            path (str): 체크포인트 파일 경로.
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
        """저장된 체크포인트 파일에서 에이전트의 상태를 복원합니다.

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


=======
>>>>>>> 2f03202 (feat: Dueling DQN 에이전트 추가)
@register("dueling_dqn")
class DuelingDQNAgent(BaseAgent):
    """Dueling DQN 기반 추천 에이전트."""

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else device)
        self.q_net = DuelingQNetwork(user_dim, content_dim).to(self.device)
        self.target_q_net = DuelingQNetwork(user_dim, content_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=float(lr))
        self.buffer = ReplayBuffer(capacity=capacity)

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = eps_start
        self.epsilon_min = eps_min
        self.epsilon_dec = eps_decay
        self.update_freq = update_freq
        self.step_count = 0
        self.loss_type = loss_type

    # 나머지 메서드는 DQNAgent와 동일하게 복사/사용

    def select_action(
        self, user_state: List[float], candidate_embs: List[List[float]]
    ) -> int:
        """현재 상태에서 추천 콘텐츠(액션)를 선택합니다.

        Args:
            user_state (List[float]): 현재 사용자 상태 임베딩 벡터.
            candidate_embs (List[List[float]]): 후보 콘텐츠 임베딩 리스트.

        Returns:
            int: 선택한 콘텐츠 인덱스.
        """
        if random.random() < self.epsilon:
            return random.randrange(len(candidate_embs))
        us = torch.FloatTensor(user_state).unsqueeze(0).to(self.device)
        us_rep = us.repeat(len(candidate_embs), 1)
        ce = torch.FloatTensor(candidate_embs).to(self.device)
        with torch.no_grad():
            q_vals = self.q_net(us_rep, ce).squeeze(1)
        return int(torch.argmax(q_vals).item())

<<<<<<< HEAD
=======
    def select_slate(
        self,
        state: List[float],
        candidate_embs: Dict[str, List[List[float]]],
        max_recs: int,
    ) -> List[Tuple[str, int]]:
        """
        Epsilon-greedy를 통해 슬레이트 추천을 수행합니다.

        Args:
            state: 현재 상태 벡터 (List[float]).
            candidate_embs: 콘텐츠 타입별 후보 임베딩, ex: {'news': [[...], ...], 'video': [[...], ...]}
            max_recs: 추천할 전체 아이템 개수

        Returns:
            추천 슬레이트: List of (content_type, candidate_index)
        """
        all_candidates = []
        for ctype, embs in candidate_embs.items():
            for idx, _ in enumerate(embs):
                all_candidates.append((ctype, idx))

        if not all_candidates:
            return []

        # 탐험(Exploration)
        if random.random() < self.epsilon:
            sample_count = min(max_recs, len(all_candidates))
            return random.sample(all_candidates, sample_count)

        # 활용(Exploitation): 각 타입별로 Q값 계산
        q_values_with_pos = []
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # 평가 모드 설정
        self.q_net.eval()

        for ctype, embs in candidate_embs.items():
            if not embs:
                continue

            cand_tensor = torch.tensor(embs, dtype=torch.float32, device=self.device)
            state_rep = state_tensor.repeat(len(embs), 1)

            with torch.no_grad():
                q_vals = self.q_net(state_rep, cand_tensor).squeeze(1)

            for i, q_val in enumerate(q_vals):
                q_values_with_pos.append(((ctype, i), q_val.item()))

        # 다시 학습 모드로 전환
        self.q_net.train()

        # Q값 기준으로 정렬하여 상위 max_recs개 선택
        q_values_with_pos.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [item[0] for item in q_values_with_pos[:max_recs]]
        return top_candidates

>>>>>>> 2f03202 (feat: Dueling DQN 에이전트 추가)
    def store(
        self,
        user_state: List[float],
        content_emb: List[float],
        reward: float,
        next_state: List[float],
        next_cands_embs: Dict[str, List[List[float]]],
        done: bool,
    ) -> None:
        """경험을 리플레이 버퍼에 저장합니다.

        Args:
            user_state (List[float]): 현재 상태 임베딩.
            content_emb (List[float]): 액션에 해당하는 콘텐츠 임베딩.
            reward (float): 보상.
            next_state (List[float]): 다음 상태 임베딩.
            next_cands_embs (Dict[str, List[List[float]]]): 다음 상태에서의 후보군 임베딩 (타입별).
            done (bool): 에피소드 종료 여부.
        """
        self.buffer.push(
            (user_state, content_emb), reward, (next_state, next_cands_embs), done
        )

    def learn(self) -> float:
        """리플레이 버퍼에서 샘플을 추출해 Q 네트워크를 업데이트합니다."""
        if len(self.buffer) < self.batch_size:
            return

        self.step_count += 1

        # 2. 미니배치 샘플 추출
        user_states, content_embs, rewards, next_info, dones = self.buffer.sample(
            self.batch_size
        )
        next_states, next_cands_embs = next_info

        # 3. 텐서 변환 (dtype과 device를 한 번에 지정)
        us = torch.tensor(user_states, dtype=torch.float32, device=self.device)
        ce = torch.tensor(content_embs, dtype=torch.float32, device=self.device)
        rs = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        ds = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # 4. Q(s, a) 계산 (q_net은 train 모드)
        q_sa = self.q_net(us, ce)

        # 5. 다음 상태에서의 최대 Q값을 벡터화로 계산 (target_q_net은 eval 모드)
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

        # 6. 타겟 계산
        target = rs + self.gamma * max_nq * (1 - ds)

        # 7. 손실 계산
        if self.loss_type == "mse":
            loss = F.mse_loss(q_sa, target)
        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(q_sa, target)
        else:
            raise ValueError(f"지원하지 않는 loss_type입니다: {self.loss_type}")

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        ## [Gradient clipping] ##
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 9. 일정 주기로 타겟 네트워크 동기화
        if self.step_count % self.update_freq == 0:
            logging.info(
                f"Step {self.step_count}: Loss = {loss.item()}, Epsilon = {self.epsilon:.4f}"
            )
            self.target_q_net.load_state_dict(self.q_net.state_dict())
<<<<<<< HEAD
        return loss.item()

        return loss.item()
=======
>>>>>>> 2f03202 (feat: Dueling DQN 에이전트 추가)

    def decay_epsilon(self):
        """탐험률(epsilon)을 감소시킵니다."""
        self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)

    def save(self, path: str) -> None:
        """에이전트의 상태를 파일로 저장합니다.

        Args:
            path (str): 체크포인트 파일 경로.
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
        logging.info(f"[DuelingQNAgent] Checkpoint saved to {path}")

    def load(self, path: str) -> None:
        """저장된 체크포인트 파일에서 에이전트의 상태를 복원합니다.

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
        logging.info(f"[DuelingQNAgent] Checkpoint loaded from {path}")
