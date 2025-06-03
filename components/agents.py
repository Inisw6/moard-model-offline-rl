import random
from typing import List, Dict
import torch
import torch.nn.functional as F
from components.base import BaseAgent
from components.registry import register
from models.q_network import QNetwork
from replay.replay_buffer import ReplayBuffer
from itertools import chain


@register("dqn")
class DQNAgent(BaseAgent):
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
        device: str = "cpu",
    ) -> None:
        """
        DQN 기반 추천 에이전트.

        Args:
            user_dim (int): 사용자 상태 임베딩 차원
            content_dim (int): 콘텐츠 임베딩 차원
            lr (float): 학습률
            batch_size (int): 배치 크기
            eps_start (float): 초기 탐험률
            eps_min (float): 최소 탐험률
            eps_decay (float): 탐험률 감소 계수
            gamma (float): 감가율
            update_freq (int): 타겟 네트워크 업데이트 주기
            capacity (int): 리플레이 버퍼 용량
            device (str): 'cpu' or 'cuda'
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else device)
        self.q_net = QNetwork(user_dim, content_dim).to(self.device)
        self.target_q_net = QNetwork(user_dim, content_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=capacity)

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = eps_start
        self.epsilon_min = eps_min
        self.epsilon_dec = eps_decay
        self.update_freq = update_freq
        self.step_count = 0

    def select_action(
        self, user_state: List[float], candidate_embs: List[List[float]]
    ) -> int:
        """
        액션(=추천 콘텐츠 인덱스) 선택.

        Args:
            user_state (List[float]): 현재 사용자 상태 임베딩 (벡터)
            candidate_embs (List[List[float]]): 후보 콘텐츠 임베딩 리스트

        Returns:
            int: 선택한 콘텐츠 인덱스
        """
        self.step_count += 1
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
        샘플을 리플레이 버퍼에 저장.

        Args:
            user_state (List[float]): 현재 상태 임베딩
            content_emb (List[float]): 액션에 해당하는 콘텐츠 임베딩
            reward (float): 보상
            next_state (List[float]): 다음 상태 임베딩
            next_cands_embs (Dict[str, List[List[float]]]): 다음 상태에서의 후보군 임베딩 (타입별)
            done (bool): 에피소드 종료 여부
        """
        self.buffer.push(
            (user_state, content_emb), reward, (next_state, next_cands_embs), done
        )

    def learn(self) -> None:
        """리플레이 버퍼에서 샘플을 추출해 Q 네트워크를 업데이트합니다."""
        if len(self.buffer) < self.batch_size:
            return
        user_states, content_embs, rewards, next_info, dones = self.buffer.sample(
            self.batch_size
        )
        next_states, next_cands_embs = next_info

        us = torch.FloatTensor(user_states).to(self.device)
        ce = torch.FloatTensor(content_embs).to(self.device)
        rs = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        ds = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_sa = self.q_net(us, ce)

        max_next_q_list: List[torch.Tensor] = []
        # for-loop(각 샘플별)로 target_q_net 평가 (추후 배치 평가로 최적화 가능)
        for ns, nxt in zip(next_states, next_cands_embs):
            # nxt: Dict[str, List[List[float]]] - 모든 타입 후보 벡터 합치기
            all_embs = list(chain.from_iterable(nxt.values()))
            if not all_embs:
                # 후보가 없으면 0 보상
                max_next_q_list.append(torch.tensor(0.0, device=self.device))
                continue
            usn = torch.FloatTensor(ns).unsqueeze(0).to(self.device)
            usn_rep = usn.repeat(len(all_embs), 1)
            cen = torch.FloatTensor(all_embs).to(self.device)
            with torch.no_grad():
                qn = self.target_q_net(usn_rep, cen).squeeze(1)
            max_next_q_list.append(qn.max())
        max_nq = torch.stack(max_next_q_list).unsqueeze(1)

        target = rs + self.gamma * max_nq * (1 - ds)
        loss = F.mse_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)
        if self.step_count % self.update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str) -> None:
        """Q 네트워크 파라미터를 파일로 저장."""
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str) -> None:
        """Q 네트워크 파라미터를 파일에서 불러오기."""
        self.q_net.load_state_dict(torch.load(path))
        self.q_net.eval()
