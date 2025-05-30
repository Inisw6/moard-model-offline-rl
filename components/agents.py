import random
import numpy as np
import torch
import torch.nn.functional as F
from components.base import BaseAgent
from components.registry import register
from models.q_network import QNetwork
from replay.replay_buffer import ReplayBuffer

@register("dqn")
class DQNAgent(BaseAgent):
    def __init__(self, user_dim, content_dim,
                 lr, batch_size, eps_start, eps_min, eps_decay,
                 gamma, update_freq, capacity, device='cpu'):
        self.device      = torch.device("cuda" if torch.cuda.is_available() else device)
        self.q_net       = QNetwork(user_dim, content_dim).to(self.device)
        self.target_q_net= QNetwork(user_dim, content_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer   = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer      = ReplayBuffer(capacity=capacity)

        self.gamma       = gamma
        self.batch_size  = batch_size
        self.epsilon     = eps_start
        self.epsilon_min = eps_min
        self.epsilon_dec = eps_decay
        self.update_freq = update_freq
        self.step_count  = 0

    def select_action(self, user_state, candidate_embs):
        self.step_count += 1
        if random.random() < self.epsilon:
            return random.randrange(len(candidate_embs))
        us = torch.FloatTensor(user_state).unsqueeze(0).to(self.device)
        us_rep = us.repeat(len(candidate_embs), 1)
        ce = torch.FloatTensor(candidate_embs).to(self.device)
        with torch.no_grad():
            q_vals = self.q_net(us_rep, ce).squeeze(1)
        return int(torch.argmax(q_vals).item())

    def store(self, user_state, content_emb, reward, next_state, next_cands_embs, done):
        self.buffer.push((user_state, content_emb), reward, (next_state, next_cands_embs), done)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return
        user_states, content_embs, rewards, next_info, dones = self.buffer.sample(self.batch_size)
        next_states, next_cands_embs = next_info

        us = torch.FloatTensor(user_states).to(self.device)
        ce = torch.FloatTensor(content_embs).to(self.device)
        rs = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        ds = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_sa = self.q_net(us, ce)

        max_next_q_list = []
        for ns, nxt in zip(next_states, next_cands_embs):
            all_embs = sum(nxt.values(), [])
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
    
    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(path))
        self.q_net.eval()