import random

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer   = []

    def push(self, state_cont_pair, reward, next_info, done):
        # state_cont_pair: (user_state, content_emb)
        # next_info: (next_state, next_cands_embs)
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state_cont_pair, reward, next_info, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        # unpack: ((s,ce), r, (s_next, next_cands_embs), done)
        sc, r, ni, d = zip(*batch)
        s, ce = zip(*sc)
        ns, next_embs = zip(*ni)
        return s, ce, r, (ns, next_embs), d

    def __len__(self):
        return len(self.buffer)
