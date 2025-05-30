import gymnasium as gym
from gymnasium import spaces
import numpy as np
from components.base import BaseEnv
from components.registry import register

@register("rec_env")
class RecEnv(gym.Env, BaseEnv):
    def __init__(self, cold_start, max_steps, top_k,
                 embedder, candidate_generator, reward_fn, context):
        super().__init__()
        self.context             = context
        self.max_steps           = max_steps
        self.top_k               = top_k
        self.embedder            = embedder
        self.candidate_generator = candidate_generator
        self.reward_fn           = reward_fn
        self.step_count          = 0

        state_dim = embedder.output_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.context.reset()
        user = self._generate_fake_user()
        state = self.embedder.embed_user(user)
        return state, {}

    def step(self, action):
        ctype, idx = action
        self.step_count += 1

        cand_dict = self.candidate_generator.get_candidates(None)
        cands = cand_dict[ctype]
        selected = cands[idx]
        reward = self.reward_fn.calculate(selected)

        user_next = self._generate_fake_user()
        next_state = self.embedder.embed_user(user_next)

        done = self.step_count >= self.max_steps
        self.context.step()
        return next_state, reward, done, {}

    def get_candidates(self, state):
        return self.candidate_generator.get_candidates(state)

    def _generate_fake_user(self):
        from datetime import datetime
        return {
            "recent_logs": [
                {"category": ["finance"], "emotion": 0.8, "dwell": 25, "type": "youtube"},
                {"category": ["tech"],    "emotion": 0.6, "dwell": 15, "type": "blog"},
            ],
            "current_time": datetime.now()
        }