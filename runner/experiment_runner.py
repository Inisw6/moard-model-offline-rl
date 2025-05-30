import yaml
import random
import numpy as np
import torch
from components.registry import make
from components.rec_context import RecContextManager, get_recommendation_quota

class ExperimentRunner:
    def __init__(self, config_path="config/experiment.yaml"):
        self.cfg = yaml.safe_load(open(config_path))

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run_single(self, seed):
        self.set_seed(seed)
        cfg = self.cfg

        # instantiate components
        embedder = make(cfg["embedder"]["type"], **cfg["embedder"]["params"])
        candgen = make(cfg["candidate_generator"]["type"], **cfg["candidate_generator"]["params"])
        reward_fn = make(cfg["reward_fn"]["type"], **cfg["reward_fn"]["params"])
        context = RecContextManager(cfg["env"]["params"]["cold_start"])

        # environment and agent
        env = make(
            cfg["env"]["type"],
            **cfg["env"]["params"],
            embedder=embedder,
            candidate_generator=candgen,
            reward_fn=reward_fn,
            context=context
        )
        agent = make(
            cfg["agent"]["type"],
            user_dim=embedder.user_dim,
            content_dim=embedder.content_dim,
            capacity=cfg["replay"]["capacity"],
            **cfg["agent"]["params"]
        )

        total_eps = cfg["experiment"]["total_episodes"]
        max_recs = cfg["experiment"]["max_recommendations"]

        for ep in range(total_eps):
            state, _ = env.reset()
            done = False
            while not done:
                cand_dict = env.get_candidates(state)
                user_pref = embedder.estimate_preference(state)
                quota = get_recommendation_quota(user_pref, context, max_total=max_recs)

                for ctype, cnt in quota.items():
                    cands = cand_dict[ctype]
                    cembs = [embedder.embed_content(c) for c in cands]
                    for _ in range(cnt):
                        idx = agent.select_action(state, cembs)
                        next_state, r, done, _ = env.step((ctype, idx))
                        next_cand_dict = env.get_candidates(next_state)
                        next_cembs = {t:[embedder.embed_content(c) for c in cs]
                                      for t, cs in next_cand_dict.items()}
                        agent.store(state, cembs[idx], r, next_state, next_cembs, done)
                        agent.learn()
                        state = next_state
                        if done:
                            break
                    if done:
                        break
            print(f"Episode {ep+1}/{total_eps} eps={agent.epsilon:.3f}")

    def run_all(self):
        seeds = self.cfg["experiment"].get("seeds", [0])
        for s in seeds:
            self.run_single(s)

if __name__ == "__main__":
    ExperimentRunner().run_all()