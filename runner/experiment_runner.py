import yaml
import random
import numpy as np
import torch
import logging

from typing import List, Dict, Any
from components.registry import make
from components.rec_context import RecContextManager, get_recommendation_quota

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ExperimentRunner:
    def __init__(self, config_path: str = "config/experiment.yaml") -> None:
        self.cfg: Dict[str, Any] = yaml.safe_load(open(config_path))

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run_single(self, seed: int) -> None:
        self.set_seed(seed)
        cfg: Dict[str, Any] = self.cfg

        embedder = make(cfg["embedder"]["type"], **cfg["embedder"]["params"])
        candgen = make(
            cfg["candidate_generator"]["type"], **cfg["candidate_generator"]["params"]
        )
        reward_fn = make(cfg["reward_fn"]["type"], **cfg["reward_fn"]["params"])
        context = RecContextManager(cfg["env"]["params"]["cold_start"])

        env = make(
            cfg["env"]["type"],
            **cfg["env"]["params"],
            embedder=embedder,
            candidate_generator=candgen,
            reward_fn=reward_fn,
            context=context,
        )
        agent = make(
            cfg["agent"]["type"],
            user_dim=embedder.user_dim,
            content_dim=embedder.content_dim,
            capacity=cfg["replay"]["capacity"],
            **cfg["agent"]["params"],
        )

        total_eps: int = cfg["experiment"]["total_episodes"]
        max_recs: int = cfg["experiment"]["max_recommendations"]

        for ep in range(total_eps):
            print(f"\n--- Episode {ep+1}/{total_eps} ---")
            state, _ = env.reset()
            done: bool = False

            while not done:
                cand_dict: Dict[str, List[Any]] = env.get_candidates(state)
                user_pref: Dict[str, float] = embedder.estimate_preference(state)
                quota: Dict[str, int] = get_recommendation_quota(
                    user_pref, context, max_total=max_recs
                )

                logging.info(
                    f"  Loop Start: ep={ep + 1}, env_step={env.step_count}, done={done}"
                )
                logging.info(f"    State (first 5): {state[:5]}")
                logging.info(f"    User Pref: {user_pref}")
                logging.info(f"    Quota: {quota}")

                if not quota or all(v == 0 for v in quota.values()):
                    logging.warning(
                        "    Quota is empty or all zeros. Breaking inner loop to prevent infinite loop."
                    )
                    done = True
                    continue

                for ctype, cnt in quota.items():
                    logging.info(f"    Content Type Loop: ctype={ctype}, cnt={cnt}")
                    if cnt == 0:
                        continue

                    cands = cand_dict.get(ctype, [])
                    if not cands:
                        logging.warning(
                            f"    No candidates found for ctype {ctype}. Skipping."
                        )
                        continue

                    cembs = [embedder.embed_content(c) for c in cands]
                    if not cembs:
                        logging.warning(
                            f"    No valid candidate embeddings for ctype {ctype}. Skipping."
                        )
                        continue

                    # todo: 상태 갱신 후 새로운 후보 생성
                    for i_rec in range(cnt):
                        logging.info(
                            f"      Recommendation Loop: rec_num={i_rec + 1}/{cnt} for {ctype}, env_step={env.step_count}"
                        )
                        idx: int = agent.select_action(state, cembs)
                        if idx >= len(cembs):
                            logging.error(
                                f"    agent.select_action returned invalid index {idx} for {len(cembs)} candidates. Skipping recommendation."
                            )
                            continue

                        selected_content_emb = cembs[idx]
                        next_state, r, step_done, truncated, info = env.step(
                            (ctype, idx)
                        )
                        done = step_done or truncated

                        logging.info(
                            f"        env.step returned: reward={r}, step_done={step_done}, truncated={truncated}, final_done={done}"
                        )
                        logging.info(f"        Next state (first 5): {next_state[:5]}")

                        next_cand_dict: Dict[str, List[Any]] = env.get_candidates(
                            next_state
                        )
                        next_cembs: Dict[str, List[Any]] = {
                            t: [embedder.embed_content(c) for c in cs]
                            for t, cs in next_cand_dict.items()
                        }
                        agent.store(
                            state, selected_content_emb, r, next_state, next_cembs, done
                        )
                        agent.learn()
                        state = next_state

                        if done:
                            logging.info(
                                f"        Episode finished at rec_num={i_rec + 1} for {ctype} due to done/truncated flag."
                            )
                            break
                    if done:
                        break
            logging.info(
                f"--- Episode {ep + 1} End. Agent Epsilon: {agent.epsilon:.3f} ---"
            )

    def run_all(self) -> None:
        seeds: List[int] = self.cfg["experiment"].get("seeds", [0])
        for s in seeds:
            self.run_single(s)


if __name__ == "__main__":
    ExperimentRunner().run_all()
