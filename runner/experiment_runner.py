import yaml
import random
import numpy as np
import torch
import logging
import os
from typing import List, Dict, Any, Tuple
from datetime import datetime

from components.registry import make
from components.rec_context import RecContextManager, get_recommendation_quota
from components.rec_utils import enforce_type_constraint, compute_all_q_values


class ExperimentRunner:
    """
    실험 전체 루프를 실행하는 클래스.
    YAML config를 읽어 각 실험을 seed별로 반복 수행.
    """

    def __init__(self, config_path: str = "config/experiment.yaml") -> None:
        """
        ExperimentRunner 객체를 초기화합니다.

        Args:
            config_path (str): 실험 설정 파일 경로 (YAML)
        """
        self.cfg: Dict[str, Any] = yaml.safe_load(open(config_path))
        # self.cfg = self._load_config(config_path)
        self.result_log_path = self.cfg.get("experiment", {}).get(
            "result_log_path", "experiment_results.log"
        )
        logging.info("ExperimentRunner initialized.")

    # 추후 추가 예정
    # def _load_config(self, config_path: str) -> Dict[str, Any]:
    #     try:
    #         with open(config_path, "r") as f:
    #             cfg = yaml.safe_load(f)
    #         if not isinstance(cfg, dict):
    #             raise ValueError("Config file does not contain a valid dict.")
    #         return cfg
    #     except FileNotFoundError:
    #         logging.error(f"Config file not found: {config_path}")
    #         raise
    #     except yaml.YAMLError as e:
    #         logging.error(f"YAML parsing error: {e}")
    #         raise

    def set_seed(self, seed: int) -> None:
        """
        전체 환경의 랜덤 시드를 고정합니다.

        Args:
            seed (int): 사용할 시드 값
        """
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logging.info(f"Random seed set: {seed}")

    def run_single(self, seed: int) -> None:
        self.set_seed(seed)
        cfg: Dict[str, Any] = self.cfg

        # Embedder / CandidateGenerator / Reward / Context 초기화
        embedder = make(cfg["embedder"]["type"], **cfg["embedder"]["params"])
        candgen = make(
            cfg["candidate_generator"]["type"], **cfg["candidate_generator"]["params"]
        )
        reward_fn = make(cfg["reward_fn"]["type"], **cfg["reward_fn"]["params"])
        context = RecContextManager(cfg["env"]["params"]["cold_start"])

        # LLM 시뮬레이터 생성
        from components.llm_simu import LLMUserSimulator
        llm_simulator = LLMUserSimulator(**cfg["llm_simulator"]["params"])
        
        # 환경 생성 (LLM 시뮬레이터 포함)
        env = make(
            cfg["env"]["type"],
            **cfg["env"]["params"],
            embedder=embedder,
            candidate_generator=candgen,
            reward_fn=reward_fn,
            context=context,
            llm_simulator=llm_simulator,  # LLM 시뮬레이터 전달
        )
        
        # 에이전트 생성
        agent = make(
            cfg["agent"]["type"],
            user_dim=embedder.user_dim,
            content_dim=embedder.content_dim,
            capacity=cfg["replay"]["capacity"],
            **cfg["agent"]["params"],
        )

        total_eps: int = cfg["experiment"]["total_episodes"]
        max_recs: int = cfg["experiment"]["max_recommendations"]

        episode_metrics = []

        episode_metrics = []

        for ep in range(total_eps):
            try:
                logging.info(f"\n--- Episode {ep+1}/{total_eps} (seed={seed}) ---")
                # 기존처럼 쿼리 하드코딩 (나중에 변경 가능)
                query: str = "삼성전자"

                state, _ = env.reset(options={"query": query})
                done: bool = False
                total_reward = 0.0
                rec_count = 0
                emb_cache: Dict[Any, Any] = {}

                while not done:
                    cand_dict: Dict[str, List[Any]] = env.get_candidates()
                    for ctype, cands in cand_dict.items():
                        for cand in cands:
                            cid = getattr(cand, "id", id(cand))
                            if cid not in emb_cache:
                                emb_cache[cid] = embedder.embed_content(cand)

                    q_values: Dict[str, List[float]] = compute_all_q_values(
                        state, cand_dict, embedder, agent, emb_cache=emb_cache
                    )

                    enforce_list: List[Tuple[str, int]] = enforce_type_constraint(
                        q_values, top_k=max_recs
                    )

                    for ctype, idx in enforce_list:
                        cands = cand_dict.get(ctype, [])
                        if idx < 0 or idx >= len(cands):
                            logging.warning(
                                f"Invalid candidate index {idx} for type '{ctype}'. Skipping."
                            )
                            continue

                        selected = cands[idx]
                        cid = getattr(selected, "id", id(selected))
                        selected_emb = emb_cache[cid]

                        step_result = env.step((ctype, idx))
                        if (
                            not isinstance(step_result, (tuple, list))
                            or len(step_result) != 5
                        ):
                            raise ValueError(
                                "env.step must return (next_state, reward, done, truncated, info)"
                            )

                        next_state, r, step_done, truncated, info = step_result
                        done = step_done or truncated
                        total_reward += r
                        rec_count += 1

                        logging.info(
                            f"    Recommended: (type={ctype}, idx={idx}) → reward={r}, done={done}"
                        )

                        next_cand_dict: Dict[str, List[Any]] = env.get_candidates()
                        next_cembs: Dict[str, List[Any]] = {
                            t: [
                                emb_cache.get(
                                    getattr(c, "id", id(c)), embedder.embed_content(c)
                                )
                                for c in cs
                            ]
                            for t, cs in next_cand_dict.items()
                        }
                        agent.store(
                            state, selected_emb, r, next_state, next_cembs, done
                        )
                        agent.learn()

                        state = next_state

                        if done:
                            break

                logging.info(
                    f"--- Episode {ep+1} End. Agent Epsilon: {getattr(agent, 'epsilon', float('nan')):.3f} ---"
                )
                episode_metrics.append(
                    {
                        "seed": seed,
                        "episode": ep + 1,
                        "query": query,
                        "total_reward": total_reward,
                        "recommendations": rec_count,
                        "datetime": datetime.now().isoformat(),
                    }
                )
            except Exception as e:
                logging.error(
                    f"Error in episode {ep+1}, seed {seed}: {e}", exc_info=True
                )

        self.save_results(episode_metrics)

    def save_results(self, metrics: List[Dict[str, Any]]):
        import csv

        csv_path = self.result_log_path.replace(".log", ".csv")
        fieldnames = metrics[0].keys() if metrics else []
        if not fieldnames:
            return
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in metrics:
                writer.writerow(row)

    def run_all(self) -> None:
        """
        config에 정의된 모든 seed에 대해 실험을 실행합니다.
        """
        seeds: List[int] = self.cfg["experiment"].get("seeds", [0])
        for s in seeds:
            try:
                self.run_single(s)
            except Exception as e:
                logging.error(f"Seed {s} experiment failed: {e}", exc_info=True)
                continue


if __name__ == "__main__":
    """
    main 엔트리포인트. 실험을 실행합니다.
    """
    ExperimentRunner().run_all()
