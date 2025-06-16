import os
import csv
import random
import logging
import yaml
import torch
import numpy as np
import pandas as pd

from datetime import datetime
from typing import Any, Dict, List

from components.registry import make
from components.database.db_utils import (
    get_stock_names,
    get_contents,
    get_user_logs,
    get_users,
)
from components.recommendation.rec_utils import compute_all_q_values
from components.simulation.random_simulator import RandomResponseSimulator
from components.simulation.llm_simulator import LLMResponseSimulator
from components.simulation.llm_simu import LLMUserSimulator


class ExperimentRunner:
    """YAML 설정 파일을 기반으로 RL 추천 실험을 반복 실행합니다.

    Attributes:
        cfg (Dict[str, Any]): 전체 실험 설정 딕셔너리.
    """

    def __init__(self, config_path: str = "config/experiment.yaml") -> None:
        """ExperimentRunner 인스턴스를 초기화합니다.

        Args:
            config_path (str): YAML 설정 파일 경로.
        """
        with open(config_path, "r") as f:
            self.cfg: Dict[str, Any] = yaml.safe_load(f)
        logging.info("ExperimentRunner initialized with config: %s", config_path)

    def set_seed(self, seed: int) -> None:
        """모든 난수 생성기의 시드를 고정하여 재현성을 보장합니다.

        Args:
            seed (int): 사용할 랜덤 시드 값.
        """
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logging.info("Random seed set to %d", seed)

    def run_single(self, seed: int) -> None:
        """단일 시드에 대해 전체 에피소드를 실행하고 결과를 저장합니다.

        Args:
            seed (int): 실험 시드.

        Raises:
            ValueError: 지원하지 않는 시뮬레이터 타입 또는 env.step 반환값 오류 시.
        """
        self.set_seed(seed)
        exp_cfg = self.cfg["experiment"]
        exp_name = exp_cfg.get("experiment_name", "default_exp")

        # 로그/모델 저장 경로 준비
        step_log = exp_cfg["step_log_path"].format(experiment_name=exp_name, seed=seed)
        eps_log = exp_cfg["episode_log_path"].format(
            experiment_name=exp_name, seed=seed
        )
        model_dir = exp_cfg["model_save_dir"].format(
            experiment_name=exp_name, seed=seed
        )
        os.makedirs(os.path.dirname(step_log), exist_ok=True)
        os.makedirs(os.path.dirname(eps_log), exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # 데이터 로드 및 전처리
        contents_df = get_contents()
        users_df = get_users()
        logs_df = get_user_logs()
        if not logs_df.empty and not contents_df.empty:
            logs_with_type = pd.merge(
                logs_df,
                contents_df[["id", "type"]].rename(
                    columns={"id": "content_id", "type": "content_db_type"}
                ),
                on="content_id",
                how="left",
            )
        else:
            logs_with_type = logs_df

        # 컴포넌트 생성
        embedder = make(**self.cfg["embedder"])
        candgen = make(**self.cfg["candidate_generator"], contents_df=contents_df)
        reward_fn = make(**self.cfg["reward_fn"])

        sim_cfg = self.cfg["response_simulator"]
        if sim_cfg["type"] == "random":
            response_sim = RandomResponseSimulator(**sim_cfg.get("params", {}))
        elif sim_cfg["type"] == "llm":
            llm_params = sim_cfg["params"].pop("llm_simulator", {}).get("params", {})
            llm_client = LLMUserSimulator(**llm_params)
            response_sim = LLMResponseSimulator(
                llm_simulator=llm_client, **sim_cfg.get("params", {})
            )
        else:
            raise ValueError(f"Unsupported simulator type: {sim_cfg['type']}")

        env = make(
            **self.cfg["env"],
            embedder=embedder,
            candidate_generator=candgen,
            reward_fn=reward_fn,
            response_simulator=response_sim,
            contents_df=contents_df,
            users_df=users_df,
            logs_with_type_df=logs_with_type,
        )

        agent = make(
            **self.cfg["agent"],
            user_dim=embedder.user_dim,
            content_dim=embedder.content_dim,
            capacity=self.cfg["replay"]["capacity"],
        )

        # 실험 설정
        total_eps = exp_cfg["total_episodes"]
        max_recs = exp_cfg["max_recommendations"]
        stock_names = get_stock_names()[
            : exp_cfg.get("max_stocks", len(get_stock_names()))
        ]
        if not stock_names:
            raise ValueError("No stock names available for queries")
        logging.info("Using %d stocks: %s", len(stock_names), stock_names)

        # 에피소드별 쿼리 리스트 생성
        queries: List[str] = []
        cycles, rem = divmod(total_eps, len(stock_names))
        for _ in range(cycles):
            random.shuffle(stock_names)
            queries.extend(stock_names)
        random.shuffle(stock_names)
        queries.extend(stock_names[:rem])

        step_metrics: List[Dict[str, Any]] = []
        episode_metrics: List[Dict[str, Any]] = []

        # 에피소드 루프
        for ep, query in enumerate(queries, start=1):
            try:
                logging.info(
                    "Episode %d/%d start (seed=%d, query=%s)",
                    ep,
                    total_eps,
                    seed,
                    query,
                )
                state, _ = env.reset(options={"query": query})
                done = False
                emb_cache: Dict[Any, Any] = {}
                total_reward = 0.0
                rec_count = 0
                qvals_accum: List[float] = []

                while not done:
                    # 후보군 임베딩 캐싱
                    cand_dict = env.get_candidates()
                    for ctype, cands in cand_dict.items():
                        for c in cands:
                            cid = c["id"]
                            emb_cache.setdefault(cid, embedder.embed_content(c))

                    # 슬레이트 선택
                    slate = agent.select_slate(
                        state,
                        {
                            t: [emb_cache[c["id"]].tolist() for c in lst]
                            for t, lst in cand_dict.items()
                        },
                        max_recs=max_recs,
                    )

                    # Exploitation 시 Q-value 기록
                    if random.random() >= agent.epsilon:
                        qvals = compute_all_q_values(
                            state, cand_dict, embedder, agent, emb_cache
                        )
                        qvals_accum.extend(qvals[t][i] for t, i in slate)

                    # 환경 단계 실행
                    next_state, reward, done, _, info = env.step(slate)
                    total_reward = reward
                    rec_count += len(slate)

                    # 경험 저장 및 학습
                    next_cands = env.get_candidates()
                    next_embs = {
                        t: [emb_cache[c["id"]] for c in lst]
                        for t, lst in next_cands.items()
                    }
                    for ctype, idx in slate:
                        selected = cand_dict[ctype][idx]
                        agent.store(
                            state,
                            emb_cache[selected["id"]],
                            info["individual_rewards"].get(int(selected["id"]), 0.0),
                            next_state,
                            next_embs,
                            done,
                        )
                    loss = agent.learn()
                    agent.decay_epsilon()
                    state = next_state

                    step_metrics.append(
                        {
                            "seed": seed,
                            "query": query,
                            "episode": ep,
                            "step_reward": reward,
                            "recommendations": rec_count,
                            "clicks": info.get("total_clicks", 0),
                            "click_ratio": (
                                info.get("total_clicks", 0) / rec_count
                                if rec_count
                                else 0.0
                            ),
                            "epsilon": agent.epsilon,
                            "loss": loss or -1.0,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                # 에피소드 종료 처리
                qvar = float(np.var(qvals_accum)) if qvals_accum else float("nan")
                episode_metrics.append(
                    {
                        "seed": seed,
                        "episode": ep,
                        "query": query,
                        "total_reward": total_reward,
                        "recommendations": rec_count,
                        "clicks": info.get("total_clicks", 0),
                        "click_ratio": (
                            info.get("total_clicks", 0) / rec_count
                            if rec_count
                            else 0.0
                        ),
                        "epsilon": agent.epsilon,
                        "qvalue_variance": qvar,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                if ep % exp_cfg.get("save_interval", 5) == 0:
                    path = os.path.join(model_dir, f"model_ep{ep}.pth")
                    agent.save(path)
                    logging.info("Saved checkpoint: %s", path)

            except Exception:
                logging.exception("Error in episode %d (seed=%d)", ep, seed)

        # 최종 저장
        final_path = os.path.join(model_dir, "model_final.pth")
        agent.save(final_path)
        logging.info("Saved final model: %s", final_path)

        self._save_csv(step_metrics, step_log)
        self._save_csv(episode_metrics, eps_log)

    def _save_csv(self, data: List[Dict[str, Any]], path: str) -> None:
        """메트릭 리스트를 CSV 파일로 저장합니다.

        Args:
            data (List[Dict[str, Any]]): 저장할 메트릭 딕셔너리 목록.
            path (str): CSV 파일 경로.
        """
        if not data:
            return

        fieldnames = list(data[0].keys())
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(data)
        logging.info("Wrote %d rows to %s", len(data), path)

    def run_all(self) -> None:
        """설정된 모든 시드에 대해 run_single을 호출하여 실험을 수행합니다."""
        seeds = self.cfg["experiment"].get("seeds", [0])
        for s in seeds:
            try:
                self.run_single(s)
            except Exception:
                logging.exception("Experiment failed for seed %d", s)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ExperimentRunner().run_all()
