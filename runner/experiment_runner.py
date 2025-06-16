import os
import csv
import logging
from datetime import datetime
from typing import List, Dict, Any

import random
import numpy as np
import pandas as pd
import torch
import yaml

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
    """
    실험 전체 루프를 실행하는 클래스.

    YAML 설정 파일의 seed마다 실험을 반복 실행합니다.

    Attributes:
        cfg (Dict[str, Any]): 실험 전체 설정값
    """

    def __init__(self, config_path: str = "config/experiment.yaml") -> None:
        """ExperimentRunner 객체를 초기화합니다.

        Args:
            config_path (str): 실험 설정 파일 경로 (YAML).
        """
        self.cfg: Dict[str, Any] = yaml.safe_load(open(config_path))
        logging.info("ExperimentRunner initialized.")

    def set_seed(self, seed: int) -> None:
        """
        전체 환경의 random seed를 고정합니다.

        Args:
            seed (int): 사용할 랜덤 시드 값
        """
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logging.info(f"Random seed set: {seed}")

    def run_single(self, seed: int) -> None:
        """
        하나의 시드로 모든 episode를 실행합니다.

        Args:
            seed (int): 실험 reproducibility를 위한 시드

        Raises:
            ValueError: 지원하지 않는 시뮬레이터 타입 또는 env.step 반환값 이상
        """
        self.set_seed(seed)
        cfg: Dict[str, Any] = self.cfg
        exp_cfg = cfg["experiment"]

        # 파일 및 디렉토리 준비
        exp_name = exp_cfg.get("experiment_name", "default_exp")
        step_log_path = exp_cfg["step_log_path"].format(
            experiment_name=exp_name, seed=seed
        )
        episode_log_path = exp_cfg["episode_log_path"].format(
            experiment_name=exp_name, seed=seed
        )
        model_save_dir = exp_cfg["model_save_dir"].format(
            experiment_name=exp_name, seed=seed
        )
        os.makedirs(os.path.dirname(step_log_path), exist_ok=True)
        os.makedirs(os.path.dirname(episode_log_path), exist_ok=True)
        os.makedirs(model_save_dir, exist_ok=True)

        # 데이터 로드
        contents_df = get_contents()
        users_df = get_users()
        logs_df = get_user_logs()

        # 로그와 콘텐츠 병합
        if not logs_df.empty and not contents_df.empty:
            logs_with_type_df = pd.merge(
                logs_df,
                contents_df[["id", "type"]].rename(
                    columns={"id": "content_id", "type": "content_db_type"}
                ),
                on="content_id",
                how="left",
            )
        else:
            logs_with_type_df = logs_df

        # 컴포넌트 인스턴스화
        embedder = make(cfg["embedder"]["type"], **cfg["embedder"]["params"])
        candgen = make(
            cfg["candidate_generator"]["type"],
            **cfg["candidate_generator"]["params"],
            contents_df=contents_df,
        )
        reward_fn = make(cfg["reward_fn"]["type"], **cfg["reward_fn"]["params"])

        # 사용자 반응 시뮬레이터
        sim_cfg = cfg["response_simulator"]
        sim_type = sim_cfg["type"]
        sim_params = sim_cfg.get("params", {}).copy()
        if sim_type == "random":
            response_simulator = RandomResponseSimulator(**sim_params)
        elif sim_type == "llm":
            llm_client_cfg = sim_params.pop("llm_simulator")
            llm_client = LLMUserSimulator(**llm_client_cfg.get("params", {}))
            response_simulator = LLMResponseSimulator(
                llm_simulator=llm_client, **sim_params
            )
        else:
            raise ValueError(f"Unsupported simulator type: {sim_type}")

        # 환경 및 에이전트 생성
        env = make(
            cfg["env"]["type"],
            **cfg["env"]["params"],
            embedder=embedder,
            candidate_generator=candgen,
            reward_fn=reward_fn,
            response_simulator=response_simulator,
            contents_df=contents_df,
            users_df=users_df,
            logs_with_type_df=logs_with_type_df,
        )
        agent = make(
            cfg["agent"]["type"],
            user_dim=embedder.user_dim,
            content_dim=embedder.content_dim,
            capacity=cfg["replay"]["capacity"],
            **cfg["agent"]["params"],
        )

        total_eps = cfg["experiment"]["total_episodes"]
        max_recs = cfg["experiment"]["max_recommendations"]
        max_stocks = cfg["experiment"].get("max_stocks", 89)

        # 쿼리(주식 이름) 선정
        stock_names: List[str] = get_stock_names()
        if not stock_names:
            raise ValueError("No stock names available in the database")
        stock_names = stock_names[:max_stocks]
        logging.info(
            f"Selected {len(stock_names)} stocks for experiment: {stock_names}"
        )

        queries = []
        cycles, rem = divmod(total_eps, len(stock_names))
        for _ in range(cycles):
            random.shuffle(stock_names)
            queries.extend(stock_names)
        random.shuffle(stock_names)
        queries.extend(stock_names[:rem])

        episode_metrics = []
        step_metrics = []

        for ep, query in enumerate(queries, start=1):
            qvalue_list = []
            try:
                logging.info(
                    f"\n--- Episode {ep}/{total_eps} (seed={seed}, query={query}) ---"
                )
                state, _ = env.reset(options={"query": query})
                done = False
                total_reward = 0.0
                rec_count = 0
                emb_cache: Dict[Any, Any] = {}

                while not done:
                    # 후보 임베딩 캐시
                    cand_dict: Dict[str, List[Any]] = env.get_candidates()
                    for ctype, cands in cand_dict.items():
                        for cand in cands:
                            cid = cand.get("id")
                            if cid not in emb_cache:
                                emb_cache[cid] = embedder.embed_content(cand)

                    # 에이전트로 추천 슬레이트 선택
                    candidate_embs: Dict[str, List[List[float]]] = {}
                    for ctype, contents in cand_dict.items():
                        if contents:
                            embeddings = [
                                emb_cache.get(c.get("id"), embedder.embed_content(c))
                                for c in contents
                            ]
                            candidate_embs[ctype] = [emb.tolist() for emb in embeddings]
                        else:
                            candidate_embs[ctype] = []
                    enforce_list = agent.select_slate(
                        state, candidate_embs, max_recs=max_recs
                    )

                    # Q-value 계산 (로깅용, exploitation 시만)
                    if random.random() >= agent.epsilon:
                        q_values: Dict[str, List[float]] = compute_all_q_values(
                            state, cand_dict, embedder, agent, emb_cache=emb_cache
                        )
                        qvalue_list.extend(
                            [q_values[t][idx] for t, idx in enforce_list]
                        )

                    # step 실행 및 결과 처리
                    step_result = env.step(enforce_list)
                    if not (
                        isinstance(step_result, (tuple, list)) and len(step_result) == 5
                    ):
                        raise ValueError(
                            "env.step must return a tuple or list of length 5: (next_state, reward, done, truncated, info)"
                        )
                    next_state, total_reward, step_done, truncated, info = step_result
                    done = step_done or truncated
                    rec_count += len(enforce_list)

                    logging.info(
                        f"    Recommended {len(enforce_list)} contents → total_reward={total_reward}, done={done}"
                    )
                    logging.info(
                        f"    Clicks: {info.get('total_clicks', 0)}/{len(enforce_list)}"
                    )

                    # 다음 state의 후보 임베딩 미리 캐싱
                    next_cand_dict: Dict[str, List[Any]] = env.get_candidates()
                    next_cembs: Dict[str, List[Any]] = {
                        t: [
                            emb_cache.get(c.get("id"), embedder.embed_content(c))
                            for c in cs
                        ]
                        for t, cs in next_cand_dict.items()
                    }

                    # 리플레이 버퍼에 transition 저장 및 학습
                    for ctype, idx in enforce_list:
                        cands = cand_dict.get(ctype, [])
                        if idx < 0 or idx >= len(cands):
                            continue
                        selected = cands[idx]
                        cid = selected.get("id")
                        selected_emb = emb_cache[cid]
                        individual_reward = info.get("individual_rewards", {}).get(
                            int(selected.get("id")), 0.0
                        )
                        agent.store(
                            state,
                            selected_emb,
                            individual_reward,
                            next_state,
                            next_cembs,
                            done,
                        )
                    loss = agent.learn()
                    state = next_state
                    agent.decay_epsilon()

                    step_metrics.append(
                        {
                            "seed": seed,
                            "query": query,
                            "total_reward": total_reward,
                            "recommendations": rec_count,
                            "clicks": info.get("total_clicks", 0),
                            "click_ratio": (
                                info.get("total_clicks", 0) / rec_count
                                if rec_count
                                else 0
                            ),
                            "epsilon": getattr(agent, "epsilon", float("nan")),
                            "datetime": datetime.now().isoformat(),
                            "loss": loss if type(loss) == float else -1,
                        }
                    )

                logging.info(
                    f"--- Episode {ep} End. Agent Epsilon: {getattr(agent, 'epsilon', float('nan')):.3f} ---"
                )

                # Q-value 분산 계산 (에피소드별)
                qvalue_variance = float("nan")
                if qvalue_list:
                    qvalue_variance = np.var(qvalue_list)
                logging.info(
                    f"--- Q-value Variance (Episode {ep}): {qvalue_variance:.6f}"
                )

                episode_metrics.append(
                    {
                        "seed": seed,
                        "episode": ep,
                        "query": query,
                        "total_reward": total_reward,
                        "recommendations": rec_count,
                        "clicks": info.get("total_clicks", 0),
                        "click_ratio": (
                            info.get("total_clicks", 0) / rec_count if rec_count else 0
                        ),
                        "epsilon": getattr(agent, "epsilon", float("nan")),
                        "datetime": datetime.now().isoformat(),
                        "qvalue_variance": qvalue_variance,
                    }
                )

                if ep % 5 == 0:
                    model_path = os.path.join(model_save_dir, f"dqn_model_ep{ep}.pth")
                    agent.save(model_path)
                    logging.info(f"Model saved to {model_path}")

            except Exception as e:
                logging.error(f"Error in episode {ep}, seed {seed}: {e}", exc_info=True)

        # 마지막 모델 및 결과 저장
        final_model_path = os.path.join(model_save_dir, "dqn_model_final.pth")
        agent.save(final_model_path)
        logging.info(f"Final model saved to {final_model_path}")

        self.save_results(step_metrics, step_log_path)
        self.save_results(episode_metrics, episode_log_path)

    def save_results(self, metrics: List[Dict[str, Any]], csv_path: str) -> None:
        """
        메트릭 리스트를 CSV 파일에 저장합니다.

        Args:
            metrics (List[Dict[str, Any]]): 저장할 메트릭 리스트
            csv_path (str): append할 CSV 경로
        """
        if not metrics:
            return

        fieldnames = metrics[0].keys()
        write_header = not os.path.exists(csv_path)

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in metrics:
                writer.writerow(row)
        logging.info(f"Saved {len(metrics)} rows to {csv_path}")

    def run_all(self) -> None:
        """
        설정된 모든 시드에 대해 실험을 실행합니다.
        """
        seeds: List[int] = self.cfg["experiment"].get("seeds", [0])
        for s in seeds:
            try:
                self.run_single(s)
            except Exception as e:
                logging.error(f"Seed {s} experiment failed: {e}", exc_info=True)
                continue


if __name__ == "__main__":
    ExperimentRunner().run_all()
