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
from components.recommendation.rec_utils import (
    compute_all_q_values,
)
from components.simulation.simulators import (
    RandomResponseSimulator,
    LLMResponseSimulator,
)
from components.simulation.llm_simu import LLMUserSimulator


class ExperimentRunner:
    """
    실험 전체 루프를 실행합니다.

    YAML 구성 파일에 정의된 대로 각 시드를 반복 수행합니다.

    Attributes:
        cfg (Dict[str, Any]): 실험 설정을 담은 딕셔너리
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
        ExperimentRunner 객체를 초기화합니다.

        Args:
            config_path (str): 실험 설정 YAML 파일 경로
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
        단일 시드에 대해 모든 에피소드를 실행합니다.

        Args:
            seed (int): 재현성을 위한 랜덤 시드

        Raises:
            ValueError: 지원하지 않는 시뮬레이터 타입이거나 env.step() 반환값이 올바르지 않은 경우
        """
        self.set_seed(seed)
        cfg: Dict[str, Any] = self.cfg
        exp_cfg = cfg["experiment"]

        # 경로 설정 및 디렉토리 생성
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

        # 데이터 로딩 및 사전 처리
        contents_df = get_contents()
        users_df = get_users()
        logs_df = get_user_logs()

        # 로그 + 콘텐츠 type 병합
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

        # Embedder / CandidateGenerator / Reward 초기화
        embedder = make(cfg["embedder"]["type"], **cfg["embedder"]["params"])
        candgen = make(
            cfg["candidate_generator"]["type"],
            **cfg["candidate_generator"]["params"],
            contents_df=contents_df,  # 주입
        )
        reward_fn = make(cfg["reward_fn"]["type"], **cfg["reward_fn"]["params"])

        # 사용자 반응 시뮬레이터 생성
        sim_cfg = cfg["response_simulator"]
        sim_type = sim_cfg["type"]
        sim_params = sim_cfg.get("params", {}).copy()

        if sim_type == "random":
            response_simulator = RandomResponseSimulator(**sim_params)
        elif sim_type == "llm":
            # LLM 클라이언트(LLMUserSimulator) 생성
            llm_client_cfg = sim_params.pop("llm_simulator")
            llm_client = LLMUserSimulator(**llm_client_cfg.get("params", {}))

            # LLMResponseSimulator 생성
            response_simulator = LLMResponseSimulator(
                llm_simulator=llm_client, **sim_params
            )
        else:
            raise ValueError(f"Unsupported simulator type: {sim_type}")

        # 환경 생성
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
        max_stocks: int = cfg["experiment"].get("max_stocks", 89)

        # 주식 이름 리스트 가져오기
        stock_names: List[str] = get_stock_names()
        if not stock_names:
            raise ValueError("No stock names available in the database")

        # max_stocks만큼만 종목 선택
        stock_names = stock_names[:max_stocks]
        logging.info(
            f"Selected {len(stock_names)} stocks for experiment: {stock_names}"
        )

        # 쿼리에 랜덤성 추가
        queries = []
        cycles, rem = divmod(total_eps, len(stock_names))
        for _ in range(cycles):
            random.shuffle(stock_names)
            queries.extend(stock_names)
        random.shuffle(stock_names)
        queries.extend(stock_names[:rem])

        episode_metrics = []
        step_metrics = []

        # 쿼리별 에피소드 시작
        for ep, query in enumerate(queries, start=1):
            qvalue_list = []
            try:
                # 각 에피소드 시작 시 로그 출력
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

                    # 에이전트의 select_slate를 통해 슬레이트 선택
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

                    # Q-value 계산 (로깅용)
                    if random.random() >= agent.epsilon:
                        # Exploitation인 경우에만 Q-value 계산
                        q_values: Dict[str, List[float]] = compute_all_q_values(
                            state, cand_dict, embedder, agent, emb_cache=emb_cache
                        )
                        qvalue_list.extend(
                            [q_values[t][idx] for t, idx in enforce_list]
                        )

                    # step 실행
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
                        f"    Recommended {len(enforce_list)} contents → "
                        f"total_reward={total_reward}, done={done}"
                    )
                    logging.info(
                        f"    Clicks: {info.get('total_clicks', 0)}/{len(enforce_list)}"
                    )

                    # 계산 로직 한번만 실행
                    next_cand_dict: Dict[str, List[Any]] = env.get_candidates()
                    next_cembs: Dict[str, List[Any]] = {
                        t: [
                            emb_cache.get(c.get("id"), embedder.embed_content(c))
                            for c in cs
                        ]
                        for t, cs in next_cand_dict.items()
                    }

                    # RL 학습 데이터 저장 및 학습
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
                    # 에이전트 학습 -> 위치에 따라 다름, 지금은 6개 모두 추천 작업 진행 후 학습
                    loss = agent.learn()
                    state = next_state

                    # ε 감소: 한 env.step 당 한 번만 적용
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

                # 에피소드 종료 후 Q-value 분산 계산
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

                # 에피소드 5개 진행마다 모델 파일 저장
                if ep % 5 == 0:
                    model_path = os.path.join(model_save_dir, f"dqn_model_ep{ep}.pth")
                    agent.save(model_path)
                    logging.info(f"Model saved to {model_path}")

            except Exception as e:
                logging.error(f"Error in episode {ep}, seed {seed}: {e}", exc_info=True)

        # 최종 모델 저장
        final_model_path = os.path.join(model_save_dir, "dqn_model_final.pth")
        agent.save(final_model_path)
        logging.info(f"Final model saved to {final_model_path}")

        self.save_results(step_metrics, step_log_path)
        self.save_results(episode_metrics, episode_log_path)

    def save_results(self, metrics: List[Dict[str, Any]], csv_path: str) -> None:
        """
        메트릭 리스트를 CSV 파일에 저장합니다.

        Args:
            metrics (List[Dict[str, Any]]): 저장할 메트릭 딕셔너리 리스트
            csv_path (str): 결과를 append할 CSV 파일 경로
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
