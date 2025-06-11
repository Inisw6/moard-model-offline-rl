import yaml
import random
import numpy as np
import torch
import logging
import os
import csv
from typing import List, Dict, Any, Tuple
from datetime import datetime

from components.registry import make
from components.database.db_utils import get_stock_names
from components.recommendation.rec_context import RecContextManager
from components.recommendation.rec_utils import (
    enforce_type_constraint,
    compute_all_q_values,
)
from components.simulation.simulators import (
    RandomResponseSimulator,
    LLMResponseSimulator,
)
from components.simulation.llm_simu import LLMUserSimulator


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
        """
        실험 루프를 실행하여 구성된 모든 에피소드를 처리합니다.

        Args:
            seed (int): The random seed used to initialize all RNGs for reproducibility.

        Raises:
            ValueError: If an unsupported simulator type is specified or if env.step() returns
                        an invalid result format.
        """
        self.set_seed(seed)
        cfg: Dict[str, Any] = self.cfg

        # Embedder / CandidateGenerator / Reward / Context 초기화
        embedder = make(cfg["embedder"]["type"], **cfg["embedder"]["params"])
        candgen = make(
            cfg["candidate_generator"]["type"], **cfg["candidate_generator"]["params"]
        )
        reward_fn = make(cfg["reward_fn"]["type"], **cfg["reward_fn"]["params"])
        context = RecContextManager(cfg["env"]["params"]["cold_start"])

        # 사용자 반응 시뮬레이터 생성
        sim_cfg = cfg["response_simulator"]
        sim_type = sim_cfg["type"]
        sim_params = sim_cfg.get("params", {}).copy()

        if sim_type == "random":
            response_simulator = RandomResponseSimulator(**sim_params)
        elif sim_type == "llm":
            # 1. LLM 클라이언트(LLMUserSimulator) 생성
            llm_client_cfg = sim_params.pop("llm_simulator")
            llm_client = LLMUserSimulator(**llm_client_cfg.get("params", {}))

            # 2. LLMResponseSimulator 생성
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
            context=context,
            response_simulator=response_simulator,
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

        queries = []
        cycles, rem = divmod(total_eps, len(stock_names))
        for _ in range(cycles):
            random.shuffle(stock_names)
            queries.extend(stock_names)
        random.shuffle(stock_names)
        queries.extend(stock_names[:rem])

        episode_metrics = []

        for ep, query in enumerate(queries, start=1):
            try:
                # 각 에피소드 시작 시 로그 출력
                logging.info(
                    f"\n--- Episode {ep}/{total_eps} (seed={seed}, query={query}) ---"
                )

                state, _ = env.reset(options={"query": query})
                done: bool = False
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

                    # ε–greedy 탐험/활용 분기
                    if random.random() < agent.epsilon:
                        # Exploration: 무작위 슬레이트 생성
                        enforce_list: List[Tuple[str, int]] = []
                        types = list(cand_dict.keys())
                        for _ in range(max_recs):
                            t = random.choice(types)
                            idx = random.randrange(len(cand_dict[t]))
                            enforce_list.append((t, idx))
                    else:
                        # Exploitation: Q값 기반 슬레이트 생성
                        q_values: Dict[str, List[float]] = compute_all_q_values(
                            state, cand_dict, embedder, agent, emb_cache=emb_cache
                        )
                        enforce_list: List[Tuple[str, int]] = enforce_type_constraint(
                            q_values, top_k=max_recs
                        )

                    # 환경에 한 번에 step 실행
                    step_result = env.step(enforce_list)
                    if (
                        not isinstance(step_result, (tuple, list))
                        or len(step_result) != 5
                    ):
                        raise ValueError(
                            "env.step must return (next_state, reward, done, truncated, info)"
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
                        # todo: 학습 위치 고려중
                        # agent.learn()
                    agent.learn()
                    state = next_state
                logging.info(
                    f"--- Episode {ep} End. Agent Epsilon: {getattr(agent, 'epsilon', float('nan')):.3f} ---"
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
                    }
                )

                if ep % 5 == 0:
                    save_dir = "saved_models"
                    os.makedirs(save_dir, exist_ok=True)
                    model_path = os.path.join(
                        save_dir, f"dqn_model_seed{seed}_ep{ep}.pth"
                    )
                    agent.save(model_path)
                    logging.info(f"Model saved to {model_path}")

            except Exception as e:
                logging.error(f"Error in episode {ep}, seed {seed}: {e}", exc_info=True)

        # 최종 모델 저장
        save_dir = "saved_models"
        os.makedirs(save_dir, exist_ok=True)
        final_model_path = os.path.join(save_dir, f"dqn_model_seed{seed}_final.pth")
        agent.save(final_model_path)
        logging.info(f"Final model saved to {final_model_path}")

        self.save_results(episode_metrics)

    def save_results(self, metrics: List[Dict[str, Any]]) -> None:
        """
        실험 결과(metrics)를 CSV 파일로 저장합니다.

        Args:
            metrics (List[Dict[str, Any]]): 저장할 메트릭 리스트.
                각 딕셔너리의 키가 CSV의 컬럼명(fieldnames)이 됩니다.

        Returns:
            None

        Notes:
            - 로그 파일 경로(self.result_log_path)의 확장자를 .csv로 변경하여 저장합니다.
            - 파일이 존재하지 않으면 헤더를 작성(writeheader)하고, 이후에는 이어쓰기 모드("a")로 기록합니다.
            - metrics가 비어 있으면 아무 작업도 수행하지 않습니다.
        """
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
