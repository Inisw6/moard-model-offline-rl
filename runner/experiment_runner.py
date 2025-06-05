import yaml
import random
import numpy as np
import torch
import logging

from typing import List, Dict, Any
from components.registry import make
from components.rec_context import RecContextManager, get_recommendation_quota


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

    def set_seed(self, seed: int) -> None:
        """
        전체 환경의 랜덤 시드를 고정합니다.

        Args:
            seed (int): 사용할 시드 값
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run_single(self, seed: int) -> None:
        """
        단일 seed에 대해 실험 에피소드를 수행합니다.

        Args:
            seed (int): 사용할 시드 값
        """
        self.set_seed(seed)
        cfg: Dict[str, Any] = self.cfg

        # 1) Embedder / CandidateGenerator / Reward / Context / Env / Agent 초기화
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
            logging.info(f"\n--- Episode {ep+1}/{total_eps} ---")

            # todo: query를 정해야 하는데 여기서 종목 하나로 에피소드 끝까지 진행해야함!!!
            query: str = "종목12 뉴스"

            # 환경 reset: 쿼리를 options에 전달
            state, _ = env.reset(options={"query": query})
            done: bool = False

            while not done:
                cand_dict: Dict[str, List[Any]] = env.get_candidates()
                user_pref: Dict[str, float] = embedder.estimate_preference(state)
                quota: Dict[str, int] = get_recommendation_quota(
                    user_pref, context, max_total=max_recs
                )

                logging.info(
                    f"  Loop Start: ep={ep + 1}, env_step={env.step_count}, done={done}"
                )
                logging.info(f"    Query: {query}")
                logging.info(f"    State (first 5): {state[:5]}")
                logging.info(f"    User Pref: {user_pref}")
                logging.info(f"    Quota: {quota}")

                if not quota or all(v == 0 for v in quota.values()):
                    logging.warning(
                        "    Quota is empty or all zeros. Breaking inner loop to prevent infinite loop."
                    )
                    done = True
                    continue

                # 콘텐츠 타입별 추천 루프
                for ctype, cnt in quota.items():
                    logging.info(f"    Content Type Loop: ctype={ctype}, cnt={cnt}")
                    if cnt == 0:
                        continue

                    cands: List[Any] = cand_dict.get(ctype, [])
                    if not cands:
                        logging.warning(
                            f"    No candidates found for ctype {ctype}. Skipping."
                        )
                        continue

                    # 후보 임베딩 생성
                    cembs: List[Any] = [embedder.embed_content(c) for c in cands]
                    if not cembs:
                        logging.warning(
                            f"    No valid candidate embeddings for ctype {ctype}. Skipping."
                        )
                        continue

                    # 해당 타입에서 cnt번 추천 반복
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

                        selected_content_emb: Any = cembs[idx]
                        next_state, r, step_done, truncated, info = env.step(
                            (ctype, idx)
                        )
                        done = step_done or truncated

                        logging.info(
                            f"        env.step returned: reward={r}, step_done={step_done}, truncated={truncated}, final_done={done}"
                        )
                        logging.info(f"        Next state (first 5): {next_state[:5]}")

                        # 다음 후보와 임베딩 미리 준비 (optional)
                        next_cand_dict: Dict[str, List[Any]] = env.get_candidates()
                        next_cembs: Dict[str, List[Any]] = {
                            t: [embedder.embed_content(c) for c in cs]
                            for t, cs in next_cand_dict.items()
                        }

                        # 에이전트를 위한 replay 저장 및 학습
                        agent.store(
                            state, selected_content_emb, r, next_state, next_cembs, done
                        )
                        agent.learn()

                        # state 업데이트
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
        """
        config에 정의된 모든 seed에 대해 실험을 실행합니다.
        """
        seeds: List[int] = self.cfg["experiment"].get("seeds", [0])
        for s in seeds:
            self.run_single(s)


if __name__ == "__main__":
    """
    main 엔트리포인트. 실험을 실행합니다.
    """
    ExperimentRunner().run_all()
