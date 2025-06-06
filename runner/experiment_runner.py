import yaml
import random
import numpy as np
import torch
import logging

from typing import List, Dict, Any, Tuple
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
        self.set_seed(seed)
        cfg: Dict[str, Any] = self.cfg

        # Embedder / CandidateGenerator / Reward / Context / Env / Agent 초기화
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
            # (A) 쿼리 설정
            # todo: query를 정해야 하는데 여기서 종목 하나로 에피소드 끝까지 진행해야함!!!
            query: str = "종목12 뉴스"

            # (B) Env reset: 쿼리를 전달
            state, _ = env.reset(options={"query": query})
            done: bool = False

            while not done:
                # (C) 후보 생성: env 내부에서 current_query를 사용
                cand_dict: Dict[str, List[Any]] = env.get_candidates()

                # (D) 모든 후보에 대해 Q값 계산
                q_values: Dict[str, List[float]] = compute_all_q_values(
                    state, cand_dict, embedder, agent
                )

                # (E) 후처리: 타입별 최소 1개를 보장하여 최종 (ctype, idx) 리스트 생성
                enforce_list: List[Tuple[str, int]] = enforce_type_constraint(
                    q_values, top_k=max_recs
                )

                # (F) 최종 추천 리스트 순회하며 env.step 호출
                for ctype, idx in enforce_list:
                    # 1) idx가 후보 리스트 범위 내에 있는지 한번 더 체크
                    cands = cand_dict.get(ctype, [])
                    if idx < 0 or idx >= len(cands):
                        logging.warning(
                            f"Invalid candidate index {idx} for type '{ctype}'. Skipping."
                        )
                        continue

                    selected_emb = embedder.embed_content(
                        cands[idx]
                    )  # 에이전트 저장용 임베딩

                    # 2) env.step 호출
                    next_state, r, step_done, truncated, info = env.step((ctype, idx))
                    done = step_done or truncated

                    logging.info(
                        f"    Recommended: (type={ctype}, idx={idx}) → reward={r}, done={done}"
                    )

                    # 3) 다음 후보 계산(옵션)
                    next_cand_dict: Dict[str, List[Any]] = env.get_candidates()

                    # 4) 에이전트 리플레이 저장 및 학습
                    #    next_cembs 형식: {'youtube': [emb0, emb1, ...], ...}
                    next_cembs: Dict[str, List[Any]] = {
                        t: [embedder.embed_content(c) for c in cs]
                        for t, cs in next_cand_dict.items()
                    }
                    agent.store(state, selected_emb, r, next_state, next_cembs, done)
                    agent.learn()

                    # 5) 상태 업데이트
                    state = next_state

                    if done:
                        break

            logging.info(
                f"--- Episode {ep+1} End. Agent Epsilon: {agent.epsilon:.3f} ---"
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
