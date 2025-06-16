import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_metrics(
    csv_path: str,
    window: int = 5,
    figsize: tuple[int, int] = (12, 10),
    seeds: Optional[list[int]] = None,
) -> None:
    """에피소드별 학습 지표를 플로팅합니다.

    CSV 파일에서 seed별로 Total Reward, Click Ratio, Epsilon, Q-value Variance를
    가져와 2×2 서브플롯으로 그립니다.

    Args:
        csv_path (str): 에피소드 로그 CSV 파일 경로.
        window (int, optional): 이동 평균 윈도우 크기. 기본값 5.
        figsize (tuple[int, int], optional): 플롯 전체 크기. 기본값 (12, 10).
        seeds (Optional[list[int]], optional):
            특정 seed 리스트만 플롯하려면 전달. None이면 모든 seed를 사용.

    Raises:
        FileNotFoundError: 지정된 CSV 파일이 존재하지 않을 때.
        ValueError: DataFrame에 필요한 컬럼이 없을 때.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(path)

    required_cols = {
        "seed",
        "episode",
        "total_reward",
        "click_ratio",
        "epsilon",
        "qvalue_variance",
    }
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"필수 컬럼 누락: {missing}")

    # 그릴 seed 목록 결정
    seed_values = seeds or sorted(df["seed"].unique())

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axes_flat = axes.flatten()

    # (1) Total Reward
    ax = axes_flat[0]
    for seed in seed_values:
        grp = df[df["seed"] == seed]
        ax.plot(grp["episode"], grp["total_reward"], label=f"Seed {seed}")
        ax.plot(
            grp["episode"], grp["total_reward"].rolling(window).mean(), linestyle="--"
        )
    ax.set(title="Total Reward", ylabel="Reward")
    ax.legend()
    ax.grid(True)

    # (2) Click Ratio
    ax = axes_flat[1]
    for seed in seed_values:
        grp = df[df["seed"] == seed]
        ax.plot(grp["episode"], grp["click_ratio"], label=f"Seed {seed}")
        ax.plot(
            grp["episode"], grp["click_ratio"].rolling(window).mean(), linestyle="--"
        )
    ax.set(title="Click Ratio", ylabel="Ratio")
    ax.legend()
    ax.grid(True)

    # (3) Epsilon
    ax = axes_flat[2]
    for seed in seed_values:
        grp = df[df["seed"] == seed]
        ax.plot(grp["episode"], grp["epsilon"], label=f"Seed {seed}")
    ax.set(title="Epsilon", xlabel="Episode", ylabel="Epsilon")
    ax.legend()
    ax.grid(True)

    # (4) Q-value Variance
    ax = axes_flat[3]
    for seed in seed_values:
        grp = df[df["seed"] == seed]
        ax.plot(grp["episode"], grp["qvalue_variance"], label=f"Seed {seed}")
    ax.set(title="Q-value Variance", xlabel="Episode", ylabel="Variance")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        plot_training_metrics("./data/logs/dqn_base/seed_0/episodes.csv")
    except Exception as e:
        logging.error("플로팅 중 오류 발생: %s", e)
