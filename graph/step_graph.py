import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_loss_curve(
    csv_path: str,
    window: int = 5,
    figsize: tuple[int, int] = (10, 5),
    min_loss: float = 0.0,
) -> None:
    """손실(loss) 값을 스텝별로 플로팅하고 이동 평균을 표시합니다.

    Args:
        csv_path (str): 스텝별 손실이 저장된 CSV 파일 경로.
        window (int, optional): 이동 평균 윈도우 크기. 기본값 5.
        figsize (tuple[int, int], optional): 플롯 크기. 기본값 (10, 5).
        min_loss (float, optional): 최소 손실 값 필터링 기준. 기본값 0.0.

    Raises:
        FileNotFoundError: csv_path에 파일이 없을 경우.
        KeyError: 'loss' 컬럼이 없을 경우.
        ValueError: 필터링 후 남은 데이터가 없을 경우.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(path)
    if "loss" not in df.columns:
        raise KeyError("CSV에 'loss' 컬럼이 없습니다.")

    # 손실 값이 min_loss 이상인 행만 사용
    valid = df["loss"] >= min_loss
    if not valid.any():
        raise ValueError(f"'loss' >= {min_loss}인 데이터가 없습니다.")

    steps = df.index[valid] + 1
    loss_values = df.loc[valid, "loss"]
    rolling_mean = loss_values.rolling(window).mean()

    plt.figure(figsize=figsize)
    plt.plot(steps, loss_values, label="Loss")
    plt.plot(steps, rolling_mean, linestyle="--", label=f"Rolling Mean ({window})")
    plt.title("Moving Average of Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        plot_loss_curve("./data/logs/dqn_base/seed_0/steps.csv")
    except Exception as e:
        logging.error("플롯 생성 중 오류 발생: %s", e)
