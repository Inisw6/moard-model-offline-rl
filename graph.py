import pandas as pd
import matplotlib.pyplot as plt

# CSV 로드
df = pd.read_csv("./data/experiment_results.csv")

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
window = 5  # 이동평균 윈도우

# 1) seed마다, 총 보상 (Total Reward)
ax = axes[0, 0]
for seed, grp in df.groupby("seed"):
    ax.plot(grp["episode"], grp["total_reward"], label=f"seed {seed}")
    ax.plot(grp["episode"], grp["total_reward"].rolling(window).mean(), linestyle="--")

ax.set_title("Total Reward")
ax.set_ylabel("Reward")
ax.legend()
ax.grid(True)

# 2) seed마다, 클릭 비율 (Click Ratio)
ax = axes[0, 1]
for seed, grp in df.groupby("seed"):
    ax.plot(grp["episode"], grp["click_ratio"], label=f"seed {seed}")
    ax.plot(grp["episode"], grp["click_ratio"].rolling(window).mean(), linestyle="--")

ax.set_title("Click Ratio")
ax.set_ylabel("Ratio")
ax.legend()
ax.grid(True)

# 3) seed마다, epsilon 값
ax = axes[1, 0]
for seed, grp in df.groupby("seed"):
    ax.plot(grp["episode"], grp["epsilon"], label=f"seed {seed}")
ax.set_title("Epsilon")
ax.set_xlabel("Episode")
ax.set_ylabel("Epsilon")
ax.legend()
ax.grid(True)

# 4) seed마다, 추천 수 (Recommendations)
ax = axes[1, 1]
for seed, grp in df.groupby("seed"):
    ax.plot(grp["episode"], grp["recommendations"], label=f"seed {seed}")
ax.set_title("Recommendations")
ax.set_xlabel("Episode")
ax.set_ylabel("Count")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()
