import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 경로
csv_path = 'data/experiment_results.csv'

# CSV 로드
df = pd.read_csv(csv_path)
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# [total reward]
ax = axes[0]
steps = df['total_reward'].index + 1
total_reward = df['total_reward']
rolling_mean = total_reward.rolling(window=5).mean()
ax.plot(steps, df["total_reward"], label="Total Reward")
ax.plot(steps, rolling_mean, linestyle='--', label="Rolling Mean (window=5)")
ax.set_title("DuelingQN Total Reward")
ax.set_ylabel("Reward")
ax.legend()
ax.grid(True)

# [loss]
ax = axes[1]
steps = df[df['loss'] >= 0].index + 1
loss_values = df[df['loss'] >= 0]['loss']
rolling_mean = loss_values.rolling(window=5).mean()
ax.plot(steps, loss_values, label="Loss")
ax.plot(steps, rolling_mean, linestyle='--', label="Rolling Mean (window=5)")
ax.set_title('DuelingQN Loss (Window 5)')
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()