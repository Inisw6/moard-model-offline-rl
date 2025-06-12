import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 경로
csv_path = 'data/experiment_results.csv'

# CSV 로드
df = pd.read_csv(csv_path)

# 인덱스를 Step 번호로 사용
steps = df[df['loss'] >= 0].index + 1
loss_values = df[df['loss'] >= 0]['loss']
rolling_mean = loss_values.rolling(window=5).mean()

# Plot
plt.figure(figsize=(10, 5))
plt.plot(steps, loss_values)
plt.plot(steps, rolling_mean, linestyle='--')
plt.title('Moving Average of Loss (Window 5)')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
