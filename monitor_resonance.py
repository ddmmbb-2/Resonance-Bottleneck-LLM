import pandas as pd
import matplotlib.pyplot as plt
import time
from IPython import display

def plot_monitor():
    plt.ion() # 開啟互動模式
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx() # 建立雙 Y 軸

    while True:
        try:
            # 讀取數據
            df = pd.read_csv("train_log.csv")
            if len(df) < 2: 
                time.sleep(5)
                continue

            ax1.clear()
            ax2.clear()

            # 繪製 Loss 曲線 (左軸)
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Cross-Entropy Loss', color='tab:red')
            ax1.plot(df['step'], df['loss'], color='tab:red', label='Loss')
            ax1.tick_params(axis='y', labelcolor='tab:red')
            ax1.set_yscale('log') # 建議用對數坐標，因為初始 Loss 很高

            # 繪製 Gate 活性曲線 (右軸)
            ax2.set_ylabel('Gate Active (Sparsity)', color='tab:blue')
            ax2.plot(df['step'], df['gate'], color='tab:blue', label='Gate', alpha=0.6)
            ax2.tick_params(axis='y', labelcolor='tab:blue')
            ax2.set_ylim(0, 1) # Gate 範圍固定在 0~1

            plt.title('D2-V13-Resonance Training Monitor')
            fig.tight_layout()
            plt.pause(10) # 每 10 秒更新一次圖表

        except Exception as e:
            print(f"等待日誌更新中... {e}")
            time.sleep(5)

if __name__ == "__main__":
    plot_monitor()