import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

def plot_monitor():
    plt.ion() 
    # 建立三個子圖：Loss/LR, Gate Active, Entropy
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 建立雙 Y 軸給第一個圖 (Loss 和 LR)
    ax1_lr = ax1.twinx()

    while True:
        try:
            df = pd.read_csv("train_log.csv")
            if len(df) < 5: 
                print("等待更多數據中...")
                time.sleep(5)
                continue

            # 清除舊內容
            ax1.clear()
            ax1_lr.clear()
            ax2.clear()
            ax3.clear()

            # 計算移動平均 (Smoothing) 讓趨勢更明顯
            window = min(len(df), 20)
            smooth_loss = df['loss'].rolling(window=window).mean()
            smooth_gate = df['gate'].rolling(window=window).mean()
            smooth_ent = df['entropy'].rolling(window=window).mean()

            # --- 圖 1: Loss (左) & Learning Rate (右) ---
            ax1.set_ylabel('Loss (Log Scale)', color='tab:red')
            ax1.plot(df['step'], df['loss'], color='tab:red', alpha=0.2) # 原始資料
            ax1.plot(df['step'], smooth_loss, color='tab:red', linewidth=2, label='Loss (SMA)') # 平滑線
            ax1.set_yscale('log')
            ax1.tick_params(axis='y', labelcolor='tab:red')
            ax1.grid(True, which="both", ls="-", alpha=0.2)

            ax1_lr.set_ylabel('Learning Rate', color='gray')
            ax1_lr.plot(df['step'], df['lr'], color='gray', linestyle='--', alpha=0.7, label='LR')
            ax1_lr.tick_params(axis='y', labelcolor='gray')

            # --- 圖 2: Gate Active (Sparsity) ---
            ax2.set_ylabel('Gate Active', color='tab:blue')
            ax2.plot(df['step'], df['gate'], color='tab:blue', alpha=0.2)
            ax2.plot(df['step'], smooth_gate, color='tab:blue', linewidth=2, label='Gate (SMA)')
            ax2.set_ylim(0, 1) # Gate 範圍 0~1
            ax2.axhline(y=0.5, color='black', linestyle=':', alpha=0.3) # 參考線
            ax2.tick_params(axis='y', labelcolor='tab:blue')

            # --- 圖 3: Gate Entropy (關鍵健康指標) ---
            ax3.set_xlabel('Steps')
            ax3.set_ylabel('Entropy', color='tab:green')
            ax3.plot(df['step'], df['entropy'], color='tab:green', alpha=0.2)
            ax3.plot(df['step'], smooth_ent, color='tab:green', linewidth=2, label='Entropy (SMA)')
            ax3.tick_params(axis='y', labelcolor='tab:green')
            # 理想 Entropy 應保持在一定水準，若接近 0 代表 Routing Collapse
            
            plt.suptitle('🌊 D2-V15 Resonance-Memory Training Monitor', fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.pause(10)

        except Exception as e:
            print(f"等待日誌更新中... {e}")
            time.sleep(5)

if __name__ == "__main__":
    plot_monitor()
