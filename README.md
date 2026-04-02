

# 🌊 Resonance Memory Transformer (D2-V15) | 共振記憶轉換器

**Resonance Memory Transformer (D2-V15)** is an experimental Large Language Model (LLM) architecture exploring the synergy between **"Wave-Interference Implicit Gating"** and **"Linear Attention Memory Filtering."**

**Resonance Memory Transformer (D2-V15)** 是一個探索 **「波干涉隱式門控」** 與 **「線性注意力記憶過濾」** 結合的實驗性大型語言模型 (LLM) 架構。

This project evolved from `D2-Subset-LLM`. In the **V15** iteration, we've implemented major breakthroughs: moving beyond stateless linear attention to a neuro-inspired **EMA Memory Decay** and **Dual-Bounded Resonance** system. It achieves $O(N)$ linear complexity while maintaining robust logical extraction for long-context tasks.

本專案從 `D2-Subset-LLM` 演進而來，在 **V15** 版本中實現了重大突破：捨棄了傳統的無狀態線性注意力，引入了受生物神經學啟發的 **EMA 記憶衰減機制 (Memory Decay)** 與 **雙重限幅共振系統 (Dual-Bounded Resonance)**，在保持 $O(N)$ 線性複雜度的同時，賦予模型強大的長文本邏輯提煉能力。

---

## 🧠 Core Philosophy | 核心理念

Traditional linear attention often suffers from "memory blurring" and numerical instability as sequence length increases. Meanwhile, traditional MoE (Mixture of Experts) requires massive discrete router parameters. 

**V15 solves these via:**
1.  **Wave-Interference Routing (波干涉隱式路由):** No explicit expert networks. Features are mapped to "waves" with **Amplitude** and **Phase**. Interaction between features (constructive or destructive interference) naturally emerges as a context-specific activation (Gate).
2.  **Head-wise Memory Filter (多頭記憶過濾器):** Each head possesses independent resonance frequencies and decay rates. The model doesn't just "look" at data; it dynamically decides which memories to amplify or forget via gated resonance.
![png](20260402.png)
---

## ✨ V15 Key Features | 技術亮點

* **EMA Memory Decay (EMA 記憶衰減):** Upgraded from simple `cumsum` to a gated RNN-state update, providing "recency bias" and preventing memory collapse.
* **Denominator Normalization (數學對齊歸一化):** Synchronized EMA update for the normalizer state $Z_t$, eliminating scale drift common in linear attention.
* **Dual-Bounded Stability (雙重限幅穩定器):** Uses $\sigma$ for amplitude and $\tanh$ to anchor phase within $[-\pi, \pi]$, ensuring smooth gradient flow.
* **Entropy Regularization (熵正則化):** Gate Entropy loss prevents "Routing Collapse" (where gates stuck at all-0 or all-1).

---

## 📐 Mathematical Core | 數學核心

In V15, the hidden state $S_t$ and normalizer $Z_t$ follow an Exponential Moving Average (EMA) fused with resonance gating:

### 1. Resonance Gating (共振門控)
$$Gate_t = \sigma\left(\tanh(A_{sem} \cdot A_{ctx} \cdot \cos(\theta_{sem} - \theta_{ctx})) \cdot Temp\right)$$

### 2. Memory State Update (記憶狀態更新)
$$S_t = S_{t-1} \cdot dt + \Big((K_t \otimes V_t) \cdot (1 + Gate_t)\Big) \cdot (1 - dt)$$

### 3. Normalizer Update (歸一化器更新)
$$Z_t = Z_{t-1} \cdot dt + K_t \cdot (1 - dt)$$

> *Note: $dt$ is the learned decay rate, bounded between 0.1 and 0.99.*

---

## 🚀 Specifications | 系統規格

| Feature | Specification |
| :--- | :--- |
| **Parameters (參數規模)** | ~190M (Optimized for Consumer GPUs) |
| **Hardware (硬體需求)** | Single **RTX 3060 12GB** for scratch training |
| **Complexity (複雜度)** | $O(N)$ Space & Time |
| **Precision (精度)** | `bfloat16` Mixed Precision + Gradient Checkpointing |
| **Tokenizer (分詞器)** | Custom 16K BPE (High semantic density) |

---

## 🛠️ Quick Start | 快速開始

**1. Install Dependencies | 安裝環境**
```bash
pip install torch transformers tokenizers tqdm pandas matplotlib
```

**2. Prepare Data | 準備數據**
Place your `.txt` corpus into the `data/` folder.
將你的 `.txt` 語料庫放入 `data/` 資料夾中。

**3. Start Training | 啟動訓練**
```bash
python d2-v15-resonance-plus.py
```

**4. Async Monitoring | 異步監控**
Monitor loss curves and resonance gating real-time:
```bash
python monitor_resonance.py
```

---

## 📊 Training Metrics | 監控指標說明

* **Gate (Sparsity):** Reflects the activation ratio of the resonance mechanism. A healthy model stabilizes between `0.3 ~ 0.6`.
* **Ent (Entropy):** Diversity of the gating mechanism. If this drops to 0, "Routing Collapse" has occurred.

---

## 📜 License & Acknowledgements | 授權與致謝

* This project is licensed under the **MIT License**.
* Core wave-function concepts were inspired by the [qllm2](https://github.com/gowrav-vishwakarma/qllm2) project, extensively refactored here with EMA, Memory Filtering, and Dual Bounding.

---
