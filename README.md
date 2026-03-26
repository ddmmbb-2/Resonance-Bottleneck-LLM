
# 🌊 Resonance Memory Transformer (D2-V15)

**Resonance Memory Transformer (共振記憶轉換器)** 是一個探索 **「波干涉隱式門控」** 與 **「線性注意力記憶過濾」** 結合的實驗性大型語言模型 (LLM) 架構。

本專案從 `D2-Subset-LLM` 演進而來，在 **V15** 版本中實現了重大突破：捨棄了傳統的無狀態線性注意力（Stateless Linear Attention），引入了受生物神經學啟發的 **EMA 記憶衰減機制 (Memory Decay)** 與 **雙重限幅共振系統 (Dual-Bounded Resonance)**，在保持 $O(N)$ 線性複雜度的同時，賦予模型強大的長文本邏輯提煉能力。

---

## 🧠 核心理念：為什麼需要「共振記憶」？

傳統的線性注意力機制容易隨著序列長度增加而產生「記憶模糊」與「數值爆炸」。而傳統的 MoE (Mixture of Experts) 則需要龐大且離散的 Router 參數。

**V15 架構透過以下機制解決這些痛點：**

1. **波干涉隱式路由 (Wave-Interference Routing)：**
   不使用顯式的專家網路。模型將上下文特徵映射為「波」的**振幅 (Amplitude)** 與 **相位 (Phase)**。透過不同特徵間的干涉（相長或相消），自然湧現出針對特定語境的激活狀態（Gate）。
2. **多頭記憶過濾器 (Head-wise Memory Filter)：**
   每個 Attention Head 擁有獨立的共振頻率與衰減率（Decay）。模型不再只是「決定要看什麼」，而是透過動態的 Gate 控制「哪些歷史記憶需要被保留與放大」，哪些需要被遺忘。

---

## ✨ V15 突破性架構升級 (The "++" Features)

* **EMA 記憶衰減 (Exponential Moving Average Decay)：**
  將單純的記憶累加 ($cumsum$) 升級為帶有遺忘機制的 RNN 狀態更新，賦予模型「近因偏差」並防止記憶堆積崩塌。
* **嚴格的數學對齊 (Denominator Normalization)：**
  同步使用 EMA 更新分母 (Normalizer) 狀態 $Z_t$，徹底解決線性注意力中極易發生的 Scale 飄移問題。
* **雙重限幅穩定器 (Dual Bounding)：**
  使用 $\text{Sigmoid}$ 限制振幅，使用 $\text{Tanh}$ 將相位錨定於 $[-\pi, \pi]$，並對最終干涉進行阻尼運算，確保梯度流動極度平滑。
* **熵正則化 (Entropy Loss)：**
  引入 Gate Entropy 懲罰項，防止門控機制坍縮（Collapse）為全 0 或全 1 的退化狀態。

---

## 📐 數學核心 (Mathematical Formulation)

在 V15 中，隱藏狀態 $S_t$ 與分母 $Z_t$ 的更新遵循嚴格的指數移動平均（EMA）結合共振門控（$Gate_t$）：

**1. 共振門控計算 (Resonance Gating):**
$$Gate_t = \sigma\left(\tanh(A_{sem} \cdot A_{ctx} \cdot \cos(\theta_{sem} - \theta_{ctx})) \cdot Temp\right)$$

**2. 記憶狀態更新 (Memory State Update):**
$$S_t = S_{t-1} \cdot dt + \Big((K_t \otimes V_t) \cdot (1 + Gate_t)\Big) \cdot (1 - dt)$$

**3. 歸一化器更新 (Normalizer Update):**
$$Z_t = Z_{t-1} \cdot dt + K_t \cdot (1 - dt)$$

*(註：$dt$ 為模型自適應學習出的衰減率 Decay，限制於 0.1 ~ 0.99 之間)*

---

## 🚀 系統規格與效能

* **參數規模 (Parameters)：** ~190M Parameters (甜甜圈區間，完美適配消費級 GPU)
* **硬體需求：** 單張 **NVIDIA RTX 3060 12GB** 即可進行全量從頭訓練 (Train from scratch)。
* **精度與優化：** 原生支援 `bfloat16` 混合精度與 Gradient Checkpointing。
* **分詞器 (Tokenizer)：** 自定義 16K BPE 字典，極大化文本壓縮率與語義密度。
* **計算複雜度：** 推理與顯存佔用均為 $O(N)$。

---

## 🛠️ 快速開始 (Quick Start)

**1. 安裝依賴環境**
```bash
pip install torch transformers tokenizers tqdm pandas matplotlib
```

**2. 準備訓練數據**
將你的 `.txt` 語料庫（如 Wikipedia、程式碼等）放入 `data/` 資料夾中。

**3. 啟動 V15 模型訓練**
```bash
python d2-v15-resonance-plus.py
```

**4. 啟動異步訓練監控 (推薦)**
在另一個終端機執行監控腳本，即時觀測 Loss 大跳水與 Gate 共振的過程：
```bash
python monitor_resonance.py
```

---

## 📊 監控指標說明

訓練過程中，除了標準的 Cross-Entropy Loss，我們引入了兩個觀察模型「心智健康度」的關鍵指標：
* **Gate (Sparsity):** 反映共振機制的激活比例。健康的模型通常會在初期波動，隨後穩定在 `0.3 ~ 0.6` 之間，展現出良好的抗噪特徵。
* **Ent (Entropy):** 門控的資訊熵。若趨近於 0，代表發生了 Routing Collapse，模型失去了記憶篩選的多樣性。

---

## 📜 授權協議與致謝 (License & Acknowledgements)

* 本專案採用 **MIT License** 授權。
* 本架構的核心波函數與相位概念，最初靈感源自 [qllm2](https://github.com/gowrav-vishwakarma/qllm2) 專案，並在此基礎上進行了深度的數學重構與架構演進（引入 EMA、Memory Filter 與雙重限幅）。
```

