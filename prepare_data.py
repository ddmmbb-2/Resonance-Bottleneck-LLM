import os
import glob
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer

# 配置
data_dir = "data"
output_file = "corpus_v15.bin"
vocab_file = "bpe_tokenizer_v12.json"

if not os.path.exists(vocab_file):
    raise FileNotFoundError(f"❌ 找不到分詞器檔案 {vocab_file}")

tokenizer = Tokenizer.from_file(vocab_file)
txt_files = glob.glob(os.path.join(data_dir, "*.txt"))

# 如果舊的 bin 檔存在，先刪除，避免數據重複追加
if os.path.exists(output_file):
    os.remove(output_file)

print(f"🚀 開始處理 {len(txt_files)} 個文件 (分塊模式)...")

# 以二進位追加模式打開輸出的 bin 檔案
with open(output_file, "ab") as bin_file:
    for fpath in txt_files:
        print(f"正在處理: {os.path.basename(fpath)}")
        
        # 為了避免一次讀取 825MB，我們逐行讀取
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            chunk_text = ""
            # 每累積 50,000 行處理一次，平衡速度與記憶體
            for i, line in enumerate(f):
                chunk_text += line
                
                if i % 10000 == 0 and i > 0:
                    # Tokenize 這一塊
                    tokens = tokenizer.encode(chunk_text).ids
                    if tokens:
                        # 轉為 uint16 並寫入硬碟
                        np.array(tokens, dtype=np.uint16).tofile(bin_file)
                    chunk_text = "" # 清空記憶體
            
            # 處理最後剩餘的部分
            if chunk_text:
                tokens = tokenizer.encode(chunk_text).ids
                if tokens:
                    np.array(tokens, dtype=np.uint16).tofile(bin_file)

print(f"✅ 全部轉換完成！")
print(f"📊 最終檔案路徑: {os.path.abspath(output_file)}")