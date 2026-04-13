import os
import glob
import numpy as np
import re  # 🆕 引入正則表達式
from tqdm import tqdm
from tokenizers import Tokenizer

# 配置
data_dir = "data"
output_file = "corpus_v15_clean.bin" # 🆕 改個名字，保留舊的對比
vocab_file = "bpe_tokenizer_v12.json"

tokenizer = Tokenizer.from_file(vocab_file)
txt_files = glob.glob(os.path.join(data_dir, "*.txt"))

# 🆕 資料清洗函數：移除模型不需要的雜訊
def clean_text(text):
    # 1. 移除維基百科的連結與檔案格式 (例如 [[File:xxx]], [[Category:xxx]])
    text = re.sub(r'\[\[(?:File|Category|Image|Special|User):.*?\]\]', '', text)
    # 2. 移除過長的亂碼字串 (例如長度超過 10 的純字母數字組合，通常是 ID)
    text = re.sub(r'\b[A-Za-z0-9]{10,}\b', '', text)
    # 3. 移除重複的符號 (例如大量的 ......... 或 --------)
    text = re.sub(r'[\.\-\=\_\*]{4,}', ' ', text)
    # 4. 移除常見的垃圾 HTML 標籤
    text = re.sub(r'<.*?>', '', text)
    # 5. 只保留合理的內容：過短的行（例如只有 1-2 個字）通常是噪聲
    lines = text.split('\n')
    filtered_lines = [line.strip() for line in lines if len(line.strip()) > 5]
    return "\n".join(filtered_lines)

print(f"🚀 開始「精煉」處理 {len(txt_files)} 個文件...")

with open(output_file, "ab") as bin_file:
    for fpath in txt_files:
        fname = os.path.basename(fpath)
        print(f"正在清洗並轉換: {fname}")
        
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            chunk_text = ""
            for i, line in enumerate(f):
                chunk_text += line
                
                if i % 10000 == 0 and i > 0:
                    # 🆕 在編碼前先執行清洗
                    refined_text = clean_text(chunk_text)
                    tokens = tokenizer.encode(refined_text).ids
                    if tokens:
                        np.array(tokens, dtype=np.uint16).tofile(bin_file)
                    chunk_text = "" 
            
            if chunk_text:
                refined_text = clean_text(chunk_text)
                tokens = tokenizer.encode(refined_text).ids
                if tokens:
                    np.array(tokens, dtype=np.uint16).tofile(bin_file)

print(f"✅ 精煉完成！高品質資料已存至: {output_file}")