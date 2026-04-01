import pandas as pd
import os
import glob
import json
import random
import re
from tqdm import tqdm

try:
    from opencc import OpenCC
    cc = OpenCC('s2t') # 簡體轉繁體
except ImportError:
    cc = None

# =================配置區=================
DATA_DIR = "raw_data"      
OUTPUT_FILE = "data/v16_training_data.txt"
MIN_LINE_LEN = 15          
MAX_LINE_LEN = 3000        
DEDUPLICATE = True         
# ========================================

def clean_text(text):
    if not isinstance(text, str) or not text.strip(): 
        return ""
    if cc: 
        text = cc.convert(text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def process_factory_v16():
    all_lines = set() if DEDUPLICATE else []
    files = glob.glob(os.path.join(DATA_DIR, "*.parquet")) + \
            glob.glob(os.path.join(DATA_DIR, "*.jsonl")) + \
            glob.glob(os.path.join(DATA_DIR, "*.txt"))
    
    print(f"🏭 數據工廠 V16.2 啟動！偵測到 {len(files)} 個檔案")
    total_extracted = 0
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f"📦 處理中: {file_name}")
        file_count = 0
        
        # --- 1. 處理 Parquet (針對 Wikipedia 優化) ---
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            # 優先級：markdown > text > html
            col = 'markdown' if 'markdown' in df.columns else \
                  ('text' if 'text' in df.columns else \
                  ('html' if 'html' in df.columns else df.columns[0]))
            
            print(f"  🎯 使用欄位: [{col}]")
            for full_content in df[col].dropna().astype(str):
                paragraphs = full_content.split('\n')
                for p in paragraphs:
                    p = re.sub(r'^#+\s+', '', p) # 移除 Markdown 標題符號
                    cleaned = clean_text(p)
                    if MIN_LINE_LEN < len(cleaned) < MAX_LINE_LEN:
                        if DEDUPLICATE: all_lines.add(cleaned)
                        else: all_lines.append(cleaned)
                        file_count += 1

        # --- 2. 處理 JSONL (針對 CSC 或其他數據) ---
        elif file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        content = data.get('text') or data.get('content') or data.get('target')
                        if content:
                            cleaned = clean_text(content)
                            if MIN_LINE_LEN < len(cleaned) < MAX_LINE_LEN:
                                if DEDUPLICATE: all_lines.add(cleaned)
                                else: all_lines.append(cleaned)
                                file_count += 1
                    except: continue

        # --- 3. 處理普通 TXT (小說、代碼等) ---
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # 這裡改用單換行切割，增加段落密度
                paragraphs = content.split('\n')
                for p in paragraphs:
                    cleaned = clean_text(p)
                    if MIN_LINE_LEN < len(cleaned) < MAX_LINE_LEN:
                        if DEDUPLICATE: all_lines.add(cleaned)
                        else: all_lines.append(cleaned)
                        file_count += 1
        
        print(f"✅ {file_name} 完成，提取了 {file_count} 條優質段落")
        total_extracted += file_count

    # --- 結束迴圈後的處理 ---
    final_data = list(all_lines)
    print(f"🌀 正在隨機洗牌（Shuffle）... 最終不重複條目: {len(final_data)}")
    random.shuffle(final_data)

    print(f"💾 正在寫入: {OUTPUT_FILE}")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for line in tqdm(final_data):
            f.write(line + "\n\n") 

    print(f"🎉 任務完成！V16 訓練資料已就緒。")

if __name__ == "__main__":
    process_factory_v16()