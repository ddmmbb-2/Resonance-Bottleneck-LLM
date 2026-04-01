import os
import glob
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer

# 配置
data_dir = "data"
output_file = "corpus_v15.bin"
vocab_file = "bpe_tokenizer_v12.json"

tokenizer = Tokenizer.from_file(vocab_file)
txt_files = glob.glob(os.path.join(data_dir, "*.txt"))

# 使用 numpy 的 memmap 來高效寫入
all_token_ids = []

print(f"🚀 開始處理 {len(txt_files)} 個文件...")

for fpath in tqdm(txt_files):
    with open(fpath, 'r', encoding='utf-8') as f:
        # 如果單個文件還是太大，這裡可以改用迴圈讀取 line
        text = f.read()
        tokens = tokenizer.encode(text).ids
        all_token_ids.extend(tokens)

# 轉化為 numpy 陣列並存檔
ids_array = np.array(all_token_ids, dtype=np.uint16)
ids_array.tofile(output_file)

print(f"✅ 轉換完成！生成檔案: {output_file}")
print(f"📊 總 Token 數: {len(ids_array)}")