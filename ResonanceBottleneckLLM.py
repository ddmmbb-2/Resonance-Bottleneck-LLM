import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import math
import numpy as np  # 🆕 必須導入
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
from tokenizers import Tokenizer

# ==========================================
# 🚀 V15: Resonance Memory Transformer 配置
# ==========================================
config = {
    "d_model": 768,          
    "n_heads": 12,           
    "n_layers": 12,          
    "batch_size": 4,
    "block_size": 512,
    "accum_steps": 8,       
    "lr": 1e-4,              # ⬅️ 從頭開始，給予標準的 1e-4 動能
    "epochs": 100000,         
    "warmup_steps": 2000,    # ⬅️ 暖身拉長到 2000 步，讓初始權重更平穩過渡
    "data_dir": "data",      
    "bin_data": "corpus_v15_clean.bin", 
    "save_model": "d2_v15_resonance_plus.pth",
    "vocab_name": "bpe_tokenizer_v12.json",     
    "vocab_size": 16384,                      
    "l1_lambda": 0.001,      # ⬅️ 起步保持 0.001，讓守門員去自動調降
    "balance_lambda": 0.05,
    "entropy_lambda": 0.005   
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 V15 啟動中 | 設備: {device}")

# ==========================================
# 1. 資料加載 (高效 Memmap 版)
# ==========================================
if not os.path.exists(config["bin_data"]):
    raise FileNotFoundError(f"❌ 找不到 {config['bin_data']}！請先執行 prepare_data.py")

tokenizer = Tokenizer.from_file(config["vocab_name"])
vocab_size = tokenizer.get_vocab_size()

# 使用 memmap 映射硬碟資料，不佔用 RAM
data = np.memmap(config["bin_data"], dtype=np.uint16, mode='r')

def get_batch():
    # 隨機抽取起始點
    ix = torch.randint(len(data) - config["block_size"], (config["batch_size"],))
    
    # 讀取片段並轉為 Tensor
    x = torch.stack([torch.from_numpy(data[i:i+config["block_size"]].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+config["block_size"]+1].astype(np.int64)) for i in ix])
    
    return x.to(device), y.to(device)

# ==========================================
# 2. V15 核心架構：共振記憶注意力 (自適應衰減版)
# ==========================================
class ResonanceMemoryAttentionV15(nn.Module):
    def __init__(self, d_model, bottleneck_rank=128):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.d_head = d_model // self.n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        
        # Bottleneck 輸出 5 個分量：A1, P1, A2, P2, Decay
        self.bottleneck = nn.Sequential(
            nn.Linear(d_model, bottleneck_rank, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck_rank, self.n_heads * 5, bias=False) 
        )
        
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        self.proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.mem_norm = nn.LayerNorm(self.d_head)

    def forward(self, x):
        B, L, D = x.shape
        x_norm = self.ln(x)
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        
        # 🌊 1. 參數生成
        params = self.bottleneck(x).view(B, L, self.n_heads, 5)
        sem_amp, sem_phase, ctx_amp, ctx_phase, decay_raw = params.unbind(-1)
        
        sem_amp, ctx_amp = torch.sigmoid(sem_amp), torch.sigmoid(ctx_amp)
        sem_phase, ctx_phase = torch.tanh(sem_phase) * math.pi, torch.tanh(ctx_phase) * math.pi
        
        # 🌟 2. 自適應衰減率 (Exponential Decay)
        # 讓模型學會遺忘：0.1 (快速遺忘) ~ 0.99 (長久記憶)
        decay_rate = 0.5 + 0.49 * torch.sigmoid(decay_raw) 
        
        # 3. 干涉門控
        temp = torch.clamp(self.temperature, 0.1, 2.0)
        cos_diff = torch.cos(sem_phase - ctx_phase)
        interference = torch.tanh(sem_amp * ctx_amp * cos_diff) * temp
        gate = torch.sigmoid(interference) 

        q_f = (F.elu(q.float()) + 1.0).view(B, L, self.n_heads, self.d_head)
        k_f = (F.elu(k.float()) + 1.0).view(B, L, self.n_heads, self.d_head)
        v_f = v.float().view(B, L, self.n_heads, self.d_head)

        # 🚀 4. 並行化衰減掃描 (O(N) 終極防護版)
        # 強制轉 FP32 獲取更大的指數安全區間
        decay_rate_f32 = decay_rate.float()
        k_f32 = k_f.float()
        v_f32 = v_f.float()
        gate_f32 = gate.float()

        log_decay = torch.log(decay_rate_f32 + 1e-8)
        cum_log_decay = torch.cumsum(log_decay, dim=1)
        
        # 👉 策略：Clamp log (設立 FP32 絕對安全結界)
        # FP32 的 exp 上限約為 88.7，我們 clamp 在 -85.0。
        # 這樣等一下取負號 exp(-(-85.0)) 時，數值最大只會到 e^85 (約 8e36)，絕對安全！
        cum_log_decay = torch.clamp(cum_log_decay, min=-85.0)
        
        # 👉 策略：避免除法 (用乘法代替)
        decay_factor = torch.exp(cum_log_decay)          # 用於乘回外部
        inv_decay_factor = torch.exp(-cum_log_decay)     # 用於抵銷輸入 (取代除法)

        df_kv = decay_factor.unsqueeze(-1).unsqueeze(-1)
        inv_df_kv = inv_decay_factor.unsqueeze(-1).unsqueeze(-1)
        
        df_z = decay_factor.unsqueeze(-1)
        inv_df_z = inv_decay_factor.unsqueeze(-1)

        ema_weight = (1.0 - decay_rate_f32).unsqueeze(-1).unsqueeze(-1)
        kv_all = (k_f32.unsqueeze(-1) @ v_f32.unsqueeze(-2)) * (1.0 + gate_f32.view(B, L, self.n_heads, 1, 1))
        kv_input = kv_all * ema_weight

        # ⚡ O(N) Prefix Sum，全程無除法！
        kv_states = torch.cumsum(kv_input * inv_df_kv, dim=1) * df_kv
        kv_states = kv_states.transpose(1, 2)
        
        z_input = k_f32 * (1.0 - decay_rate_f32).unsqueeze(-1)
        z_states = torch.cumsum(z_input * inv_df_z, dim=1) * df_z

        # 安全算完後，優雅地轉回 bfloat16 輸出
        kv_states = kv_states.to(x.dtype)
        z_states = z_states.to(x.dtype)

        # 5. 輸出計算
        q_f_trans = q_f.transpose(1, 2).unsqueeze(-2) # [B, H, L, 1, d]
        out_num = torch.matmul(q_f_trans, kv_states).squeeze(-2) # [B, H, L, d]

        # 正確的分母歸一化
        den = (q_f * z_states).sum(dim=-1).unsqueeze(-1).transpose(1, 2) + 1e-6
        out = out_num / den 

        out = out.transpose(1, 2).contiguous() 
        out = self.mem_norm(out) 
        
        attn_out = out.view(B, L, D).to(x.dtype)
        return self.proj(attn_out), gate

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )
        self.ln = nn.LayerNorm(d_model)
    def forward(self, x): return self.net(self.ln(x))

class D2V15Block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = ResonanceMemoryAttentionV15(d_model)
        self.mlp = MLP(d_model)
    def forward(self, x):
        attn_out, gate = self.attn(x)
        x = x + attn_out
        x = x + self.mlp(x)
        return x, gate

class D2V15Model(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([D2V15Block(d_model) for _ in range(n_layers)])
        self.out_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight 
        
    def forward(self, x):
        x = self.embedding(x)
        gates = []
        for block in self.blocks:
            x, gate = checkpoint(block, x, use_reentrant=False)
            gates.append(gate)
            
        logits = self.head(self.out_ln(x))
        all_gates = torch.stack(gates) 
        
        sparse_loss = all_gates.mean()
        balance_loss = all_gates.mean(dim=(1, 2)).var()
        safe_gates = torch.clamp(all_gates, 1e-6, 1.0 - 1e-6)
        entropy_loss = -(safe_gates * torch.log(safe_gates)).mean() 
        
        return logits, sparse_loss, balance_loss, entropy_loss

# ==========================================
# 3. 訓練循環 (熱重啟優化版)
# ==========================================
model = D2V15Model(vocab_size, config["d_model"], config["n_layers"]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)

global_step = 0
if os.path.exists(config["save_model"]):
    print(f"♻️ 接續訓練: {config['save_model']}")
    ckpt = torch.load(config["save_model"], map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    # ⚠️ 這裡最關鍵：load_state_dict 會把舊的 2e-4 讀回來
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    global_step = ckpt.get('step', 0)

# --- 核心修復：強制重置學習率 (必須放在 load_state_dict 之後) ---
for param_group in optimizer.param_groups:
    param_group['lr'] = config["lr"]
    param_group['initial_lr'] = config["lr"]  # ⬅️ 加入這行，徹底洗掉舊的 2e-4

# 🔍 在這裡進行 DEBUG 檢查，確保修復成功
print("-" * 30)
print(f"DEBUG: Config 設定值應該是: {config['lr']}")
print(f"DEBUG: Optimizer 當前實際 LR: {optimizer.param_groups[0]['lr']:.2e}")
if optimizer.param_groups[0]['lr'] > 5e-5:
    print("⚠️ 警告：偵測到 LR 異常偏高！正在進行最後強制修正...")
    for pg in optimizer.param_groups:
        pg['lr'] = config["lr"]
print("-" * 30)

# 紀錄重啟時的起點，用來計算相對暖身
restart_step = global_step 
warmup_scheduler = LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / config["warmup_steps"]))
auto_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=25, verbose=True
)

print(f"🌟 模型參數: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"🚀 已喚醒模型！將從 Step {global_step} 開始進行 {config['warmup_steps']} 步相對暖身。")

model.train()
pbar = tqdm(initial=global_step, total=config["epochs"], desc="🧠 V15 訓練中")

while global_step < config["epochs"]:
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_sparse, total_ent = 0, 0, 0 
    
    for _ in range(config["accum_steps"]):
        xb, yb = get_batch()
        with autocast('cuda', dtype=torch.bfloat16):
            logits, sparse_loss, balance_loss, entropy_loss = model(xb)
            ce_loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            loss = (ce_loss + config["l1_lambda"] * sparse_loss + 
                    config["balance_lambda"] * balance_loss + 
                    config["entropy_lambda"] * entropy_loss) / config["accum_steps"]
        loss.backward()
        total_loss += ce_loss.item()
        total_sparse += sparse_loss.item()
        total_ent += entropy_loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    
    # 確保這兩個平均值在每一步都有計算到
    avg_loss = total_loss / config["accum_steps"]
    avg_ent = total_ent / config["accum_steps"]
    
    
    # ==========================================
    # 🛡️ 自動駕駛守門員 (Autopilot Watchdog)
    # ==========================================
    current_relative_step = global_step - restart_step
    
    if current_relative_step < config["warmup_steps"]:
        warmup_scheduler.step()
    elif global_step % 100 == 0:
        # 1. 正常的平原期自動降速 (Loss 降不下來時觸發)
        auto_scheduler.step(avg_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 🚨 2. 防燒毀機制 (Loss 暴衝警報)
        # 如果 Loss 突然飆高超過 4.8，且學習率還不算太低，強制直接砍半
        if avg_loss > 4.8 and current_lr > 1e-5:
            print(f"\n🚨 [守門員] 偵測到 Loss 暴衝 ({avg_loss:.3f})！緊急將 LR 砍半至 {current_lr * 0.5:.2e}")
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.5
                
        # ⚠️ 3. 防變笨機制 (模式坍塌警報)
        # 如果熵值跌破安全線 0.30，代表模型快要變成複讀機了
        if avg_ent < 0.30:
            # 自動放寬 L1 懲罰 (每次打 9 折，最低降到 1e-5)
            config["l1_lambda"] = max(1e-5, config["l1_lambda"] * 0.90)
            # 自動提高多樣性獎勵
            config["entropy_lambda"] = min(0.02, config["entropy_lambda"] * 1.05)
            print(f"\n⚠️ [守門員] 熵值過低 ({avg_ent:.3f})！自動放寬 L1 懲罰至 {config['l1_lambda']:.5f}")

    global_step += 1
    
    # --- 關鍵：CSV 日誌紀錄 ---
    if global_step % 10 == 0:
        log_file = "train_log.csv"
        avg_sparse = total_sparse / config["accum_steps"]
        avg_ent = total_ent / config["accum_steps"]
        current_lr = optimizer.param_groups[0]['lr']
        
        file_exists = os.path.isfile(log_file) and os.path.getsize(log_file) > 0
        with open(log_file, "a", encoding="utf-8") as f:
            if not file_exists: 
                f.write("step,loss,lr,gate,entropy\n")
            f.write(f"{global_step},{avg_loss:.6f},{current_lr:.2e},{avg_sparse:.6f},{avg_ent:.6f}\n")

    pbar.update(1)
    pbar.set_postfix({
        "Loss": f"{avg_loss:.4f}", 
        "LR": f"{optimizer.param_groups[0]['lr']:.2e}", 
        "Gate": f"{total_sparse/config['accum_steps']:.3f}"
    })

    # --- 自動存檔 (增加里程碑備份) ---
    if global_step % 5000 == 0:
        # 定義存檔內容
        checkpoint = {
            'step': global_step, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict()
        }
        
        # 1. 備份時光機 (帶編號，不會被覆蓋)
        backup_name = f"d2_v15_step_{global_step}.pth"
        torch.save(checkpoint, backup_name)
        
        # 2. 更新最新權重 (方便接續訓練用)
        torch.save(checkpoint, config["save_model"])
        
        print(f"🚩 Step {global_step} 存檔成功！已建立備份：{backup_name}")