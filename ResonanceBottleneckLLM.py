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
# 🚀 V16: 強化局部感知與長程衰減優化版 配置
# ==========================================
config = {
    "d_model": 768,          
    "n_heads": 12,           
    "n_layers": 12,          
    "batch_size": 8,         # ⬅️ 稍微調大 Batch
    "block_size": 512,
    "accum_steps": 4,        # ⬅️ 配合 Batch Size，保持梯度穩定
    "lr": 1e-4,            # ⬅️ 稍微調高起步 LR，觀察是否會噴
    "epochs": 100000,          
    "warmup_steps": 2000,    # ⬅️ 調整暖身步數
    "data_dir": "data",      
    "bin_data": "corpus_v15_clean.bin", 
    "save_model": "d2_v16_resonance_pro.pth",
    "vocab_name": "bpe_tokenizer_v12.json",     
    "vocab_size": 16384,                      
    "bottleneck_rank": 256,  # ⬅️ 增加瓶頸層寬度
    "l1_lambda": 0.0,        # 🟢 前期關閉
    "balance_lambda": 0.0,   # 🟢 前期關閉
    "entropy_lambda": 0.0    # 🟢 前期關閉
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 V16 啟動中 | 設備: {device}")

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
# 2. V16 核心架構：強化局部感知與長程衰減優化版
# ==========================================
class ResonanceMemoryAttentionV16(nn.Module):
    def __init__(self, d_model, bottleneck_rank=256):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.d_head = d_model // self.n_heads
        
        # 🟢 局部感知層 (Causal Conv1d, padding=2 防止偷看未來)
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, groups=d_model)
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        
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
        
        # 🟢 修正 1: Local Conv 殘差連接
        x_t = x.transpose(1, 2)
        conv_out = self.local_conv(x_t)[..., :-2] # 截斷未來資訊
        x_conv = x + conv_out.transpose(1, 2)     
        x_norm = self.ln(x_conv)
        
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        
        params = self.bottleneck(x_norm).view(B, L, self.n_heads, 5)
        sem_amp, sem_phase, ctx_amp, ctx_phase, decay_raw = params.unbind(-1)
        
        sem_amp, ctx_amp = torch.sigmoid(sem_amp), torch.sigmoid(ctx_amp)
        sem_phase, ctx_phase = torch.tanh(sem_phase) * math.pi, torch.tanh(ctx_phase) * math.pi
        
        # 🟢 修正 2: 更強的忘記能力 (範圍 0.3 ~ 0.95)
        decay_rate = 0.3 + 0.65 * torch.sigmoid(decay_raw) 
        
        # 🟢 修正 3: 拿掉 Gate 的 +0.1 Bias
        cos_diff = torch.cos(sem_phase - ctx_phase)
        gate = torch.sigmoid((sem_amp * ctx_amp * cos_diff) * self.temperature) 

        q_f = (F.elu(q.float()) + 1.0).view(B, L, self.n_heads, self.d_head)
        k_f = (F.elu(k.float()) + 1.0).view(B, L, self.n_heads, self.d_head)
        v_f = v.float().view(B, L, self.n_heads, self.d_head)

        dt = (1.0 - decay_rate).unsqueeze(-1).unsqueeze(-1)
        
        # 🟢 修正 4: kv_input 只乘 gate，並使用 clamp 防爆
        kv_input = (k_f.unsqueeze(-1) @ v_f.unsqueeze(-2)) * gate.unsqueeze(-1).unsqueeze(-1)
        kv_input = kv_input * dt 
        kv_input = torch.clamp(kv_input, min=-5.0, max=5.0) 
        
        log_decay = torch.log(decay_rate).unsqueeze(-1).unsqueeze(-1) 
        cum_log_decay = torch.cumsum(log_decay, dim=1)
        
        decay_factor = torch.exp(cum_log_decay)
        safe_df = decay_factor + 1e-8
        
        kv_states = torch.cumsum(kv_input / safe_df, dim=1) * decay_factor
        kv_states = kv_states.transpose(1, 2) 

        z_input = k_f * (1.0 - decay_rate).unsqueeze(-1)
        z_df = decay_factor.squeeze(-1) + 1e-8
        z_states = torch.cumsum(z_input / z_df, dim=1) * decay_factor.squeeze(-1)
        
        z_states = torch.clamp(z_states, -1e4, 1e4)

        q_f_trans = q_f.transpose(1, 2).unsqueeze(-2) 
        out_num = torch.matmul(q_f_trans, kv_states).squeeze(-2) 

        # 🟢 修正 5: 分母穩定度提升至 1e-4
        den = (q_f * z_states).sum(dim=-1).unsqueeze(-1).transpose(1, 2) + 1e-4
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

class D2V16Block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = ResonanceMemoryAttentionV16(d_model, bottleneck_rank=config["bottleneck_rank"])
        self.mlp = MLP(d_model)
    def forward(self, x):
        attn_out, gate = self.attn(x)
        x = x + attn_out
        x = x + self.mlp(x)
        return x, gate

class D2V16Model(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([D2V16Block(d_model) for _ in range(n_layers)])
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
model = D2V16Model(vocab_size, config["d_model"], config["n_layers"]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)

global_step = 0
if os.path.exists(config["save_model"]):
    print(f"♻️ 接續訓練: {config['save_model']}")
    ckpt = torch.load(config["save_model"], map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    # ⚠️ 這裡最關鍵：load_state_dict 會把舊的讀回來
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    global_step = ckpt.get('step', 0)

# --- 核心修復：強制重置學習率 (必須放在 load_state_dict 之後) ---
for param_group in optimizer.param_groups:
    param_group['lr'] = config["lr"]
    param_group['initial_lr'] = config["lr"]  # ⬅️ 徹底洗掉舊的設定

# 🔍 在這裡進行 DEBUG 檢查，確保修復成功
print("-" * 30)
print(f"DEBUG: Config 設定值應該是: {config['lr']}")
print(f"DEBUG: Optimizer 當前實際 LR: {optimizer.param_groups[0]['lr']:.2e}")
if optimizer.param_groups[0]['lr'] > 2e-4:
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
print(f"🚀 已喚醒 V16 模型！將從 Step {global_step} 開始進行 {config['warmup_steps']} 步相對暖身。")

model.train()
pbar = tqdm(initial=global_step, total=config["epochs"], desc="🧠 V16 訓練中")
has_decayed_42 = False
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
    # 🛡️ 自動駕駛守門員 (Watchdog)
    # ==========================================
    current_relative_step = global_step - restart_step
    
    if current_relative_step < config["warmup_steps"]:
        warmup_scheduler.step()
        
    # 🟢 修正 4: 收尾降速 (確保只觸發一次)
    elif avg_loss < 4.2 and not has_decayed_42:
        print(f"\n📉 [階段降速] Loss 跌破 4.2！收尾降速啟動，LR 乘以 0.7")
        for pg in optimizer.param_groups:
            pg['lr'] *= 0.7
        has_decayed_42 = True  # ⬅️ 鎖定開關，避免下一次迴圈重複降速

    # 後期解鎖正規化 (Loss 4.0 階段)
    elif global_step > 5000 and avg_loss < 4.0:
        if config["l1_lambda"] == 0.0:
            print("\n🚀 [階段解鎖] 模型已具備基礎語意，開始引入稀疏度與平衡約束！")
            config["l1_lambda"] = 0.0001
            config["entropy_lambda"] = 0.005
            config["balance_lambda"] = 0.02
            
        elif global_step % 100 == 0 and avg_ent < 0.30:
            config["l1_lambda"] = max(1e-5, config["l1_lambda"] * 0.95)
            config["entropy_lambda"] = min(0.015, config["entropy_lambda"] * 1.05)
            
    # 防燒毀機制 (暴衝檢查)
    elif global_step % 100 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        if avg_loss > 7.5 and current_lr > 1e-5:
            print(f"\n🚨 [守門員] 偵測到 Loss 暴衝！緊急降速")
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.5
            print(f"\n⚠️ [守門員週期調整] 熵值低 ({avg_ent:.3f})：L1->{config['l1_lambda']:.5f}, Ent->{config['entropy_lambda']:.5f}")

    global_step += 1
    
    # --- 關鍵：CSV 日誌紀錄 ---
    if global_step % 10 == 0:
        log_file = "train_log_v16.csv"  # ⬅️ V16 獨立日誌
        avg_sparse = total_sparse / config["accum_steps"]
        # avg_ent 已經在上面算過，這邊不用重複算
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
    if global_step % 2000 == 0:
        # 定義存檔內容
        ckpt = {
            'step': global_step, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict()
        }
        
        # 1. 備份時光機 (帶編號，不會被覆蓋)
        backup_name = f"d2_v16_step_{global_step}.pth"
        torch.save(ckpt, backup_name)  
        
        # 2. 更新最新權重 (方便接續訓練用)
        torch.save(ckpt, config["save_model"])  
        
        print(f"🚩 Step {global_step} 存檔成功！已建立備份：{backup_name}")