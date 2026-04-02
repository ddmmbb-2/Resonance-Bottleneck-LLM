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
    "batch_size": 1,         # 3060 建議從 1 開始，若顯存夠再往上加
    "block_size": 512,       
    "accum_steps": 16,       # 提高累加步數來彌補 batch_size=1 的不穩定
    "lr": 5e-5,              
    "epochs": 100000,         
    "warmup_steps": 500,    
    "data_dir": "data",      
    "bin_data": "corpus_v15.bin", # 🆕 預處理後的二進位檔
    "save_model": "d2_v15_resonance_plus.pth",
    "vocab_name": "bpe_tokenizer_v12.json",     
    "vocab_size": 16384,                      
    "l1_lambda": 0.02,       
    "balance_lambda": 0.05,
    "entropy_lambda": 0.01   
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
# 2. V15 核心架構：共振記憶注意力
# ==========================================
class ResonanceMemoryAttentionV15(nn.Module):
    def __init__(self, d_model, bottleneck_rank=128):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.d_head = d_model // self.n_heads
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
        x_norm = self.ln(x)
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        
        # 🌊 共振參數生成
        params = self.bottleneck(x).view(B, L, self.n_heads, 5)
        sem_amp, sem_phase, ctx_amp, ctx_phase, decay_raw = params.unbind(-1)
        
        sem_amp, ctx_amp = torch.sigmoid(sem_amp), torch.sigmoid(ctx_amp)
        sem_phase, ctx_phase = torch.tanh(sem_phase) * math.pi, torch.tanh(ctx_phase) * math.pi
        
        # 干涉物理模型
        temp = torch.clamp(self.temperature, 0.1, 2.0)
        cos_diff = torch.cos(sem_phase - ctx_phase)
        interference = torch.tanh(sem_amp * ctx_amp * cos_diff) * temp
        gate = torch.sigmoid(interference) 

        q_f = (F.elu(q.float()) + 1.0).view(B, L, self.n_heads, self.d_head)
        k_f = (F.elu(k.float()) + 1.0).view(B, L, self.n_heads, self.d_head)
        v_f = v.float().view(B, L, self.n_heads, self.d_head)

        # 🚀 極速並行版：使用 cumsum 模擬線性累積
        # 注意：目前的 cumsum 是 Decay=1 的狀態。
        kv_all = k_f.unsqueeze(-1) @ v_f.unsqueeze(-2) 
        gate_ext = gate.view(B, L, self.n_heads, 1, 1)
        kv_gated = kv_all * (1.0 + gate_ext)
        
        kv_states = torch.cumsum(kv_gated, dim=1).transpose(1, 2) 
        z_states = torch.cumsum(k_f, dim=1) 

        q_f_trans = q_f.transpose(1, 2).unsqueeze(-2) 
        out_num = torch.matmul(q_f_trans, kv_states).squeeze(-2) 

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
# 3. 訓練循環
# ==========================================
model = D2V15Model(vocab_size, config["d_model"], config["n_layers"]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)

global_step = 0
if os.path.exists(config["save_model"]):
    print(f"♻️ 接續訓練: {config['save_model']}")
    ckpt = torch.load(config["save_model"], map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    global_step = ckpt.get('step', 0)

warmup_scheduler = LambdaLR(optimizer, lambda s: min(1.0, s/config["warmup_steps"]))
auto_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

print(f"🌟 模型參數: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
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
    
    avg_loss = total_loss / config["accum_steps"]
    if global_step < config["warmup_steps"]:
        warmup_scheduler.step()
    elif global_step % 100 == 0:
        auto_scheduler.step(avg_loss)

    global_step += 1
    # --- 插入這段：日誌寫入 CSV ---
    if global_step % 10 == 0:  # 每 10 步紀錄一次
        log_file = "train_log.csv"
        avg_sparse = total_sparse / config["accum_steps"]
        avg_ent = total_ent / config["accum_steps"]
        current_lr = optimizer.param_groups[0]['lr']
        
        file_exists = os.path.isfile(log_file) and os.path.getsize(log_file) > 0
        with open(log_file, "a", encoding="utf-8") as f:
            if not file_exists: 
                f.write("step,loss,lr,gate,entropy\n") # 寫入標頭
            f.write(f"{global_step},{avg_loss:.6f},{current_lr:.2e},{avg_sparse:.6f},{avg_ent:.6f}\n")
    pbar.update(1)
    pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Gate": f"{total_sparse/config['accum_steps']:.3f}"})

    if global_step % 2000 == 0:
        torch.save({
            'step': global_step, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict()
        }, config["save_model"])
        print(f"🚩 Step {global_step} 存檔成功")
