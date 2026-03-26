import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import math
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
from tokenizers import Tokenizer, models, decoders, trainers, pre_tokenizers

# ==========================================
# 🚀 V15: Resonance Memory Transformer 配置
# 核心特徵：穩定波干涉 + 記憶累積門控 (EMA) + Head-wise Gating
# ==========================================
config = {
    "d_model": 768,          
    "n_heads": 12,           
    "n_layers": 24,          
    "batch_size": 2,         
    "block_size": 768,       
    "accum_steps": 1,       
    "lr": 2e-4,              
    "epochs": 40000,         
    "warmup_steps": 2000,    
    "data_dir": "data",      
    "save_model": "d2_v15_resonance_plus.pth", # 🆕 V15 專屬權重檔
    "vocab_name": "bpe_tokenizer_v12.json",     
    "vocab_size": 16384,                      
    "l1_lambda": 0.02,       
    "balance_lambda": 0.05,
    "entropy_lambda": 0.01   # 🆕 避免 Gate 坍縮的權重
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 V15 啟動中 | 設備: {device}")

# ==========================================
# 1. BPE 分詞器與資料加載
# ==========================================
txt_files = glob.glob(os.path.join(config["data_dir"], "*.txt"))
if not txt_files:
    raise FileNotFoundError(f"❌ 找不到訓練資料！請檢查 {config['data_dir']}")

if os.path.exists(config["vocab_name"]):
    tokenizer = Tokenizer.from_file(config["vocab_name"])
else:
    print(f"⚠️ 訓練新 BPE 分詞器...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=config["vocab_size"], special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"])
    tokenizer.train(files=txt_files, trainer=trainer)
    tokenizer.save(config["vocab_name"])

vocab_size = tokenizer.get_vocab_size()
with open(txt_files[0], 'r', encoding='utf-8') as f:
    data = torch.tensor(tokenizer.encode(f.read()).ids, dtype=torch.long)

def get_batch():
    ix = torch.randint(len(data) - config["block_size"], (config["batch_size"],))
    x = torch.stack([data[i:i+config["block_size"]] for i in ix])
    y = torch.stack([data[i+1:i+config["block_size"]+1] for i in ix])
    return x.to(device), y.to(device)

# ==========================================
# 2. V15 核心架構：共振記憶注意力 (EMA 版)
# ==========================================
class ResonanceMemoryAttentionV15(nn.Module):
    def __init__(self, d_model, bottleneck_rank=128):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.d_head = d_model // self.n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        
        # A1, P1, A2, P2, Decay (5 分量)
        self.bottleneck = nn.Sequential(
            nn.Linear(d_model, bottleneck_rank, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck_rank, self.n_heads * 5, bias=False) 
        )
        
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        self.proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.mem_norm = nn.LayerNorm(self.d_head) # 歸一化器

    def forward(self, x):
        B, L, D = x.shape
        x_norm = self.ln(x)
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        
        # 🌊 參數生成與限幅
        params = self.bottleneck(x).view(B, L, self.n_heads, 5)
        sem_amp, sem_phase, ctx_amp, ctx_phase, decay_raw = params.unbind(-1)
        
        sem_amp, ctx_amp = torch.sigmoid(sem_amp), torch.sigmoid(ctx_amp)
        sem_phase, ctx_phase = torch.tanh(sem_phase) * math.pi, torch.tanh(ctx_phase) * math.pi
        
        # 🌟 Decay 放寬到 0.1 ~ 0.99，允許模型「快速忘記」
        decay_rate = 0.1 + 0.89 * torch.sigmoid(decay_raw) 
        
        # 🌟 Temperature 限制，防止 Gate 進入 Hard Routing
        temp = torch.clamp(self.temperature, 0.1, 2.0)
        cos_diff = torch.cos(sem_phase - ctx_phase)
        interference = torch.tanh(sem_amp * ctx_amp * cos_diff) * temp
        gate = torch.sigmoid(interference) 

        q_f = (F.elu(q.float()) + 1.0).view(B, L, self.n_heads, self.d_head)
        k_f = (F.elu(k.float()) + 1.0).view(B, L, self.n_heads, self.d_head)
        v_f = v.float().view(B, L, self.n_heads, self.d_head)

        states = []
        z_states = [] # 新增 EMA Normalizer 狀態
        prev_s = torch.zeros(B, self.n_heads, self.d_head, self.d_head, device=x.device)
        prev_z = torch.zeros(B, self.n_heads, self.d_head, device=x.device)

        # ==================================================
        # 🚀 極速優化版：將矩陣運算提煉到迴圈外，拯救 GPU 效能！
        # ==================================================
        # 1. 一口氣計算好所有的 K @ V 和 Gate 增益
        # [B, L, heads, d_head, 1] @ [B, L, heads, 1, d_head] -> [B, L, heads, d_head, d_head]
        kv_all = k_f.unsqueeze(-1) @ v_f.unsqueeze(-2)
        gate_ext = gate.view(B, L, self.n_heads, 1, 1)
        kv_all = kv_all * (1.0 + gate_ext)
        
        # 2. 一口氣準備好所有的 Decay 係數
        dt_all = decay_rate.view(B, L, self.n_heads, 1, 1)
        inv_dt_all = 1.0 - dt_all
        
        dt_z_all = decay_rate.view(B, L, self.n_heads, 1)
        inv_dt_z_all = 1.0 - dt_z_all

        states = []
        z_states = [] 
        prev_s = torch.zeros(B, self.n_heads, self.d_head, self.d_head, device=x.device)
        prev_z = torch.zeros(B, self.n_heads, self.d_head, device=x.device)

        # ==================================================
        # 🚀 V15-Fast 極速並行版：消滅 for 迴圈，解放 3060 算力！
        # ==================================================
        # 1. 準備好 K@V 與 Gate 增益 (Memory Filter 依然存在)
        kv_all = k_f.unsqueeze(-1) @ v_f.unsqueeze(-2) # [B, L, heads, d, d]
        gate_ext = gate.view(B, L, self.n_heads, 1, 1)
        
        # 2. 動態過濾：只有 Gate 高的詞，才能進入記憶體
        kv_gated = kv_all * (1.0 + gate_ext)
        
        # 3. 🚀 核心魔法：用 PyTorch C++ 原生的 cumsum 瞬間完成時間維度 (dim=1) 的累積
        kv_states = torch.cumsum(kv_gated, dim=1).transpose(1, 2) # 轉回 [B, heads, L, d, d]
        z_states = torch.cumsum(k_f, dim=1) # [B, L, heads, d_head]
        # ==================================================

        # 計算分子
        q_f_trans = q_f.transpose(1, 2).unsqueeze(-2) # [B, heads, L, 1, d_head]
        out_num = torch.matmul(q_f_trans, kv_states).squeeze(-2) # [B, heads, L, d_head]

        # 🌟 正確的 Denominator Normalization
        den = (q_f * z_states).sum(dim=-1).unsqueeze(-1).transpose(1, 2) + 1e-6
        out = out_num / den 

        # 🌟 安全的 Mem Norm
        out = out.transpose(1, 2).contiguous() # 轉回 [B, L, heads, d_head]
        out = self.mem_norm(out) 
        
        attn_out = out.view(B, L, D).to(x.dtype)
        
        # 🌟 修正：回傳完整的 gate，保留維度 [B, L, heads] 供計算 Balance 與 Entropy
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
        all_gates = torch.stack(gates) # Shape: [layers, B, L, heads]
        
        sparse_loss = all_gates.mean()
        # 🌟 修正：維度適配 (針對 Batch 和 Length 取平均後，算 layers 和 heads 之間的差異)
        balance_loss = all_gates.mean(dim=(1, 2)).var()
        
        # 🌟 修正：安全的 Entropy 計算 (防止 log(0) 或 log(1) 造成的潛在不穩)
        safe_gates = torch.clamp(all_gates, 1e-6, 1.0 - 1e-6)
        entropy_loss = -(safe_gates * torch.log(safe_gates)).mean() 
        
        return logits, sparse_loss, balance_loss, entropy_loss


# ==========================================
# 3. 訓練流程與日誌紀錄 (V15 EMA 終極版)
# ==========================================
model = D2V15Model(vocab_size, config["d_model"], config["n_layers"]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)

def lr_lambda(current_step):
    if current_step < config["warmup_steps"]:
        return float(current_step) / float(max(1, config["warmup_steps"]))
    progress = float(current_step - config["warmup_steps"]) / float(max(1, config["epochs"] - config["warmup_steps"]))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = LambdaLR(optimizer, lr_lambda)
global_step = 0

print(f"🌟 初始化 V15 大腦 | 參數: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
model.train()
pbar = tqdm(initial=global_step, total=config["epochs"], desc="🧠 V15 訓練中", dynamic_ncols=True)

while global_step < config["epochs"]:
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_sparse, total_bal, total_ent = 0, 0, 0, 0 
    
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
        total_bal += balance_loss.item()
        total_ent += entropy_loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step(); scheduler.step(); global_step += 1; pbar.update(1)
    
    current_lr = scheduler.get_last_lr()[0]
    avg_loss = total_loss / config["accum_steps"]
    avg_sparse = total_sparse / config["accum_steps"]
    avg_ent = total_ent / config["accum_steps"]
    
    pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Gate": f"{avg_sparse:.3f}", "Ent": f"{avg_ent:.3f}"})

    if global_step % 10 == 0:
        log_file = "train_log.csv"
        file_exists = os.path.isfile(log_file) and os.path.getsize(log_file) > 0
        with open(log_file, "a", encoding="utf-8") as f:
            if not file_exists: 
                f.write("step,loss,lr,gate,entropy\n") 
            f.write(f"{global_step},{avg_loss:.6f},{current_lr:.2e},{avg_sparse:.6f},{avg_ent:.6f}\n")

    if global_step % 2000 == 0:
        torch.save({'step': global_step, 'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'scheduler_state_dict': scheduler.state_dict()}, config["save_model"])
