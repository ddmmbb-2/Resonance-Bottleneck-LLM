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
# 🌟 新增 BPE 相關套件
from tokenizers import Tokenizer, models, decoders, trainers, pre_tokenizers

# ==========================================
# 🚀 V12 進化版配置 (180M 級別) + QLLM2 波干涉 + BPE + 權重綁定
# 完美適配 RTX 3060 12GB 
# ==========================================
config = {
    "d_model": 768,          
    "n_heads": 12,           
    "n_layers": 24,          
    "batch_size": 2,         
    "block_size": 768,       
    "accum_steps": 12,       
    "lr": 2e-4,              
    "epochs": 40000,         
    "warmup_steps": 2000,    
    "data_dir": "data",      
    "save_model": "d2_v13_resonance.pth",    # 🆕 換新檔名，避免跟 V11 衝突
    "vocab_name": "bpe_tokenizer_v12.json",   # 🆕 BPE 專屬分詞器設定檔
    "vocab_size": 16384,                      # 🆕 設定 BPE 詞表大小 (保護 3060 VRAM)
    "l1_lambda": 0.02,       
    "balance_lambda": 0.05
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 運行設備: {device}")
if device == "cuda":
    print(f"🖥️ 顯卡型號: {torch.cuda.get_device_name(0)}")
    print(f"⚡ 支援 BF16: {torch.cuda.is_bf16_supported()}")

# ==========================================
# 1. 資料讀取與全新 BPE 詞表建立
# ==========================================
print(f"🔍 正在掃描 {config['data_dir']} 資料夾...")
txt_files = glob.glob(os.path.join(config["data_dir"], "*.txt"))
if not txt_files:
    raise FileNotFoundError(f"❌ 在 {config['data_dir']} 資料夾中找不到任何 .txt 檔案！請確保有訓練資料。")

all_text = ""
for file_path in txt_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        all_text += f.read() + "\n\n" 

if os.path.exists(config["vocab_name"]):
    print(f"🔒 偵測到 BPE 分詞器 {config['vocab_name']}，直接載入！")
    tokenizer = Tokenizer.from_file(config["vocab_name"])
else:
    print(f"⚠️ 找不到舊字典，正在為 V12 訓練全新 BPE 分詞器 (詞表大小: {config['vocab_size']})...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=config["vocab_size"],
        special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    tokenizer.train(files=txt_files, trainer=trainer)
    tokenizer.save(config["vocab_name"])
    print("✅ BPE 分詞器訓練完成並存檔！")

vocab_size = tokenizer.get_vocab_size()
print(f"📊 最終詞表大小: {vocab_size}")

print("⏳ 文本轉碼中，請稍候...")
encoded = tokenizer.encode(all_text)
data = torch.tensor(encoded.ids, dtype=torch.long)
print(f"📉 文本壓縮對比：原本 {len(all_text)} 個字元，壓縮後變為 {len(data)} 個 Tokens！(省下大量算力)")

def get_batch():
    ix = torch.randint(len(data) - config["block_size"], (config["batch_size"],))
    x = torch.stack([data[i:i+config["block_size"]] for i in ix])
    y = torch.stack([data[i+1:i+config["block_size"]+1] for i in ix])
    return x.to(device), y.to(device)

# ==========================================
# 2. V12 模型架構 (QLLM2 波干涉 + 權重綁定)
# ==========================================
class CausalGatedD2Attention(nn.Module):
    def __init__(self, d_model, bottleneck_rank=128): # 🆕 提升至 128
        super().__init__()
        self.n_heads = config["n_heads"]
        self.d_head = d_model // self.n_heads
        
        # 特徵主幹道
        self.qkv = nn.Linear(d_model, d_model * 3)
        
        # 🆕 V13-Stable 共振瓶頸層
        self.sem_down = nn.Linear(d_model, bottleneck_rank, bias=False)
        self.sem_up = nn.Linear(bottleneck_rank, d_model * 2, bias=False)
        
        self.ctx_down = nn.Linear(d_model, bottleneck_rank, bias=False)
        self.ctx_up = nn.Linear(bottleneck_rank, d_model * 2, bias=False)
        
        # 溫度參數，防止 Sigmoid 飽和
        self.temperature = nn.Parameter(torch.ones(1) * 0.5) 
        
        self.proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        
    def forward(self, x):
        B, L, D = x.shape
        x_norm = self.ln(x)
        
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        
        # 🌊 穩定的波參數生成
        # 語義波
        sem_feat = F.silu(self.sem_down(x))
        sem_amp, sem_phase = self.sem_up(sem_feat).chunk(2, dim=-1)
        sem_amp = F.softplus(sem_amp)
        sem_phase = torch.tanh(sem_phase) * math.pi # 🆕 相位錨定在 [-π, π]
        
        # 上下文波
        ctx_feat = F.silu(self.ctx_down(x))
        ctx_amp, ctx_phase = self.ctx_up(ctx_feat).chunk(2, dim=-1)
        ctx_amp = F.softplus(ctx_amp)
        ctx_phase = torch.tanh(ctx_phase) * math.pi # 🆕 相位錨定在 [-π, π]
        
        # 🌊 計算相干涉 (Coherence)
        # 加上溫度控制與殘差門控
        interference = (sem_amp * ctx_amp * torch.cos(sem_phase - ctx_phase)) * self.temperature
        gate = torch.sigmoid(interference)
        
        # 🆕 殘差門控：k = k + k * gate (確保基礎梯度不消失)
        k = k * (1.0 + gate) 
        
        # ... 後續 Linear Attention 計算保持不變 ...
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        
        q_f, k_f, v_f = q.float(), k.float(), v.float()
        q_f = F.elu(q_f) + 1.0
        k_f = F.elu(k_f) + 1.0
        
        kv_state = k_f.unsqueeze(-1) * v_f.unsqueeze(-2)  
        kv_cumsum = torch.cumsum(kv_state, dim=2) 
        out_num = torch.matmul(q_f.unsqueeze(-2), kv_cumsum).squeeze(-2) 
        k_cumsum = torch.cumsum(k_f, dim=2)
        out_den = (q_f * k_cumsum).sum(dim=-1, keepdim=True) + 1e-6
        
        attn_out = (out_num / out_den).to(x.dtype)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        
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

class D2V12Block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = CausalGatedD2Attention(d_model)
        self.mlp = MLP(d_model)
    def forward(self, x):
        attn_out, gate = self.attn(x)
        x = x + attn_out
        x = x + self.mlp(x)
        return x, gate

class D2V12Model(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([D2V12Block(d_model) for _ in range(n_layers)])
        self.out_ln = nn.LayerNorm(d_model)
        
        # ✨ 為了 Weight Tying，不使用 bias
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # ✨ 關鍵魔法：Weight Tying (權重綁定)，替 3060 狂省 VRAM！
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
        balance_loss = all_gates.mean(dim=(0, 1, 2)).var() 
        return logits, sparse_loss, balance_loss

# ==========================================
# 3. 初始化與完美斷點續傳 (Checkpoint)
# ==========================================
model = D2V12Model(vocab_size, config["d_model"], config["n_layers"]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)

def lr_lambda(current_step):
    if current_step < config["warmup_steps"]:
        return float(current_step) / float(max(1, config["warmup_steps"]))
    progress = float(current_step - config["warmup_steps"]) / float(max(1, config["epochs"] - config["warmup_steps"]))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = LambdaLR(optimizer, lr_lambda)

global_step = 0

if os.path.exists(config["save_model"]):
    checkpoint_data = torch.load(config["save_model"], map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint_data:
        model.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        global_step = checkpoint_data['step']
        print(f"✅ 完美接續！從 Step {global_step} 恢復 V12 完整訓練狀態！")
    else:
        model.load_state_dict(checkpoint_data)
        print(f"✅ 載入 V12 初代權重！")
else:
    print(f"🌟 初始化全新 V12 大腦！(參數規模約 {sum(p.numel() for p in model.parameters())/1e6:.1f}M)")

# ==========================================
# 4. 訓練迴圈 (BF16 + TQDM 進度條)
# ==========================================
print("🚀 V12 (BPE + 權重綁定) 轉生計畫啟動！")
model.train()

pbar = tqdm(initial=global_step, total=config["epochs"], desc="🧠 V12 訓練中", dynamic_ncols=True)

while global_step < config["epochs"]:
    optimizer.zero_grad(set_to_none=True)
    
    total_loss = 0
    total_sparse = 0
    total_bal = 0
    
    for _ in range(config["accum_steps"]):
        xb, yb = get_batch()
        
        with autocast('cuda', dtype=torch.bfloat16):
            logits, sparse_loss, balance_loss = model(xb)
            ce_loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            loss = ce_loss + config["l1_lambda"] * sparse_loss + config["balance_lambda"] * balance_loss
            loss = loss / config["accum_steps"]
            
        loss.backward()
        
        total_loss += ce_loss.item() 
        total_sparse += sparse_loss.item()
        total_bal += balance_loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

    optimizer.step()
    scheduler.step()

    global_step += 1
    pbar.update(1)
    
    current_lr = scheduler.get_last_lr()[0] 
    avg_loss = total_loss / config["accum_steps"]
    avg_sparse = total_sparse / config["accum_steps"]
    avg_bal = total_bal / config["accum_steps"]
    
    pbar.set_postfix({
        "Loss": f"{avg_loss:.4f}",
        "LR": f"{current_lr:.2e}",
        "Gate": f"{avg_sparse:.3f}"
    })

    # 🆕 修正後的紀錄邏輯
    if global_step % 10 == 0:
        log_file = "train_log.csv"
        # 檢查檔案是否已存在，且裡面是否有內容
        file_exists = os.path.isfile(log_file) and os.path.getsize(log_file) > 0
        
        with open(log_file, "a", encoding="utf-8") as f:
            # 只有當檔案是新建立的，才寫入表頭
            if not file_exists:
                f.write("step,loss,lr,gate\n")
            f.write(f"{global_step},{avg_loss:.6f},{current_lr:.2e},{avg_sparse:.6f}\n")
        
    if global_step % 2000 == 0:
        checkpoint_data = {
            'step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint_data, config["save_model"])
        tqdm.write(f"💾 Step {global_step} 完整進度已存檔至 {config['save_model']}")

print("🎉 V12 訓練完成！")
