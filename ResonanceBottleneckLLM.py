import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
from tokenizers import Tokenizer
import csv 
import shutil

# ==========================================
# 🎯 V19-Adaptive 實驗配置 (3060 12G 滿載版 + 影子門控)
# ==========================================
config = {
    "d_model": 512,          
    "n_heads": 8,            
    "n_layers": 12,          
    "latent_dim": 256,       
    "dropout": 0.1,          
    "max_seq_len": 512,      
    "batch_size": 8,         
    "block_size": 256,       
    "accum_steps": 8,        
    "think_steps": 2,        # 🚀 增加思考步數，讓 Adaptive 更有感
    "lr": 3e-4,              
    "epochs": 100000,        
    "warmup_steps": 1000,    
    "bin_data": "corpus_v17_mixed.bin", 
    "save_model": "d2_v19_adaptive.pth", 
    "log_csv": "v19_adaptive_log.csv",   
    "vocab_name": "bpe_tokenizer_v12.json",     
    "vocab_size": 16384,
    
    # 🎯 Phase 2 影子門控超參數
    "halt_tau": 0.05,                  # 🌡️ 控制對「變化量」的敏感度
    "halt_weight": 0.03,                # ⚖️ 輔助 Loss 權重
    "inference_exit_threshold": 0.85   # ⚡ 推論時的退出閾值
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 V20-Adaptive 自適應深度版啟動中 | 設備: {device}")

# ==========================================
# 1. 資料加載與日誌初始化
# ==========================================
if not os.path.exists(config["bin_data"]):
    raise FileNotFoundError(f"❌ 找不到 {config['bin_data']}！請確認檔案位置。")

tokenizer = Tokenizer.from_file(config["vocab_name"])
vocab_size = tokenizer.get_vocab_size() 
data = np.memmap(config["bin_data"], dtype=np.uint16, mode='r')

def get_batch():
    ix = torch.randint(len(data) - config["block_size"], (config["batch_size"],))
    x = torch.stack([torch.from_numpy(data[i:i+config["block_size"]].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+config["block_size"]+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# 📊 初始化 CSV 日誌檔 (加入 Halt_Loss)
if not os.path.exists(config["log_csv"]):
    with open(config["log_csv"], mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "CE_Loss", "Halt_Loss", "LR", "Gate_Values"])

# ==========================================
# 2. 基礎組件
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        x_fp32 = x.float() 
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x_fp32 * rms).to(x.dtype)

class RoPE(nn.Module):
    def __init__(self, d_head, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, :, None, :])
        self.register_buffer("sin", emb.sin()[None, :, None, :])
    def forward(self, x):
        L = x.shape[1]
        cos, sin = self.cos[:, :L, :, :], self.sin[:, :L, :, :]
        x1, x2 = x.chunk(2, dim=-1)
        x_rot = torch.cat((-x2, x1), dim=-1)
        return x * cos + x_rot * sin

class CausalConv1d(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, groups=d_model)
    def forward(self, x):
        return self.conv(x.transpose(1, 2))[..., :-2].transpose(1, 2)

class SwiGLU(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        hidden_dim = int(d_model * 8 / 3) 
        hidden_dim = (hidden_dim + 63) // 64 * 64 
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
        self.ln = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x_norm = self.ln(x)
        out = self.w3(F.silu(self.w1(x_norm)) * self.w2(x_norm))
        return self.dropout(out)

# ==========================================
# 3. 核心 Attention
# ==========================================
class LatentResonanceAttentionV18(nn.Module):
    def __init__(self, d_model, latent_dim, dropout=0.1):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.d_head = d_model // self.n_heads
        
        self.ln = RMSNorm(d_model)
        self.latent_compress = nn.Linear(d_model, latent_dim, bias=False)
        self.qkv_expand = nn.Linear(latent_dim, d_model * 3, bias=False)
        self.reso_expand = nn.Linear(latent_dim, self.n_heads * 4, bias=False) 
        
        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)
        self.rope = RoPE(self.d_head, max_seq_len=config["max_seq_len"])
        
        self.out_gate = nn.Linear(latent_dim, d_model, bias=False)
        self.head_decay = nn.Parameter(torch.linspace(-3.0, 1.0, self.n_heads))
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.mem_norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward_with_context(self, context, query):
        return self.forward(context + query)

    def forward(self, x):
        B, L, D = x.shape
        x_norm = self.ln(x) 
        latent = F.silu(self.latent_compress(x_norm))
        
        q, k, v = self.qkv_expand(latent).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.d_head)
        k = k.view(B, L, self.n_heads, self.d_head)
        v = v.view(B, L, self.n_heads, self.d_head)
        
        q, k = self.rope(self.q_norm(q)), self.rope(self.k_norm(k))
        
        q_f, k_f, v_f = F.elu(q.float()) + 1.0, F.elu(k.float()) + 1.0, v.float()
        
        params = self.reso_expand(latent).view(B, L, self.n_heads, 4)
        sem_amp, sem_phase, ctx_amp, ctx_phase = params.unbind(-1)
        sem_amp, ctx_amp = torch.sigmoid(sem_amp), torch.sigmoid(ctx_amp)
        sem_phase, ctx_phase = torch.sigmoid(sem_phase) * math.pi, torch.sigmoid(ctx_phase) * math.pi
        
        raw_decay = 0.3 + 0.65 * torch.sigmoid(self.head_decay.view(1, 1, self.n_heads))
        decay_rate = torch.clamp(raw_decay, min=1e-5, max=0.999)
        
        dt_kv, dt_z = (1.0 - decay_rate).unsqueeze(-1).unsqueeze(-1), (1.0 - decay_rate).unsqueeze(-1)
        cos_diff = torch.cos(sem_phase - ctx_phase)
        base_gate = torch.sigmoid((sem_amp * ctx_amp * cos_diff) * self.temperature) 
        gate = torch.clamp(base_gate * 1.2 - 0.1, min=0.05, max=0.95)
        
        kv_input = (k_f.unsqueeze(-1) @ v_f.unsqueeze(-2)) * gate.unsqueeze(-1).unsqueeze(-1) * dt_kv 
        z_input = k_f * dt_z

        log_decay = torch.log(decay_rate).unsqueeze(-1) 
        cum_log_decay = torch.cumsum(log_decay.expand(B, L, -1, -1), dim=1) 
        safe_df_z = torch.exp(cum_log_decay) + 1e-8 
        safe_df_kv = safe_df_z.unsqueeze(-1)        
        
        kv_states = torch.cumsum(kv_input.float() / safe_df_kv, dim=1) * torch.exp(cum_log_decay).unsqueeze(-1)
        z_states = torch.cumsum(z_input.float() / safe_df_z, dim=1) * torch.exp(cum_log_decay)

        out_num = (q_f.unsqueeze(-2) @ kv_states.to(x.dtype)).squeeze(-2) 
        den = torch.clamp((q_f * z_states.to(x.dtype)).sum(dim=-1).unsqueeze(-1), min=1e-5) 
        
        out = self.mem_norm((out_num / den).contiguous().view(B, L, D))
        gate_val = F.silu(self.out_gate(latent))
        return self.dropout(self.proj(out) * gate_val)

# ==========================================
# 4. V20 推理模塊與主模型 (加入 Phase 2)
# ==========================================
class ResonanceReasoningCore(nn.Module):
    def __init__(self, d_model, latent_dim, think_steps=2):
        super().__init__()
        self.steps = think_steps
        self.step_modulator = nn.Embedding(think_steps, latent_dim * 2)
        self.latent_to_model = nn.Linear(latent_dim, d_model, bias=False)
        self.model_to_latent = nn.Linear(d_model, latent_dim, bias=False)
        self.reason_attn = LatentResonanceAttentionV18(d_model, latent_dim)
        self.gate = nn.Linear(latent_dim * 2, latent_dim)
        self.norm = RMSNorm(latent_dim)
        
        self.init_proj = nn.Linear(d_model, latent_dim)
        
        # 🛡️ LayerScale for Latent Updates
        self.gamma = nn.Parameter(torch.ones(latent_dim) * 1e-4)
        
        # Phase 2: 影子門控
        self.exit_gate = nn.Linear(latent_dim, 1) 
        
        # 📊 監控指標暫存
        self.last_halt_losses = []
        self.register_buffer("avg_gate_val", torch.zeros(1))
        self.register_buffer("avg_diff", torch.zeros(1)) # 監控是否假思考
        self.register_buffer("avg_halt_prob", torch.zeros(1))

    def _step(self, x, h_latent, step_idx):
        step_ids = torch.full((x.size(0),), step_idx, device=x.device, dtype=torch.long)
        mod = self.step_modulator(step_ids).unsqueeze(1) 
        scale, bias = mod.chunk(2, dim=-1)
        
        h_input = h_latent * (1.0 + scale) + bias
        h_query = self.latent_to_model(h_input)
        
        delta_model = self.reason_attn.forward_with_context(context=x, query=h_query)
        delta_latent = self.norm(self.model_to_latent(delta_model))
        
        # 🛡️ 殘差夾擠 (Residual Clamp)，防止局部爆炸
        delta_latent_clamped = torch.clamp(delta_latent, min=-4.0, max=4.0)
        
        gate_val = torch.sigmoid(self.gate(torch.cat([h_latent, delta_latent_clamped], dim=-1)) * 1.2)
        
        # 🛡️ LayerScale 應用於狀態更新
        h_next = h_latent + self.gamma * (gate_val * torch.tanh(delta_latent_clamped))

        # 🎯 解決 Halt Collapse: 嚴格使用 detach() 阻斷梯度作弊
        # 直接拿 delta_latent_clamped 當作變化量指標
        diff = torch.norm(delta_latent_clamped.detach(), p=2, dim=-1, keepdim=True)
        target_halt = torch.exp(-diff / config["halt_tau"])
        
        pred_halt_logit = self.exit_gate(h_next)
        pred_halt = torch.sigmoid(pred_halt_logit)
        halt_loss = F.binary_cross_entropy_with_logits(pred_halt_logit, target_halt)
        
        self.last_halt_losses.append(halt_loss)
        
        if self.training:
            # 更新 EMA 監控數據 (detach 確保不影響計算圖)
            self.avg_gate_val = 0.9 * self.avg_gate_val + 0.1 * gate_val.detach().mean()
            self.avg_diff = 0.9 * self.avg_diff + 0.1 * diff.mean()
            self.avg_halt_prob = 0.9 * self.avg_halt_prob + 0.1 * pred_halt.detach().mean()
            
            # 加入極微小的噪聲避免 mode lock
            h_next = h_next + torch.randn_like(h_next) * 1e-4 
            
        return h_next, pred_halt

    def forward(self, x):
        h_latent = self.init_proj(x)
        self.last_halt_losses = [] 
        
        for i in range(self.steps):
            if self.training:
                h_latent, s = checkpoint(self._step, x, h_latent, i, use_reentrant=False)
            else:
                h_latent, s = self._step(x, h_latent, i)
                # 🛡️ 保守退出：只有當所有 Token 都準備好 (min > threshold) 才退出
                if s.min() > config["inference_exit_threshold"]:
                    break
                    
        return self.latent_to_model(self.norm(h_latent))

class D2V18AttentionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = LatentResonanceAttentionV18(d_model, latent_dim=config["latent_dim"])
        self.ffn = SwiGLU(d_model, dropout=config["dropout"])
        # 🛡️ LayerScale: 初始化為 1e-4
        self.gamma_1 = nn.Parameter(torch.ones(d_model) * 1e-4)
        self.gamma_2 = nn.Parameter(torch.ones(d_model) * 1e-4)
        
    def forward(self, x):
        x = x + self.gamma_1 * self.attn(x)
        x = x + self.gamma_2 * self.ffn(x)
        return x

class D2V18ConvBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln = RMSNorm(d_model)
        self.conv = CausalConv1d(d_model)
        self.ffn = SwiGLU(d_model, dropout=config["dropout"])
        # 🛡️ LayerScale
        self.gamma_1 = nn.Parameter(torch.ones(d_model) * 1e-4)
        self.gamma_2 = nn.Parameter(torch.ones(d_model) * 1e-4)
        
    def forward(self, x):
        x = x + self.gamma_1 * self.conv(self.ln(x))
        x = x + self.gamma_2 * self.ffn(x)
        return x

class D2V19StableModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.emb_dropout = nn.Dropout(config["dropout"])
        
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            if i in [3, 7, 11]: 
                self.blocks.append(ResonanceReasoningCore(d_model, config["latent_dim"], config["think_steps"]))
            elif i % 2 == 0:
                self.blocks.append(D2V18AttentionBlock(d_model))
            else:
                self.blocks.append(D2V18ConvBlock(d_model))
                
        self.out_ln = RMSNorm(d_model) 
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight 

    def get_halt_loss(self):
        """🎯 收集所有推理層的影子損失"""
        total_halt_loss = 0
        count = 0
        for block in self.blocks:
            if isinstance(block, ResonanceReasoningCore):
                if block.last_halt_losses:
                    # Checkpoint 機制會導致 Forward 跑兩次，這裡取前 steps 個即可避免重複計算
                    valid_losses = block.last_halt_losses[:config["think_steps"]]
                    total_halt_loss += torch.stack(valid_losses).mean()
                    count += 1
        return total_halt_loss / count if count > 0 else 0
        
    def forward(self, x):
        x = self.emb_dropout(self.embedding(x))
        for block in self.blocks:
            if isinstance(block, ResonanceReasoningCore):
                x = x + block(x) # 內部已處理 checkpoint
            else:
                x = block(x)
        return self.head(self.out_ln(x))

# ==========================================
# 5. 訓練與監控迴圈
# ==========================================
model = D2V19StableModel(config["vocab_size"], config["d_model"], config["n_layers"]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)

global_step = 0
if os.path.exists(config["save_model"]):
    print(f"♻️ 接續訓練: {config['save_model']}")
    ckpt = torch.load(config["save_model"], map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    global_step = ckpt.get('step', 0)

for param_group in optimizer.param_groups:
    param_group['initial_lr'] = config["lr"]
    param_group['lr'] = config["lr"]

def get_lr_multiplier(step):
    if step < config["warmup_steps"]:
        return (step + 1) / config["warmup_steps"]
    
    decay_steps = config["epochs"] - config["warmup_steps"]
    current_decay_step = step - config["warmup_steps"]
    
    min_lr_ratio = 0.1 
    cosine_decay = min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * current_decay_step / decay_steps))
    return cosine_decay

warmup_scheduler = LambdaLR(optimizer, lr_lambda=get_lr_multiplier, last_epoch=global_step)
print(f"🌟 V20-Adaptive 模型參數: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"🚀 影子門控已掛載！局部推理層在 Block 4, 8, 12，輔助權重 {config['halt_weight']}")

model.train()
pbar = tqdm(initial=global_step, total=config["epochs"], desc="🧠 V19-Adaptive 訓練中")

while global_step < config["epochs"]:
    optimizer.zero_grad(set_to_none=True)
    step_ce_loss = 0 
    step_halt_loss = 0
    
    # ==========================================
    # 🛡️ 策略 A: Gamma Warmup (動態推升殘差權重)
    # ==========================================
    target_gamma = 1e-4
    if global_step >= 10000:
        target_gamma = 5e-3
    elif global_step >= 3000:
        target_gamma = 1e-3

    # 使用 no_grad 動態調整模型中所有的 gamma 參數
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'gamma' in name:
                param.data = torch.max(param.data, torch.tensor(target_gamma, dtype=param.dtype, device=device))

    # ==========================================
    # 🛡️ 策略 B: 前期停用 Halt Learning
    # ==========================================
    current_halt_weight = 0.0 if global_step < 3000 else config["halt_weight"]

    # 🎯 這裡只需要一個迴圈，處理前向傳播、Loss 計算與反向傳播
    for _ in range(config["accum_steps"]):
        xb, yb = get_batch()
        with autocast('cuda', dtype=torch.bfloat16):
            logits = model(xb)
            
            # 1. 主線語言建模 Loss
            ce_loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            
            # 2. Phase 2 影子門控 Loss
            halt_loss = model.get_halt_loss()
            
            # 總和 Loss (使用動態權重 current_halt_weight)
            combined_loss = ce_loss + (current_halt_weight * halt_loss)
            loss_to_back = combined_loss / config["accum_steps"]
        
        loss_to_back.backward()
        
        # 把數值累加搬到這裡
        step_ce_loss += ce_loss.item()
        step_halt_loss += halt_loss.item() if isinstance(halt_loss, torch.Tensor) else halt_loss

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    

    avg_ce = step_ce_loss / config["accum_steps"]
    avg_halt = step_halt_loss / config["accum_steps"]
    
    warmup_scheduler.step()
    global_step += 1
    
    gate_vals = [b.avg_gate_val.item() for b in model.blocks if isinstance(b, ResonanceReasoningCore)]
    gate_str = f"[{','.join([f'{g:.3f}' for g in gate_vals])}]" if gate_vals else "N/A"
    current_lr = optimizer.param_groups[0]['lr']

    # 在訓練迴圈後段：
    diffs = [b.avg_diff.item() for b in model.blocks if isinstance(b, ResonanceReasoningCore)]
    halts = [b.avg_halt_prob.item() for b in model.blocks if isinstance(b, ResonanceReasoningCore)]
    
    # 格式化為字串
    diff_str = f"[{','.join([f'{d:.1f}' for d in diffs])}]" if diffs else "N/A"
    # P(Exit) 是機率，保留 2 位小數即可 (例如 0.48)
    halt_str = f"[{','.join([f'{h:.2f}' for h in halts])}]" if halts else "N/A"

    # 🚨 實作崩塌警報
    # 1. 假思考警報：如果有任何一層的 diff 低於 0.001
    diff_alert = " ⚠️(假思考!)" if diffs and any(d < 0.001 for d in diffs) else ""
    
    # 2. 提早坍塌警報：如果在前 5000 步，P(Exit) 就飆破 0.95
    halt_alert = " ⚠️(Halt坍塌!)" if halts and any(h > 0.95 for h in halts) and global_step < 5000 else ""

    pbar.update(1)
    pbar.set_postfix({
        "CE": f"{avg_ce:.3f}", 
        "HW": f"{current_halt_weight:.2f}", # 縮寫為 HW
        "Diff": diff_str + diff_alert,      
        "P(Ex)": halt_str + halt_alert      # 縮寫為 P(Ex)
    })

    # 📝 CSV 日誌寫入
    if global_step % 10 == 0:
        with open(config["log_csv"], mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([global_step, f"{avg_ce:.4f}", f"{avg_halt:.4f}", f"{current_lr:.6f}", gate_str])

    # 💾 模型存檔
    if global_step % 1000 == 0:
        # 1. 儲存最新權重 (覆蓋式，用於接續訓練)
        ckpt = {
            'step': global_step, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(ckpt, config["save_model"])
        
        # 2. 建立備份檔案 (獨立檔案，防止模型崩潰後無法救回)
        # 格式例如: d2_v19_adaptive_step_1000.pth
        backup_path = config["save_model"].replace(".pth", f"_step_{global_step}.pth")
        shutil.copy2(config["save_model"], backup_path)
        
        print(f"\n🚩 Step {global_step} 存檔成功！")
        print(f"📦 已建立備份: {backup_path}")
        print(f"📊 已同步記錄至 {config['log_csv']}。Gate 狀態: {gate_str}")

        # 💡 (選用) 自動清理舊備份的邏輯：
        # 如果你擔心硬碟爆掉，可以手動刪除更早之前的備份，
        # 但建議前期保留 5~10 個備份點以供對比 Loss 曲線。