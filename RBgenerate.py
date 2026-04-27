import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import os
import math

# ==========================================
# 1. V16 核心配置 (與訓練腳本完全同步)
# ==========================================
config = {
    "d_model": 768,
    "n_heads": 12,
    "n_layers": 12,
    "block_size": 512,
    "vocab_name": "bpe_tokenizer_v12.json",
    "save_model": "d2_v16_resonance_pro.pth", # 預設載入最新的 V16 權重
    "bottleneck_rank": 256,
}

# ==========================================
# 2. V16 模型架構 (從 ResonanceBottleneckLLM.py 完整移植)
# ==========================================
class ResonanceMemoryAttentionV16(nn.Module):
    def __init__(self, d_model, bottleneck_rank=256):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.d_head = d_model // self.n_heads
        
        # 🟢 V16 新增：局部感知層
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
        
        # 🟢 V16：Local Conv 殘差連接與截斷
        x_t = x.transpose(1, 2)
        conv_out = self.local_conv(x_t)[..., :-2] 
        x_conv = x + conv_out.transpose(1, 2)     
        x_norm = self.ln(x_conv)
        
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        
        # 🌊 V16：五元組共振參數
        params = self.bottleneck(x_norm).view(B, L, self.n_heads, 5)
        sem_amp, sem_phase, ctx_amp, ctx_phase, decay_raw = params.unbind(-1)
        sem_amp, ctx_amp = torch.sigmoid(sem_amp), torch.sigmoid(ctx_amp)
        sem_phase, ctx_phase = torch.tanh(sem_phase) * math.pi, torch.tanh(ctx_phase) * math.pi
        
        # 🟢 V16：強化忘記能力 (0.3 ~ 0.95)
        decay_rate = 0.3 + 0.65 * torch.sigmoid(decay_raw) 
        cos_diff = torch.cos(sem_phase - ctx_phase)
        gate = torch.sigmoid((sem_amp * ctx_amp * cos_diff) * self.temperature) 

        q_f = (F.elu(q.float()) + 1.0).view(B, L, self.n_heads, self.d_head)
        k_f = (F.elu(k.float()) + 1.0).view(B, L, self.n_heads, self.d_head)
        v_f = v.float().view(B, L, self.n_heads, self.d_head)

        dt = (1.0 - decay_rate).unsqueeze(-1).unsqueeze(-1)
        
        # 🟢 V16：記憶更新與 Clamp 防爆
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

        # 🟢 V16：分母穩定度 1e-4
        den = (q_f * z_states).sum(dim=-1).unsqueeze(-1).transpose(1, 2) + 1e-4
        out = out_num / den 

        out = out.transpose(1, 2).contiguous() 
        out = self.mem_norm(out) 
        return self.proj(out.view(B, L, D).to(x.dtype)), gate

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
        for block in self.blocks:
            x, _ = block(x) # 推論時不需要回傳 gate 與 checkpoint
        return self.head(self.out_ln(x))

# ==========================================
# 3. 推論與交互邏輯
# ==========================================
def generate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(config["vocab_name"])
    model = D2V16Model(tokenizer.get_vocab_size(), config["d_model"], config["n_layers"]).to(device)
    
    # 自動尋找最新的權重或里程碑備份
    model_path = config["save_model"]
    if not os.path.exists(model_path):
        print(f"⚠️ 找不到預設權重 {model_path}，嘗試尋找最新備份...")
        import glob
        backups = glob.glob("d2_v16_step_*.pth")
        if backups:
            model_path = sorted(backups, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        else:
            print("❌ 找不到任何權重檔案！")
            return

    print(f"♻️ 正在載入 V16 權重: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    # 兼容舊版存檔格式與完整 ckpt 格式
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"🚀 V16 準備就緒！(Temp: 0.7, RepPenalty: 1.2)")
    
    while True:
        prompt = input("\n💡 請輸入開頭 (輸入 'q' 退出): ")
        if prompt.lower() == 'q': break
        
        input_ids = torch.tensor(tokenizer.encode(prompt).ids, dtype=torch.long).unsqueeze(0).to(device)
        print(f"🤖 V16 生成中: ", end="", flush=True)
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(200): # 增加生成長度到 200
                context = generated[:, -config["block_size"]:]
                logits = model(context)
                logits = logits[:, -1, :] / 0.7 # 採樣溫度
        
                # 重複懲罰
                for token_id in set(generated[0].tolist()):
                    logits[0, token_id] /= 1.2 
                
                # Top-K 採樣
                v, _ = torch.topk(logits, 40)
                logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat((generated, next_token), dim=1)
                
                word = tokenizer.decode([next_token.item()])
                print(word, end="", flush=True)
                
                if next_token.item() == tokenizer.token_to_id("<|endoftext|>"): break
        print("\n" + "-"*30)

if __name__ == "__main__":
    generate()