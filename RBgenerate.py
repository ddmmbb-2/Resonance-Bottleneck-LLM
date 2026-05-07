import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tokenizers import Tokenizer

# ==========================================
# 1. V19-Adaptive 架構定義 (必須與訓練腳本完全一致)
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

class LatentResonanceAttentionV18(nn.Module):
    def __init__(self, d_model, latent_dim, dropout=0.1):
        super().__init__()
        self.n_heads = 8
        self.d_head = d_model // self.n_heads
        self.ln = RMSNorm(d_model)
        self.latent_compress = nn.Linear(d_model, latent_dim, bias=False)
        self.qkv_expand = nn.Linear(latent_dim, d_model * 3, bias=False)
        self.reso_expand = nn.Linear(latent_dim, self.n_heads * 4, bias=False) 
        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)
        self.rope = RoPE(self.d_head, max_seq_len=512)
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
        q, k, v = [t.view(B, L, self.n_heads, self.d_head) for t in (q, k, v)]
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
        gate = torch.clamp(torch.sigmoid((sem_amp * ctx_amp * cos_diff) * self.temperature) * 1.2 - 0.1, 0.05, 0.95)
        kv_input = (k_f.unsqueeze(-1) @ v_f.unsqueeze(-2)) * gate.unsqueeze(-1).unsqueeze(-1) * dt_kv 
        z_input = k_f * dt_z
        log_decay = torch.log(decay_rate).unsqueeze(-1) 
        cum_log_decay = torch.cumsum(log_decay.expand(B, L, -1, -1), dim=1) 
        safe_df_z = torch.exp(cum_log_decay) + 1e-8 
        kv_states = torch.cumsum(kv_input / safe_df_z.unsqueeze(-1), dim=1) * torch.exp(cum_log_decay).unsqueeze(-1)
        z_states = torch.cumsum(z_input / safe_df_z, dim=1) * torch.exp(cum_log_decay)
        out_num = (q_f.unsqueeze(-2) @ kv_states.to(x.dtype)).squeeze(-2) 
        den = torch.clamp((q_f * z_states.to(x.dtype)).sum(dim=-1).unsqueeze(-1), min=1e-5) 
        out = self.mem_norm((out_num / den).view(B, L, D))
        return self.dropout(self.proj(out) * F.silu(self.out_gate(latent)))

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
        self.gamma = nn.Parameter(torch.ones(latent_dim) * 1e-4)
        self.exit_gate = nn.Linear(latent_dim, 1)

        self.register_buffer("avg_gate_val", torch.zeros(1))
        self.register_buffer("avg_diff", torch.zeros(1))
        self.register_buffer("avg_halt_prob", torch.zeros(1))

    def _step(self, x, h_latent, step_idx):
        step_ids = torch.full((x.size(0),), step_idx, device=x.device, dtype=torch.long)
        mod = self.step_modulator(step_ids).unsqueeze(1) 
        scale, bias = mod.chunk(2, dim=-1)
        h_query = self.latent_to_model(h_latent * (1.0 + scale) + bias)
        delta_model = self.reason_attn.forward_with_context(context=x, query=h_query)
        delta_latent = torch.clamp(self.norm(self.model_to_latent(delta_model)), -4.0, 4.0)
        gate_val = torch.sigmoid(self.gate(torch.cat([h_latent, delta_latent], dim=-1)) * 1.2)
        h_next = h_latent + self.gamma * (gate_val * torch.tanh(delta_latent))
        pred_halt = torch.sigmoid(self.exit_gate(h_next))
        return h_next, pred_halt

    def forward(self, x):
        h_latent = self.init_proj(x)
        for i in range(self.steps):
            h_latent, s = self._step(x, h_latent, i)
            if s.min() > 0.85: # Inference exit threshold
                break
        return self.latent_to_model(self.norm(h_latent))

class D2V19StableModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(0.1)
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            if i in [3, 7, 11]: 
                self.blocks.append(ResonanceReasoningCore(d_model, 256, 2))
            elif i % 2 == 0: self.blocks.append(D2V18AttentionBlock(d_model))
            else: self.blocks.append(D2V18ConvBlock(d_model))
        self.out_ln = RMSNorm(d_model) 
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight 

    def forward(self, x):
        x = self.emb_dropout(self.embedding(x))
        for block in self.blocks:
            x = x + block(x) if isinstance(block, ResonanceReasoningCore) else block(x)
        return self.head(self.out_ln(x))

class D2V18AttentionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = LatentResonanceAttentionV18(d_model, 256)
        self.ffn = SwiGLU(d_model)
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
        self.ffn = SwiGLU(d_model)
        self.gamma_1 = nn.Parameter(torch.ones(d_model) * 1e-4)
        self.gamma_2 = nn.Parameter(torch.ones(d_model) * 1e-4)
    def forward(self, x):
        x = x + self.gamma_1 * self.conv(self.ln(x))
        x = x + self.gamma_2 * self.ffn(x)
        return x

# ==========================================
# 2. 載入與推理配置 (指向 V19 權重)
# ==========================================

config = {
    "d_model": 512,
    "n_layers": 12,
    "vocab_size": 16384,
    "vocab_name": "bpe_tokenizer_v12.json",
    "save_model": "d2_v19_adaptive.pth", 
    "max_seq_len": 512,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer.from_file(config["vocab_name"])

# 🎯 修改這裡：直接從 tokenizer 取得實際大小，不要用 config["vocab_size"]
actual_vocab_size = tokenizer.get_vocab_size()
print(f"📊 Tokenizer 實際詞彙量: {actual_vocab_size}")

# 使用實際大小初始化模型
# 🎯 在初始化模型時，手動 +1 給點緩衝空間
# 這樣就算 Tokenizer 噴出 16384，模型也接得住
model = D2V19StableModel(actual_vocab_size + 1, config["d_model"], config["n_layers"]).to(device)

try:
    # 1. 讀取權重檔到記憶體中
    ckpt = torch.load(config["save_model"], map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']

    # 2. 手動處理 Embedding 與 Head 的形狀不匹配 (Shape Surgery)
    checkpoint_vocab_size = state_dict['embedding.weight'].shape[0]
    model_vocab_size = actual_vocab_size + 1

    if checkpoint_vocab_size < model_vocab_size:
        print(f"🔧 偵測到詞彙量不匹配，正在手動補齊第 {model_vocab_size} 個位置...")
        
        # 幫 Embedding 補上一列 (0 或者是平均值)
        old_emb = state_dict['embedding.weight']
        new_emb = torch.zeros((model_vocab_size, old_emb.shape[1]), device=old_emb.device, dtype=old_emb.dtype)
        new_emb[:checkpoint_vocab_size, :] = old_emb
        state_dict['embedding.weight'] = new_emb
        
        # 因為這個模型有 Weight Tying (Head 跟 Embedding 共用權重)，所以 Head 也要更新
        if 'head.weight' in state_dict:
            state_dict['head.weight'] = new_emb

    # 3. 載入修改後的 state_dict
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"✅ V19-Adaptive 載入成功 (已手動修正權重偏移)")
    print(f"📊 原始權重大小: {checkpoint_vocab_size} -> 現行模型大小: {model_vocab_size}")

except Exception as e:
    print(f"❌ 載入失敗: {e}")
    import traceback
    traceback.print_exc() # 顯示更詳細的錯誤訊息以便追蹤
    exit()

# ==========================================
# 3. 生成函數 (Nucleus Sampling)
# ==========================================

@torch.no_grad()
def generate_response(prompt, temperature=0.7, top_p=0.9, repetition_penalty=1.15):
    input_text = f"User: {prompt}\nAssistant:"
    input_ids = tokenizer.encode(input_text).ids
    
    if max(input_ids) >= actual_vocab_size:
        input_ids = [min(idx, actual_vocab_size - 1) for idx in input_ids]
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    print("V19-3K: ", end="", flush=True)
    generated_tokens = []
    
    for _ in range(256): 
        x_cond = x[:, -config["max_seq_len"]:]
        logits = model(x_cond)[:, -1, :]
        
        # 修正版重複懲罰 (處理負數 Logits 的情況)
        for token_id in set(generated_tokens):
            if logits[0, token_id] < 0:
                logits[0, token_id] *= repetition_penalty
            else:
                logits[0, token_id] /= repetition_penalty
        
        logits = logits / temperature
        
        # 修正版 Top-p 採樣 (在 Logits 階段過濾)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        indices_to_remove = cumulative_probs > top_p
        indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
        indices_to_remove[..., 0] = 0
        
        # 將不要的 Token 權重設為負無窮大
        sorted_logits[indices_to_remove] = float('-inf')
        
        # 散射回原本的排序
        logits.scatter_(1, sorted_indices, sorted_logits)
        
        # 安全地計算機率
        probs = F.softmax(logits, dim=-1)
        
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)
        next_id = next_token.item()
        generated_tokens.append(next_id)
        
        char = tokenizer.decode([next_id])
        if next_id == tokenizer.token_to_id("<|endoftext|>"): break
        print(char, end="", flush=True)
        if len(generated_tokens) > 3 and all(t == generated_tokens[-1] for t in generated_tokens[-3:]): break
            
    print("\n")

if __name__ == "__main__":
    print("\n--- 🧠 V19 Adaptive 3,000 Step 測試模式 ---")
    while True:
        user_input = input("你: ")
        if user_input.lower() in ['exit', 'quit']: break
        generate_response(user_input)