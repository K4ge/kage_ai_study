import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# 读取文本
with open("data/wiki_zh_10mb.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("文本前200个字符：")
print(text[:200])
print("总字符数：", len(text))

# 去重
tokenizer = Tokenizer.from_file("data/my_tokenizer.json")

vocab_size = tokenizer.get_vocab_size()
print("词表大小 vocab_size =", vocab_size)

def encode(s):
    return tokenizer.encode(s).ids

def decode(ids):
    return tokenizer.decode(ids)

data = encode(text)
print("编码后前100个 token：", data[:100])

data = torch.tensor(data, dtype=torch.long)
print("data.shape:", data.shape)

n = int(0.9 * len(data))   # 前90%做训练集
train_data = data[:n]
val_data = data[n:]

print("train_data.shape:", train_data.shape)
print("val_data.shape:", val_data.shape)


embed_dim = 128
num_heads = 4
n_layer = 4
block_size = 128
batch_size = 32
temperature = 0.8
dropout = 0.2
top_k = 20
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.query(x)  # (B, T, C)
        K = self.key(x)  # (B, T, C)
        V = self.value(x)  # (B, T, C)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        score = Q @ K.transpose(-2, -1)  # (B, nh, T, T)
        score = score / (self.head_dim ** 0.5)

        mask = torch.tril(torch.ones(T, T, device=x.device))
        score = score.masked_fill(mask == 0, float('-inf'))

        att = torch.softmax(score, dim=-1)  # (B, nh, T, T)
        att = self.attn_dropout(att)
        out = att @ V  # (B, nh, T, hs)

        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.proj(out)  # (B, T, C)
        out = self.resid_dropout(out)

        return out


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.attention = SelfAttention(embed_dim, num_heads)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, n_layer, num_heads, dropout):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)

        self.blocks = nn.Sequential(
            *[Block(embed_dim, num_heads) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)   # (B,T,C)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)  # (T,C)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)   # (B,T,vocab_size)

        if targets is not None:
            B, T, C = logits.shape
            logits_2d = logits.view(B * T, C)
            targets_1d = targets.view(B * T)
            loss = F.cross_entropy(logits_2d, targets_1d)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens,temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]   # 只保留最后 block_size 个 token
            logits, _ = self(idx_cond)             # (B,T,C)
            logits = logits[:, -1, :]              # 只取最后一个位置的预测 (B,C)
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx


# 初始化模型
model = GPT(
    vocab_size=vocab_size,
    embed_dim=256,
    block_size=block_size,
    n_layer=4,
    num_heads=num_heads,
    dropout=dropout
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

def get_batch(split):
    data_source = train_data if split == "train" else val_data

    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i:i + block_size] for i in ix])
    y = torch.stack([data_source[i + 1:i + block_size + 1] for i in ix])

    x = x.to(device)
    y = y.to(device)

    return x, y

x, y = get_batch("train")
print("train x shape:", x.shape)
print("train y shape:", y.shape)

x, y = get_batch("val")
print("val x shape:", x.shape)
print("val y shape:", y.shape)

eval_iters = 20

def estimate_loss():
    out = {}

    model.eval()

    with torch.no_grad():
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)

            for k in range(eval_iters):
                x, y = get_batch(split)
                logits, loss = model(x, y)
                losses[k] = loss.item()

            out[split] = losses.mean().item()

    model.train()
    return out

losses = estimate_loss()
print(losses)

max_iters = 1000
eval_interval = 100

for step in range(max_iters):

    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    x, y = get_batch("train")
    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()

start_text = "人工智能"
start_ids = torch.tensor([encode(start_text)], dtype=torch.long, device=device)

generated = model.generate(
    start_ids,
    max_new_tokens=300,
    temperature=0.9,
    top_k=100
)[0].cpu().tolist()

result = decode(generated)
print("生成结果：")
print(result)