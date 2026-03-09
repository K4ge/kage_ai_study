import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# 定义文本
text = "一直以来对AI就挺感兴趣，但由于工作忙一直没有下定决心好好学习AI。同时目前工作对AI的融合要求也越来越高，能够掌握AI方面的底层知识对于自己工作及个人竞争力的提升有很大帮助。恰逢马年回去相亲认识一亲戚朋友，南大硕士研究AI智能驾驶相关知识，对人工智能机计算机底层芯片相关知识特别专业，个人对该方向也很感兴趣，恰逢此契机点燃了我内心的火炬，也跟随该项目记录自己的AI成长计划。"

# 去重
chars = sorted(list(set(text)))
print(chars)

# 定义词表
vocab_size = len(chars)
print(vocab_size)

# 词到数值
stoi = {ch: i for i, ch in enumerate(chars)}
print(stoi)

# 数值到词
itos = {i: ch for i, ch in enumerate(chars)}
print(itos)

# 编码：词到数值
def encode(s):
    return [stoi[c] for c in s]


# 解码：数值到词
def decode(l):
    return ''.join([itos[i] for i in l])

print(decode([3, 2, 4, 4, 5]))

# 编码
data = encode(text)
print(data)

# 变为 tensor
data = torch.tensor(data, dtype=torch.long)
print(data)

# 随机获取2批，每批4个token
block_size = 4
batch_size = 2

def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

# 看一下 batch
x, y = get_batch(data)
print(x.shape)
print(y.shape)
print(x)
print(y)


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.query(x)   # (B,T,C)
        K = self.key(x)     # (B,T,C)
        V = self.value(x)   # (B,T,C)

        score = Q @ K.transpose(-2, -1)   # (B,T,T)
        score = score / (C ** 0.5)

        mask = torch.tril(torch.ones(T, T, device=x.device))
        score = score.masked_fill(mask == 0, float('-inf'))

        att = torch.softmax(score, dim=-1)   # (B,T,T)
        out = att @ V                         # (B,T,C)

        return out


class Block(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.attention = SelfAttention(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, n_layer):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)

        self.blocks = nn.Sequential(
            *[Block(embed_dim) for _ in range(n_layer)]
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

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]   # 只保留最后 block_size 个 token
            logits, _ = self(idx_cond)             # (B,T,C)
            logits = logits[:, -1, :]              # 只取最后一个位置的预测 (B,C)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx


# 初始化模型
model = GPT(
    vocab_size=vocab_size,
    embed_dim=256,
    block_size=block_size,
    n_layer=2
).to(device)

print(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# 训练
for step in range(1000):
    x, y = get_batch(data)
    x = x.to(device)
    y = y.to(device)

    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"step {step}, loss = {loss.item():.4f}")

# 生成
start = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(start, 1000)[0].cpu().tolist()
print(decode(generated))