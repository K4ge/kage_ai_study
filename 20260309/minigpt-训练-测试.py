import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# 定义文本
text = text = """新时代云城发轫，互联网与人工智能的爆炸式增长使科技的浤浤浪潮汹涌澎拜，许多问题只需轻轻按下“搜索”一键便可获得答案，那么，我们的问题是否会越来越少？私以为，我们的问题不会越来越少，也不应该越来越少。

　　“问题”是人类对未知的探索，对已知“从来如此，便对么”的深刻反思，人类知识的产生，智慧的集聚与思想的深邃都在提出问题与解决问题的征途中实现的。“问题”有很多种。科技能回答知识型问题，但无法回答“我是谁，我从哪里来，要到哪里去”的智慧型问题。试想一个由数据与算法堆砌而成的人工智能，如何去思考去回答后验的具有人类本体性意义的终极命题？如何给出“生命、自由与爱”的答案？

　　与此同时，真理的具体性条件性决定随着社会实践的发展，“问题”将以不同的形态不同的内涵不断产生。认识的无限性，决定“问题”是无限的。

　　由是观之，不是所有“问题”，科技都可以给出答案，不是所有“问题”都有答案，科技也许可以使我们知识型问题减少，但它无法阻止我们对问题的深度挖掘，再挖掘。

　　既然问题必不可少，问题意识更不可在信息的横流中黯淡熄灭。问题意识，是人类学家项飙敏锐观察年轻人生存状态后提出“附近的消失”精神窘境，尝试“重建附近”来阻止个体间连通性与互渗性的消亡；是电影学家戴锦华在洞察科技危机后深入思考人文与人工智能的关系，呼吁对人文情怀与价值理性的唤醒……他们都在时代的宏大叙事下细致入微地观察现实，以特有的敏感性与高度的问题意识尝试提出问题、解决问题——这背后，是高度的人文关怀与社会责任感，是对时代命题与人生母题孜孜思考的向上精神的奔赴性。

　　然反观当下，多少人已被无用的信息充斥、异化成丧失精神独立性的“单向度人”？有人浑浑噩噩，沉湎于低级肤浅的泛娱乐化潮流而忘却严肃的深度的思考；有人丧失质疑精神，在“暗室效应”中成为群体中的“横态木偶”，只会情绪化思考……究其病因，是问题意识的懈怠，是积极思考方式与实事求是思考态度的缺位。

　　因此，欲稳立于信息爆炸的历史节点，我们应保持审慎的态度，善于发问，高举问题意识的火炬，驱散“理所当然”的黑暗，构建起自己的精神岛屿，完成陀思妥耶夫斯基笔下“人不是齿轮”的自证。

　　阿伦特曾言：“用思考和发问来恢复我们作为人的存在本质。”科技浤浪澎湃，我们仍应“思考和发问”，来寻得人的本质、“问题”的意义。

""".replace(" ","").replace("\n","")

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
block_size = 32
batch_size = 5

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
prompt = "新时代"
idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
generated = model.generate(idx, 1000)[0].cpu().tolist()
print(decode(generated))