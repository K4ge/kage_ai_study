import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        B,T,C = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        score = Q @ K.transpose(-2,-1)

        score = score / (C ** 0.5)

        mask = torch.tril(torch.ones(T,T)).to(x.device)

        score = score.masked_fill(mask==0, float('-inf'))

        att = torch.softmax(score, dim=-1)

        out = att @ V

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
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        else:

            loss = None
        return logits, loss