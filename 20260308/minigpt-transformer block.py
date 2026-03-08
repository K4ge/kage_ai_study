import torch
import torch.nn as nn

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