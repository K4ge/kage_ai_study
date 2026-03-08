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

