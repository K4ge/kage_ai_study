import torch
import torch.nn as nn

B,T,C = 2,4,6

x = torch.randn(B,T,C)

Wq = torch.randn(C,C)
Wk = torch.randn(C,C)
Wv = torch.randn(C,C)

Q = x @ Wq
K = x @ Wk
V = x @ Wv

print(Q.shape)
print(K.shape)
print(V.shape)
score = Q @ K.transpose(-2,-1)
print(score.shape)

score = score / (C ** 0.5)

mask = torch.tril(torch.ones(T,T))
score = score.masked_fill(mask==0, float('-inf'))

print(score)

att = torch.softmax(score, dim=-1)
print(att)

out = att @ V
