import torch
#定义文本
text = "hello world"
#去重
chars = sorted(list(set(text)))
print(chars)
#定义词表
vocab_size = len(chars)
print(vocab_size)

#词到数值
stoi = {ch:i for i,ch in enumerate(chars)}
print(stoi)
#数值到词
itos = {i:ch for i,ch in enumerate(chars)}
print(itos)

#编码：词到数值
def encode(s):
    return [stoi[c] for c in s]
print(encode('hello'))

#解码：数值到词
def decode(l):
    return ''.join([itos[i] for i in l])

print(decode([3, 2, 4, 4, 5]))

#编码
data = encode(text)
print(data)


#变为tensor
data = torch.tensor(encode(text), dtype=torch.long)

print(data)

#随机获取2批，每批4个token
block_size = 4
batch_size = 2
def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i + block_size] for i in ix])

    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x, y

#获取到x有两批[2,4]y也同理要错位
x, y = get_batch(data)

print(x.shape)
print(y.shape)

print(x)
print(y)