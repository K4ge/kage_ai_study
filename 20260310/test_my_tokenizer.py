from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("data/my_tokenizer.json")

samples = [
    "人工智能正在快速发展，中国的研究人员提出了很多新的方法。",
    "2026年，AI model 在中国发展很快，ChatGPT-like 系统越来越多。"
]

for text in samples:
    print("=" * 60)
    print("原文：", text)

    enc = tokenizer.encode(text)

    print("token 数量：", len(enc.ids))
    print("tokens：")
    print(enc.tokens)

    decoded = tokenizer.decode(enc.ids)
    print("decode 回来：")
    print(decoded)