from transformers import AutoTokenizer

sample_text = "2026年，AI model 在中国发展很快，ChatGPT-like 系统越来越多。"

tokenizer_names = [
    "bert-base-chinese",
    "gpt2",
]

for name in tokenizer_names:
    print("=" * 60)
    print("tokenizer:", name)

    tokenizer = AutoTokenizer.from_pretrained(name)
    enc = tokenizer(sample_text, add_special_tokens=False)

    input_ids = enc["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print("token 数量：", len(input_ids))
    print("tokens:")
    print(tokens)

    print("decode 回来：")
    print(tokenizer.decode(input_ids))
    print()