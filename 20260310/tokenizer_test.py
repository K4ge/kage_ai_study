from transformers import AutoTokenizer

sample_text = "2026年，AI model 在中国发展很快，ChatGPT-like 系统越来越多。"
# sample_text = "人工智能正在快速发展，中国的研究人员提出了很多新的方法。"
# char-level
char_tokens = list(sample_text)
print("=== char-level ===")
print("原始文本长度：", len(sample_text))
print("token 数量：", len(char_tokens))
print(char_tokens)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
enc = tokenizer(sample_text, add_special_tokens=False)

input_ids = enc["input_ids"]
tokens = tokenizer.convert_ids_to_tokens(input_ids)

print("\n=== tokenizer ===")
print("token 数量：", len(input_ids))
print(tokens)
print(input_ids)

decoded = tokenizer.decode(input_ids)
print("\ndecode 回来：")
print(decoded)