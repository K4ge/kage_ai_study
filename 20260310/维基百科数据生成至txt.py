from datasets import load_dataset
from opencc import OpenCC

# 繁体转简体
cc = OpenCC('t2s')

dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.zh",
    split="train",
    streaming=True
)

output_file = open("wiki_zh_10mb.txt", "w", encoding="utf-8")

size_limit = 10 * 1024 * 1024   # 10MB
current_size = 0

for i, x in enumerate(dataset):

    text = x["text"]

    # 繁体转简体
    text = cc.convert(text)

    # 每篇文章加换行
    text = text.strip() + "\n\n"

    output_file.write(text)

    current_size += len(text.encode("utf-8"))

    if i % 100 == 0:
        print(f"articles: {i}, size: {current_size/1024/1024:.2f} MB")

    if current_size > size_limit:
        break

output_file.close()

print("finished!")