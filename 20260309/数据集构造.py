from datasets import load_dataset
from opencc import OpenCC

cc = OpenCC('t2s')

dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.zh",
    split="train",
    streaming=True
)

for i, x in enumerate(dataset):

    text = x["text"]

    # 繁体 → 简体
    text = cc.convert(text)

    print(text[:200])

    if i > 5:
        break