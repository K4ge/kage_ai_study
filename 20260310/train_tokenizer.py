from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.normalizers import NFKC

# 1. 定义一个 BPE tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# 2. 规范化
tokenizer.normalizer = NFKC()

# 3. 预切分
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 4. 解码器
tokenizer.decoder = decoders.ByteLevel()

# 5. 训练器
trainer = trainers.BpeTrainer(
    vocab_size=5000,
    min_frequency=2,
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
)

# 6. 用你的语料训练
files = ["data/wiki_zh_10mb.txt"]
tokenizer.train(files, trainer)

# 7. 保存
tokenizer.save("data/my_tokenizer.json")

print("tokenizer 训练完成并已保存！")