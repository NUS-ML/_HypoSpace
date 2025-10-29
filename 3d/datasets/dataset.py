import json
import random

# ==== 可自定义部分 ====
input_path = "3d/datasets/dataset.jsonl"  # 你的完整数据集路径
train_path = "3d/datasets/train.jsonl"
val_path = "3d/datasets/val.jsonl"
split_ratio = 0.8  # 训练集比例
random_seed = 42
# ======================

# 读取 jsonl 文件（每行是一个 JSON）
with open(input_path, "r", encoding="utf-8") as f:
    data = [json.loads(line.strip()) for line in f if line.strip()]

print(f"原始数据总量: {len(data)} 条")

# 打乱
random.seed(random_seed)
random.shuffle(data)

# 划分
split_index = int(len(data) * split_ratio)
train_data = data[:split_index]
val_data = data[split_index:]

# 写入训练集
with open(train_path, "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 写入验证集
with open(val_path, "w", encoding="utf-8") as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 数据集划分完成！")
print(f"训练集: {len(train_data)} 条 -> {train_path}")
print(f"验证集: {len(val_data)} 条 -> {val_path}")
