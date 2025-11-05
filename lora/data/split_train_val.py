#!/usr/bin/env python3
"""
从训练集中随机抽取验证集的脚本

使用方法:
1. 自动推荐比例: python split_train_val.py --input train.jsonl
2. 指定比例: python split_train_val.py --input train.jsonl --val_ratio 0.1
3. 指定数量: python split_train_val.py --input train.jsonl --val_size 500
"""

import argparse
import json
import random
from pathlib import Path


def count_lines(file_path):
    """快速统计文件行数"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def recommend_val_ratio(total_size):
    """根据数据集大小推荐验证集比例"""
    if total_size < 100:
        return 0.2, "数据集较小，建议20%作为验证集"
    elif total_size < 1000:
        return 0.15, "数据集中等，建议15%作为验证集"
    elif total_size < 10000:
        return 0.1, "数据集较大，建议10%作为验证集"
    else:
        return 0.05, "数据集很大，建议5%作为验证集（至少500条）"


def split_dataset(input_path, output_train, output_val, val_ratio=None, val_size=None, seed=42):
    """
    分割数据集为训练集和验证集
    
    Args:
        input_path: 输入文件路径
        output_train: 训练集输出路径
        output_val: 验证集输出路径
        val_ratio: 验证集比例 (0-1之间)
        val_size: 验证集固定数量
        seed: 随机种子
    """
    random.seed(seed)
    
    # 统计总数据量
    print(f"正在读取数据集: {input_path}")
    total_size = count_lines(input_path)
    print(f"总数据量: {total_size} 条")
    
    # 确定验证集大小
    if val_size is not None:
        # 指定了固定数量
        if val_size >= total_size:
            raise ValueError(f"验证集数量 ({val_size}) 不能大于等于总数据量 ({total_size})")
        actual_val_size = val_size
        actual_ratio = val_size / total_size
        print(f"\n使用指定的验证集数量: {val_size} 条 (占比 {actual_ratio:.1%})")
    elif val_ratio is not None:
        # 指定了比例
        if not 0 < val_ratio < 1:
            raise ValueError(f"验证集比例必须在 0 和 1 之间，当前值: {val_ratio}")
        actual_val_size = int(total_size * val_ratio)
        actual_ratio = val_ratio
        print(f"\n使用指定的验证集比例: {val_ratio:.1%} ({actual_val_size} 条)")
    else:
        # 自动推荐
        recommended_ratio, reason = recommend_val_ratio(total_size)
        actual_val_size = max(int(total_size * recommended_ratio), min(50, total_size // 10))
        actual_ratio = actual_val_size / total_size
        print(f"\n自动推荐: {reason}")
        print(f"验证集大小: {actual_val_size} 条 (占比 {actual_ratio:.1%})")
    
    train_size = total_size - actual_val_size
    print(f"训练集大小: {train_size} 条 (占比 {train_size/total_size:.1%})")
    
    # 确认
    user_input = input("\n是否继续? [Y/n]: ").strip().lower()
    if user_input and user_input != 'y':
        print("已取消")
        return
    
    # 读取所有数据
    print("\n正在读取数据...")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 随机打乱
    print("正在随机打乱数据...")
    random.shuffle(lines)
    
    # 分割
    val_lines = lines[:actual_val_size]
    train_lines = lines[actual_val_size:]
    
    # 写入文件
    print(f"\n正在写入训练集: {output_train}")
    with open(output_train, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    print(f"正在写入验证集: {output_val}")
    with open(output_val, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    
    print(f"\n✅ 完成!")
    print(f"   训练集: {output_train} ({len(train_lines)} 条)")
    print(f"   验证集: {output_val} ({len(val_lines)} 条)")


def main():
    parser = argparse.ArgumentParser(
        description="从训练集中随机抽取验证集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动推荐比例
  python split_train_val.py --input ../dataset/sft_mini_512.jsonl
  
  # 指定10%作为验证集
  python split_train_val.py --input ../dataset/sft_mini_512.jsonl --val_ratio 0.1
  
  # 指定500条作为验证集
  python split_train_val.py --input ../dataset/sft_mini_512.jsonl --val_size 500
  
  # 自定义输出路径
  python split_train_val.py --input data.jsonl --output_train train.jsonl --output_val val.jsonl
        """
    )
    
    parser.add_argument("--input", type=str, required=True, help="输入数据集路径")
    parser.add_argument("--output_train", type=str, default=None, help="训练集输出路径 (默认: 原文件名_train.jsonl)")
    parser.add_argument("--output_val", type=str, default=None, help="验证集输出路径 (默认: 原文件名_val.jsonl)")
    parser.add_argument("--val_ratio", type=float, default=None, help="验证集比例 (0-1之间，如 0.1 表示10%%)")
    parser.add_argument("--val_size", type=int, default=None, help="验证集固定数量 (如 500)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    
    args = parser.parse_args()
    
    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 错误: 输入文件不存在: {args.input}")
        return
    
    # 确定输出路径
    if args.output_train is None:
        args.output_train = str(input_path.parent / f"{input_path.stem}_train{input_path.suffix}")
    if args.output_val is None:
        args.output_val = str(input_path.parent / f"{input_path.stem}_val{input_path.suffix}")
    
    # 检查是否会覆盖
    if Path(args.output_train).exists() or Path(args.output_val).exists():
        print("⚠️  警告: 输出文件已存在，将被覆盖:")
        if Path(args.output_train).exists():
            print(f"   - {args.output_train}")
        if Path(args.output_val).exists():
            print(f"   - {args.output_val}")
        user_input = input("是否继续? [y/N]: ").strip().lower()
        if user_input != 'y':
            print("已取消")
            return
    
    # 执行分割
    split_dataset(
        input_path=args.input,
        output_train=args.output_train,
        output_val=args.output_val,
        val_ratio=args.val_ratio,
        val_size=args.val_size,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
