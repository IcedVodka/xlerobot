#!/usr/bin/env python

"""
XLerobot 双臂策略训练脚本

基于 LeRobot 的 lerobot-train，预设 ACT 策略和本地数据集路径。

用法：
    # 基础训练（ACT 策略，默认参数）
    PYTHONPATH=src python teleop/scripts/train_bimanual_policy.py \
        --repo_id=my_bimanual_dataset \
        --output_dir=outputs/train/my_bimanual_act

    # 指定数据集根目录（如果不在默认缓存路径）
    PYTHONPATH=src python teleop/scripts/train_bimanual_policy.py \
        --repo_id=my_bimanual_dataset \
        --dataset_root=/home/gml-cwl/.cache/huggingface/lerobot \
        --output_dir=outputs/train/my_bimanual_act

    # 调整训练参数
    PYTHONPATH=src python teleop/scripts/train_bimanual_policy.py \
        --repo_id=my_bimanual_dataset \
        --batch_size=4 \
        --steps=50000 \
        --output_dir=outputs/train/my_bimanual_act

    # 使用 Diffusion 策略（代替 ACT）
    PYTHONPATH=src python teleop/scripts/train_bimanual_policy.py \
        --repo_id=my_bimanual_dataset \
        --policy=diffusion \
        --output_dir=outputs/train/my_bimanual_diffusion

    # 恢复训练
    PYTHONPATH=src python teleop/scripts/train_bimanual_policy.py \
        --repo_id=my_bimanual_dataset \
        --resume \
        --config_path=outputs/train/my_bimanual_act/train_config.json

参数说明：
    --repo_id: 数据集名称（对应本地目录名）
    --dataset_root: 数据集父目录（默认 ~/.cache/huggingface/lerobot）
    --policy: 策略类型（act / diffusion / vqbet，默认 act）
    --batch_size: 训练批次大小（默认 4，小数据集建议不改太大）
    --steps: 总训练步数（默认 50000）
    --eval_freq: 评估频率（步数，默认 5000）
    --save_freq: 保存 checkpoint 频率（步数，默认 5000）
    --num_workers: DataLoader  workers（默认 4）
    --image_transforms: 是否启用图像增强（默认 True）
    --output_dir: 训练输出目录（包含 checkpoint、日志、配置）
    --resume: 从已有 checkpoint 恢复训练
    --config_path: resume 时使用的配置文件路径
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train a bimanual policy with LeRobot")
    parser.add_argument("--repo_id", type=str, required=True, help="数据集名称（repo_id）")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="数据集本地存储根目录（默认 ~/.cache/huggingface/lerobot）",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="act",
        choices=["act", "diffusion", "vqbet"],
        help="策略类型（默认 act，最适合双臂操作）",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="训练批次大小")
    parser.add_argument("--steps", type=int, default=50000, help="总训练步数")
    parser.add_argument("--eval_freq", type=int, default=5000, help="评估频率（步）")
    parser.add_argument("--save_freq", type=int, default=5000, help="保存 checkpoint 频率（步）")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--image_transforms",
        type=bool,
        default=True,
        help="是否启用图像增强",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="训练输出目录（如 outputs/train/my_run）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从已有 checkpoint 恢复训练",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="resume 时使用的 train_config.json 路径",
    )
    args = parser.parse_args()

    # 构建 lerobot-train 命令
    # draccus choice class 需要用 --policy.type=xxx 的格式
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--policy.type={args.policy}",
        f"--dataset.repo_id={args.repo_id}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--eval_freq={args.eval_freq}",
        f"--save_freq={args.save_freq}",
        f"--num_workers={args.num_workers}",
        f"--output_dir={args.output_dir}",
    ]

    # 数据集根目录
    if args.dataset_root is not None:
        dataset_path = Path(args.dataset_root).expanduser().resolve()
        cmd.append(f"--dataset.root={dataset_path}")

    # 图像增强
    cmd.append(f"--dataset.image_transforms.enable={args.image_transforms}")

    # Resume
    if args.resume:
        cmd.append("--resume")
        if args.config_path:
            cmd.append(f"--config_path={args.config_path}")

    print(f"[INFO] Running: {' '.join(cmd)}")
    print(f"[INFO] Policy: {args.policy}")
    print(f"[INFO] Dataset: {args.repo_id}")
    print(f"[INFO] Output dir: {args.output_dir}")
    print(f"[INFO] Training steps: {args.steps}")
    print(f"[INFO] Batch size: {args.batch_size}")

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
