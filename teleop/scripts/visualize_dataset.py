#!/usr/bin/env python

"""
XLerobot 数据集可视化脚本

封装 lerobot-dataset-viz，默认使用本地缓存路径，简化命令行。

用法：
    # 可视化默认路径下的数据集第 0 轮
    PYTHONPATH=src python teleop/scripts/visualize_dataset.py \
        --repo_id=my_bimanual_dataset

    # 指定轮次
    PYTHONPATH=src python teleop/scripts/visualize_dataset.py \
        --repo_id=my_bimanual_dataset \
        --episode-index=2

    # 指定自定义存储路径
    PYTHONPATH=src python teleop/scripts/visualize_dataset.py \
        --repo_id=my_bimanual_dataset \
        --dataset_root=~/datasets \
        --episode-index=0

    # 保存为 .rrd 文件（不启动实时 viewer，之后用 rerun 打开）
    PYTHONPATH=src python teleop/scripts/visualize_dataset.py \
        --repo_id=my_bimanual_dataset \
        --save \
        --output-dir=./viz_output
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Visualize a LeRobot dataset")
    parser.add_argument("--repo_id", type=str, required=True, help="数据集名称（repo_id）")
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="要可视化的 episode 序号（默认 0）",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="数据集本地存储根目录（默认 ~/.cache/huggingface/lerobot）",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="保存为 .rrd 文件，不启动实时 viewer",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help=".rrd 文件输出目录（配合 --save 使用）",
    )
    args = parser.parse_args()

    # 构建 lerobot-dataset-viz 命令
    # --root 需要指向数据集本身的根目录（包含 meta/ data/ videos/），不是父目录
    if args.dataset_root is not None:
        dataset_path = Path(args.dataset_root).expanduser().resolve() / args.repo_id
    else:
        from lerobot.utils.constants import HF_LEROBOT_HOME
        dataset_path = HF_LEROBOT_HOME / args.repo_id

    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_dataset_viz",
        "--repo-id", args.repo_id,
        "--episode-index", str(args.episode_index),
        "--root", str(dataset_path),
    ]

    if args.save:
        cmd.extend(["--save", "1"])
        cmd.extend(["--output-dir", args.output_dir])

    print(f"[INFO] Running: {' '.join(cmd)}")
    print(f"[INFO] Dataset: {args.repo_id}")
    print(f"[INFO] Episode index: {args.episode_index}")
    print(f"[INFO] Dataset path: {dataset_path}")

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
