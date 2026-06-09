#!/usr/bin/env python

"""
XLerobot 双臂策略推理部署脚本 (PC端)

========================================================================
功能说明
========================================================================

在 PC 端加载训练好的 LeRobot 策略模型（ACT / Diffusion / VQBeT / π0 等），
通过 ZMQ 与 Orin 上的 xlerobot_host 通信，实现自主推理控制。

支持任意 action 维度的模型：
- 双臂 only (12维): 模型只输出双臂关节，头部保持当前位置，底盘停止
- 双臂 + 头部 (14维): 模型输出双臂+头部，底盘停止
- 全身 full_body (17维): 模型输出全部动作

脚本会自动检测模型的输出维度，对缺失字段做智能补齐：
- 缺失的 .pos 字段 → 使用观测中的当前值（保持不动）
- 缺失的 .vel 字段 → 置为 0.0（停止运动）

========================================================================
完整启动命令
========================================================================

1. Orin 端先启动 Host:
    PYTHONPATH=src python -m lerobot.robots.xlerobot.xlerobot_host --robot.id=my_xlerobot

2. PC 端运行推理脚本（dataset_repo_id 自动从训练配置推断）：
    PYTHONPATH=src python teleop/scripts/infer_bimanual_policy.py \
        --model_path=outputs/train/my_keyboard_act/checkpoints/050000/pretrained_model \
        --remote_ip=10.42.0.192 \
        --camera_names=left,right,head

3. 从 HuggingFace Hub 加载模型（需手动指定数据集）：
    PYTHONPATH=src python teleop/scripts/infer_bimanual_policy.py \
        --model_path=lerobot/my_policy \
        --dataset_repo_id=lerobot/my_dataset \
        --remote_ip=10.42.0.192 \
        --camera_names=left,right,head

========================================================================
键盘控制（推理过程中）
========================================================================

    → (右箭头)   = 结束当前 episode 并保存
    ← (左箭头)   = 结束当前 episode 并重新录制
    ESC          = 完全退出推理流程

========================================================================
参数说明
========================================================================

    --model_path:       预训练模型路径（pretrained_model 目录）
    --dataset_repo_id:  数据集 repo_id（可选，默认从 model_path/train_config.json 自动推断）
    --dataset_root:     数据集本地根目录（可选）
    --remote_ip:        Orin IP 地址
    --fps:              推理频率 Hz（默认 30）
    --camera_names:     相机名称，逗号分隔（默认 left,right,head）
    --num_episodes:     推理 episode 数量（默认无限循环）
    --episode_time_s:   每 episode 最大时长（默认 300 秒）
    --task_description: 任务描述（用于推理帧构建）
    --display_data:     启用 rerun 可视化
    --device:           推理设备（默认 auto，自动检测 cuda/mps/cpu）
    --warmup_steps:     预热步数（前几帧不发动作，默认 2）
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 把 teleop/src 加入路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.utils import make_robot_action
from lerobot.utils.constants import OBS_STR
from lerobot.robots.xlerobot.config_xlerobot import XLerobotClientConfig
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient
from lerobot.utils.control_utils import init_keyboard_listener, predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = logging.getLogger(__name__)

FPS = 30
EPISODE_TIME_SEC = 300
TASK_DESCRIPTION = "My task description"


import json as json_module


def infer_dataset_repo_id(model_path: str) -> str | None:
    """从训练配置中自动推断数据集 repo_id。

    LeRobot 训练时会在 pretrained_model/train_config.json 中保存 dataset.repo_id。
    如果找到就返回，否则返回 None（需要用户手动指定）。
    """
    train_config_path = Path(model_path) / "train_config.json"
    if not train_config_path.exists():
        return None
    try:
        with open(train_config_path) as f:
            cfg = json_module.load(f)
        repo_id = cfg.get("dataset", {}).get("repo_id")
        if repo_id:
            print(f"[INFO] 从 train_config.json 自动推断 dataset_repo_id: {repo_id}")
            return repo_id
    except Exception:
        pass
    return None


def fill_missing_action_keys(
    predicted_action: dict[str, float],
    observation: dict[str, np.ndarray],
    robot_action_features: dict[str, type],
) -> dict[str, float]:
    """将模型预测的动作补齐为完整的机器人动作字典。

    策略：
    - 模型已预测的字段 → 直接使用
    - 缺失的 .pos 字段 → 使用观测中的当前值（保持不动）
    - 缺失的 .vel 字段 → 置为 0.0（停止运动）

    Args:
        predicted_action: 模型输出的动作字典（可能缺少部分字段）
        observation: 当前观测字典（包含各关节当前位置）
        robot_action_features: 机器人完整的 action features（17个键）

    Returns:
        补齐后的完整动作字典，可直接 send_action 发送给机器人
    """
    full_action: dict[str, float] = {}

    for key in robot_action_features:
        if key in predicted_action:
            full_action[key] = predicted_action[key]
        elif key.endswith(".vel"):
            full_action[key] = 0.0
        elif key.endswith(".pos"):
            # 从观测中获取当前位置值；观测中的值可能是 numpy scalar
            val = observation.get(key, 0.0)
            full_action[key] = float(val) if val is not None else 0.0
        else:
            full_action[key] = 0.0

    return full_action


def infer_loop(
    robot: XLerobotClient,
    policy: PreTrainedPolicy,
    preprocessor,
    postprocessor,
    ds_features: dict,
    device: torch.device,
    events: dict,
    fps: int = FPS,
    episode_time_s: float = EPISODE_TIME_SEC,
    task_description: str = TASK_DESCRIPTION,
    display_data: bool = False,
    warmup_steps: int = 2,
):
    """单 episode 推理循环。

    流程：
    1. 获取观测（ZMQ 接收 Orin 发送的图像 + 状态）
    2. 构建推理帧 → 预处理 → 策略推理 → 后处理
    3. 补齐缺失动作字段 → 发送动作（ZMQ 推送到 Orin）
    4. 维持目标频率

    Args:
        robot: 已连接的 XLerobotClient 实例
        policy: 已加载的 PreTrainedPolicy 模型（ACT / Diffusion / VQBeT / π0 等）
        preprocessor: 观测预处理器
        postprocessor: 动作后处理器
        ds_features: 数据集 features 字典
        device: 推理设备
        events: 键盘事件字典（exit_early / rerecord_episode / stop_recording）
        fps: 控制频率
        episode_time_s: 本 episode 最大时长
        task_description: 任务描述
        display_data: 是否启用 rerun 可视化
        warmup_steps: 预热步数（前几帧只接收观测不发送动作，让图像流稳定）
    """
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    timestamp = 0.0
    start_episode_t = time.perf_counter()
    step_count = 0

    while timestamp < episode_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # ------------------------------------------------------------------
        # 1. 获取观测
        # ------------------------------------------------------------------
        obs = robot.get_observation()

        # 预热阶段：只接收观测，不发动作（让图像流稳定）
        if step_count < warmup_steps:
            step_count += 1
            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(1 / fps - dt_s)
            timestamp = time.perf_counter() - start_episode_t
            continue

        # ------------------------------------------------------------------
        # 2. 策略推理
        # ------------------------------------------------------------------
        try:
            # 从原始观测构建数据集格式的帧（只提取需要的字段）
            # predict_action 内部会调用 prepare_observation_for_inference 做 Tensor 转换
            obs_frame = build_dataset_frame(ds_features, obs, prefix=OBS_STR)

            action_tensor = predict_action(
                observation=obs_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=task_description,
                robot_type=robot.robot_type,
            )

            # 3. 转换为动作字典（只包含模型输出的字段）
            predicted_action = make_robot_action(action_tensor, ds_features)

            # 4. 补齐缺失字段 → 完整17维动作
            full_action = fill_missing_action_keys(
                predicted_action=predicted_action,
                observation=obs,
                robot_action_features=robot.action_features,
            )

        except Exception as exc:
            logger.exception("Inference failed at step %d: %s", step_count, exc)
            print(f"[WARN] 推理失败，发送零动作: {exc}")
            # 推理失败时发送零速度、保持当前位置的保守动作
            full_action = fill_missing_action_keys(
                predicted_action={},
                observation=obs,
                robot_action_features=robot.action_features,
            )

        # ------------------------------------------------------------------
        # 5. 发送动作到 Orin
        # ------------------------------------------------------------------
        robot.send_action(full_action)

        if display_data:
            log_rerun_data(observation=obs, action=full_action)

        step_count += 1
        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(1 / fps - dt_s)
        timestamp = time.perf_counter() - start_episode_t

    print(f"[INFO] Episode 结束，共执行 {step_count} 步")


def main():
    parser = argparse.ArgumentParser(
        description="XLerobot bimanual policy inference on PC via ZMQ"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="预训练模型路径（pretrained_model 目录，或 HuggingFace Hub ID）",
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        default=None,
        help="数据集 repo_id（可选，默认从 model_path/train_config.json 自动推断）",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="数据集本地根目录（默认 ~/.cache/huggingface/lerobot）",
    )
    parser.add_argument(
        "--remote_ip",
        type=str,
        required=True,
        help="Orin IP 地址",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=FPS,
        help="推理频率 Hz（默认 30）",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        default="left,right,head",
        help="相机名称，逗号分隔（默认 left,right,head）",
    )
    parser.add_argument(
        "--camera_width",
        type=int,
        default=640,
        help="相机图像宽度",
    )
    parser.add_argument(
        "--camera_height",
        type=int,
        default=480,
        help="相机图像高度",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=0,
        help="推理 episode 数量（默认 0 = 无限循环）",
    )
    parser.add_argument(
        "--episode_time_s",
        type=int,
        default=EPISODE_TIME_SEC,
        help="每 episode 最大时长（秒）",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default=TASK_DESCRIPTION,
        help="任务描述",
    )
    parser.add_argument(
        "--display_data",
        action="store_true",
        help="启用 rerun 可视化",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="推理设备: auto/cuda/mps/cpu（默认 auto）",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2,
        help="预热步数（前几帧只接收观测不发动作，默认 2）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细日志",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # ===================================================================
    # 1. 加载模型
    # ===================================================================
    print(f"[INFO] 加载模型: {args.model_path}")

    # 1.1 读取配置，自动识别策略类型
    policy_cfg = PreTrainedConfig.from_pretrained(args.model_path)
    policy_type = policy_cfg.type
    print(f"[INFO] 策略类型: {policy_type}")

    # 1.2 动态获取策略类并加载
    PolicyClass = get_policy_class(policy_type)
    policy: PreTrainedPolicy = PolicyClass.from_pretrained(args.model_path)

    # 自动检测设备
    if args.device == "auto":
        device = get_safe_torch_device(policy.config.device)
    else:
        device = torch.device(args.device)
    policy.to(device)
    policy.eval()

    # 打印模型信息
    action_shape = policy.config.output_features["action"].shape
    action_names = getattr(
        policy.config.output_features["action"], "names", None
    )
    print(f"[INFO] 模型设备: {device}")
    print(f"[INFO] 模型输出 action shape: {action_shape}")
    if action_names:
        print(f"[INFO] 模型输出 action 字段 ({len(action_names)}个):")
        for name in action_names:
            print(f"       - {name}")

    # ===================================================================
    # 2. 加载数据集 metadata（用于 preprocessor / postprocessor）
    # ===================================================================
    dataset_repo_id = args.dataset_repo_id
    if dataset_repo_id is None:
        dataset_repo_id = infer_dataset_repo_id(args.model_path)
    if dataset_repo_id is None:
        raise ValueError(
            "无法自动推断 dataset_repo_id。请显式指定:\n"
            "  --dataset_repo_id=<your_dataset_name>"
        )

    print(f"[INFO] 加载数据集 metadata: {dataset_repo_id}")
    dataset_metadata = LeRobotDatasetMetadata(
        dataset_repo_id,
        root=args.dataset_root,
    )
    ds_features = dataset_metadata.features
    print(f"[INFO] 数据集 fps: {dataset_metadata.fps}")

    # ===================================================================
    # 3. 构建 preprocessor / postprocessor
    # ===================================================================
    print("[INFO] 构建 preprocessor / postprocessor...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.model_path,
        dataset_stats=dataset_metadata.stats,
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
        },
    )

    # ===================================================================
    # 4. 初始化 ZMQ 客户端
    # ===================================================================
    camera_configs = {}
    for cam_name in args.camera_names.split(","):
        cam_name = cam_name.strip()
        if cam_name:
            camera_configs[cam_name] = OpenCVCameraConfig(
                index_or_path="",
                fps=args.fps,
                width=args.camera_width,
                height=args.camera_height,
            )

    robot_config = XLerobotClientConfig(
        remote_ip=args.remote_ip,
        id="xlerobot_infer",
        cameras=camera_configs,
    )
    robot = XLerobotClient(robot_config)

    # ===================================================================
    # 5. 连接 Orin
    # ===================================================================
    print(f"[INFO] 连接 Orin ({args.remote_ip})...")
    robot.connect()
    print("[INFO] 已连接!")

    if not robot.is_connected:
        raise RuntimeError("Failed to connect to robot host!")

    # ===================================================================
    # 6. 初始化键盘监听
    # ===================================================================
    listener, events = init_keyboard_listener()

    # ===================================================================
    # 7. 启动 rerun 可视化（如需要）
    # ===================================================================
    if args.display_data:
        init_rerun(session_name="xlerobot_inference")

    # ===================================================================
    # 8. 推理主循环
    # ===================================================================
    episode_idx = 0
    try:
        while True:
            if args.num_episodes > 0 and episode_idx >= args.num_episodes:
                print(f"[INFO] 已完成 {args.num_episodes} 个 episode，退出")
                break

            if events["stop_recording"]:
                print("[INFO] 收到退出信号，停止推理")
                break

            log_say(
                f"开始第 {episode_idx + 1} 轮推理"
                if args.num_episodes > 0
                else f"开始第 {episode_idx + 1} 轮推理（无限循环）"
            )
            print(f"\n{'='*50}")
            print(f"[INFO] Episode {episode_idx + 1} 开始")
            print(f"{'='*50}")

            infer_loop(
                robot=robot,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                ds_features=ds_features,
                device=device,
                events=events,
                fps=args.fps,
                episode_time_s=args.episode_time_s,
                task_description=args.task_description,
                display_data=args.display_data,
                warmup_steps=args.warmup_steps,
            )

            # 处理重新录制请求
            if events["rerecord_episode"]:
                events["rerecord_episode"] = False
                events["exit_early"] = False
                print("[INFO] 重新执行当前 episode...")
                continue

            episode_idx += 1

            # 重置阶段（短暂停顿，给用户时间复位环境）
            if not events["stop_recording"]:
                print("[INFO] 推理轮次结束。按 → 开始下一轮，ESC 退出")
                # 等待一小段时间或直到用户按键
                reset_start = time.perf_counter()
                while time.perf_counter() - reset_start < 3.0:
                    if events["exit_early"]:
                        events["exit_early"] = False
                        break
                    if events["stop_recording"]:
                        break
                    time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[INFO] 被用户中断")
    finally:
        print("[INFO] 断开连接...")
        if robot.is_connected:
            robot.disconnect()
        if listener is not None:
            listener.stop()
        print("[INFO] 已退出")


if __name__ == "__main__":
    main()
