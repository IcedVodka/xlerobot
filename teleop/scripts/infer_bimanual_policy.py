#!/usr/bin/env python

"""
XLerobot 双臂策略推理部署脚本（PC 端，通过 ZMQ 远程控制 Orin 上的机器人）

========================================================================
功能说明
========================================================================

加载训练好的 LeRobot 策略（ACT / Diffusion / VQ-BeT），在 PC 上运行推理，
通过 ZMQ 连接 Orin 上运行的 xlerobot_host，实时控制真实机器人。

不需要主臂 / Joy-Con —— 策略替代人工遥操。

========================================================================
启动命令
========================================================================

1. Orin 端先启动 Host:
    PYTHONPATH=src python -m lerobot.robots.xlerobot.xlerobot_host --robot.id=my_xlerobot

2. PC 端运行本推理脚本：
    PYTHONPATH=src python teleop/scripts/infer_bimanual_policy.py \
        --checkpoint=outputs/train/my_bimanual_act/checkpoints/last \
        --remote_ip=10.42.0.192 \
        --camera_names=left,right,head \
        --duration=120 \
        --task="My task description"

参数说明：
    --checkpoint: 训练好的策略 checkpoint 目录（含 config.json + model.safetensors）
    --remote_ip: Orin IP 地址
    --fps: 控制循环频率（默认 30 Hz）
    --duration: 单次推理最大时长（秒，默认 60）
    --task: 任务描述（多任务策略需要）
    --device: 推理设备（cuda / cpu / mps，默认自动选择）
    --camera_names: 逗号分隔的相机名称（如 'left,right,head'）
    --camera_width / --camera_height: 相机图像宽高（默认 640x480）
    --use_amp: 是否启用自动混合精度（默认 True）
    --display_data: 是否启用 rerun 可视化

========================================================================
键盘控制
========================================================================

    s = 开始策略推理
    q = 停止并退出
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor.factory import make_default_processors
from lerobot.robots.xlerobot.config_xlerobot import XLerobotClientConfig
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from teleop_record_utils import filter_arm_only_features

logger = logging.getLogger(__name__)

FPS = 30
DURATION_SEC = 60


def load_policy(checkpoint_dir: str, device: str | None = None):
    """从 checkpoint 目录加载策略、配置和前/后处理器。

    支持两种路径结构：
      - .../checkpoints/last/  -> 自动查找 pretrained_model/ 子目录
      - .../checkpoints/last/pretrained_model/  -> 直接使用
    """
    checkpoint_path = Path(checkpoint_dir).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    # 自动检测 pretrained_model/ 子目录（lerobot-train 的标准输出结构）
    if not (checkpoint_path / "config.json").exists():
        pretrained_path = checkpoint_path / "pretrained_model"
        if pretrained_path.exists() and (pretrained_path / "config.json").exists():
            logger.info(f"Auto-detected pretrained_model subdirectory")
            checkpoint_path = pretrained_path
        else:
            raise FileNotFoundError(
                f"config.json not found in {checkpoint_path} or {pretrained_path}. "
                f"Please point --checkpoint to the directory containing config.json and model.safetensors "
                f"(e.g., outputs/train/.../checkpoints/last/pretrained_model)"
            )

    logger.info(f"Loading policy config from: {checkpoint_path}")
    config = PreTrainedConfig.from_pretrained(checkpoint_path)

    if device is not None:
        config.device = device

    logger.info(f"Policy type: {config.type}, device: {config.device}")

    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(checkpoint_path, config=config)
    policy.eval()

    logger.info("Loading preprocessor/postprocessor...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint_path,
    )

    return policy, config, preprocessor, postprocessor


def action_tensor_to_dict(action_tensor: torch.Tensor, action_keys: list[str]) -> dict[str, float]:
    """将策略输出的 action tensor 转换为动作字典。"""
    action = action_tensor.squeeze(0).cpu().numpy()
    return {key: float(action[i]) for i, key in enumerate(action_keys)}


class InferenceKeyboardListener:
    """简单的键盘监听器：s=开始, q=退出。"""

    def __init__(self):
        self.start = False
        self.quit = False
        self._listener = None

    def start_listening(self) -> None:
        try:
            from pynput import keyboard
        except ImportError:
            logger.warning("pynput not available, keyboard control disabled")
            return

        def on_press(key):
            try:
                if hasattr(key, "char"):
                    char = key.char.lower() if key.char else None
                    if char == "s":
                        print("[KEYBOARD] 's' pressed: starting inference")
                        self.start = True
                    elif char == "q":
                        print("[KEYBOARD] 'q' pressed: quitting")
                        self.quit = True
                elif key == keyboard.Key.esc:
                    print("[KEYBOARD] ESC pressed: quitting")
                    self.quit = True
            except Exception:
                pass

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()

    def stop_listening(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


def main():
    parser = argparse.ArgumentParser(
        description="Deploy a trained bimanual policy on PC to control XLerobot remotely via ZMQ"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained policy checkpoint directory")
    parser.add_argument("--remote_ip", type=str, required=True, help="Orin IP address")
    parser.add_argument("--fps", type=int, default=FPS, help="Control loop frequency (Hz)")
    parser.add_argument("--duration", type=float, default=DURATION_SEC, help="Max inference duration (seconds)")
    parser.add_argument("--task", type=str, default="", help="Task description for multi-task policies")
    parser.add_argument("--device", type=str, default=None, help="Inference device (cuda/cpu/mps, default auto)")
    parser.add_argument(
        "--camera_names",
        type=str,
        default="",
        help="Comma-separated camera names (e.g. 'left,right,head')",
    )
    parser.add_argument("--camera_width", type=int, default=640, help="Camera image width")
    parser.add_argument("--camera_height", type=int, default=480, help="Camera image height")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use automatic mixed precision")
    parser.add_argument("--no_amp", action="store_false", dest="use_amp", help="Disable AMP")
    parser.add_argument("--display_data", action="store_true", help="Enable rerun visualization")
    parser.add_argument("--verbose", action="store_true", help="Show verbose logs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # -----------------------------------------------------------------------
    # 1. 加载策略
    # -----------------------------------------------------------------------
    policy, policy_config, preprocessor, postprocessor = load_policy(
        args.checkpoint,
        device=args.device,
    )
    device = torch.device(policy_config.device)

    # -----------------------------------------------------------------------
    # 2. 连接机器人（ZMQ）
    # -----------------------------------------------------------------------
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
        id="xlerobot_inference",
        cameras=camera_configs,
    )
    robot = XLerobotClient(robot_config)
    robot.connect()

    logger.info(f"Connected to robot at {args.remote_ip}:5555/5556")

    # -----------------------------------------------------------------------
    # 3. 初始化处理管线 + 推断策略训练模式
    # -----------------------------------------------------------------------
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # 构建 observation dataset features（用于 build_dataset_frame）
    obs_dataset_features = hw_to_dataset_features(robot.observation_features, "observation")

    # 从策略配置推断训练模式（arms_only vs full_body）
    policy_action_dim = policy_config.output_features["action"].shape[0]
    robot_action_dim = len(robot.action_features)
    is_arms_only = policy_action_dim < robot_action_dim

    if is_arms_only:
        logger.info(f"Policy trained in arms_only mode ({policy_action_dim}D action, robot has {robot_action_dim}D)")
        obs_dataset_features = filter_arm_only_features(obs_dataset_features)
        arm_action_keys = [k for k in robot.action_features if k.startswith(("left_arm_", "right_arm_"))]
    else:
        logger.info(f"Policy trained in full_body mode ({policy_action_dim}D action)")
        arm_action_keys = list(robot.action_features)

    # -----------------------------------------------------------------------
    # 4. 键盘监听器
    # -----------------------------------------------------------------------
    kb_listener = InferenceKeyboardListener()
    kb_listener.start_listening()

    if args.display_data:
        init_rerun(session_name="xlerobot_policy_inference")

    print("\n" + "=" * 60)
    print("Policy inference ready.")
    print("Press 's' to start inference, 'q' to quit.")
    print("=" * 60 + "\n")

    try:
        while not kb_listener.quit:
            time.sleep(0.05)
            if kb_listener.start:
                kb_listener.start = False
                break
        else:
            print("[INFO] Quit before starting inference.")
            return

        # -------------------------------------------------------------------
        # 5. 推理循环
        # -------------------------------------------------------------------
        print(f"\n[INFO] Starting inference loop at {args.fps} Hz for up to {args.duration} seconds...")
        print("[INFO] Press 'q' to stop early.\n")

        start_time = time.perf_counter()
        action_count = 0

        while not kb_listener.quit:
            loop_start = time.perf_counter()
            elapsed = loop_start - start_time
            if elapsed >= args.duration:
                print(f"\n[INFO] Duration limit ({args.duration}s) reached. Stopping.")
                break

            # 获取观察
            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)

            # 构建策略输入（与训练时数据集格式一致）
            obs_for_policy = build_dataset_frame(obs_dataset_features, obs_processed, prefix="observation")

            # 策略推理
            action = predict_action(
                observation=obs_for_policy,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=args.use_amp,
                task=args.task if args.task else None,
                robot_type=robot.name,
            )

            # 转换 action tensor -> dict
            action_dict_partial = action_tensor_to_dict(action, arm_action_keys)

            # 如果是 arms_only 模式，补齐头部和底盘字段为 0
            if is_arms_only:
                action_dict = {key: 0.0 for key in robot.action_features}
                action_dict.update(action_dict_partial)
            else:
                action_dict = action_dict_partial

            # 发送动作到机器人
            act_processed_teleop = teleop_action_processor((action_dict, obs))
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
            sent_action = robot.send_action(robot_action_to_send)

            action_count += 1

            # 可视化
            if args.display_data:
                log_rerun_data(observation=obs_processed, action=sent_action)

            # 维持频率
            dt_s = time.perf_counter() - loop_start
            precise_sleep(1 / args.fps - dt_s)

        total_time = time.perf_counter() - start_time
        actual_fps = action_count / total_time if total_time > 0 else 0
        print(f"\n[INFO] Inference finished. Sent {action_count} actions in {total_time:.1f}s (avg {actual_fps:.1f} FPS)")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        logger.exception("Inference error: %s", e)
        print(f"\n[ERROR] Inference failed: {e}")
    finally:
        kb_listener.stop_listening()
        if robot.is_connected:
            robot.disconnect()
        print("[INFO] Disconnected. Done.")


if __name__ == "__main__":
    main()
