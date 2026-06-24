#!/usr/bin/env python

"""
XLerobot 双臂策略推理 + 人类介入纠正采集脚本

========================================================================
功能说明
========================================================================

在 PC 端同时加载训练好的 LeRobot 策略模型并连接遥操作主臂，通过 ZMQ
与 Orin 上的 xlerobot_host 通信。

支持两种控制模式：
    0 = autonomous（策略执行）
    1 = intervention（人类接管，policy 不再运行）

每个 episode 从 autonomous 开始，按 Space 单向切到 intervention，本 episode 内
不再切回；下一 episode 自动重置回 autonomous。

采集的数据集会在标准 LeRobot v3 格式基础上增加一个字段：
    control_mode: int64 标量，每帧记录当前控制模式

========================================================================
完整启动命令
========================================================================

1. Orin 端先启动 Host:
    PYTHONPATH=src python -m lerobot.robots.xlerobot.xlerobot_host --robot.id=my_xlerobot

2. PC 端运行本脚本（录制推理 + 介入数据）：
    PYTHONPATH=src python teleop/scripts/infer_and_record_bimanual.py \
        --model_path=outputs/train/my_keyboard_act/checkpoints/last/pretrained_model \
        --remote_ip=10.42.0.192 \
        --repo_id=my_intervention_dataset1 \
        --left_arm_port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46084903-if00 \
        --right_arm_port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_58FA093104-if00 \
        --camera_names=left,right,head \
        --display_data

========================================================================
键盘控制
========================================================================

模式切换：
    Space = 切到 intervention（单向，本 episode 内不再切回）

Episode 控制（数字键）：
    1 = 开始 / 跳过重置 并进入下一轮
    2 = 结束当前 episode
    3 = 重新录制当前 episode
    4 = 完全退出录制流程

头部控制（方向键）：
    ↑ = 抬头,  ↓ = 低头,  ← = 左转,  → = 右转

底盘控制（键盘）：
    I = 前进,  K = 后退
    J = 左移,  L = 右移
    U = 左转,  O = 右转
    N = 速度加档,  M = 速度减档

========================================================================
参数说明
========================================================================

    --model_path:       预训练模型路径（pretrained_model 目录）
    --dataset_repo_id:  数据集 repo_id（可选，默认从 model_path/train_config.json 推断）
    --dataset_root:     数据集本地根目录（可选）
    --remote_ip:        Orin IP 地址
    --repo_id:          采集数据集标识名称（必须指定）
    --left_arm_port:    左主臂串口稳定路径
    --right_arm_port:   右主臂串口稳定路径
    --fps:              控制频率 Hz（默认 30）
    --camera_names:     相机名称，逗号分隔（默认 left,right,head）
    --mode:             upper_body（双臂+头部）或 full_body（全身）
    --num_episodes:     录制 episode 数量（默认 50）
    --episode_time_s:   每 episode 最大时长（默认 300 秒）
    --reset_time_s:     episode 间重置时间（默认 10 秒）
    --task_description: 任务描述
    --display_data:     启用 rerun 可视化
    --device:           推理设备（默认 auto）
    --verbose:          显示详细日志
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# 把 teleop/src 加入路径，以便导入共用工具
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference_utils import load_policy_and_processors
from intervention_utils import ModeToggleListener, decide_action
from teleop_hw_utils import (
    build_teleop_action,
    init_keyboard_controllers,
    init_leader_arms,
    make_robot_client,
    resolve_arm_port,
)
from teleop_record_utils import (
    EpisodeKeyboardListener,
    filter_arm_head_features,
    run_recording_session,
    sync_episode_events,
)

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.utils.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

logger = logging.getLogger(__name__)

FPS = 30
NUM_EPISODES = 50
EPISODE_TIME_SEC = 300
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "My task description"


def build_camera_configs(
    camera_names: str,
    fps: int,
    width: int,
    height: int,
) -> dict[str, OpenCVCameraConfig]:
    """根据逗号分隔的相机名称构建相机配置字典。"""
    camera_configs = {}
    for cam_name in camera_names.split(","):
        cam_name = cam_name.strip()
        if cam_name:
            camera_configs[cam_name] = OpenCVCameraConfig(
                index_or_path="",
                fps=fps,
                width=width,
                height=height,
            )
    return camera_configs


def create_or_load_dataset(
    repo_id: str,
    fps: int,
    robot,
    mode: str,
    dataset_root: str | None,
    resume: bool,
) -> LeRobotDataset:
    """创建或加载带有 control_mode 字段的数据集。"""
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    full_features = {**action_features, **obs_features}

    if mode == "upper_body":
        dataset_features = filter_arm_head_features(full_features)
        print("[INFO] upper_body 模式：数据集包含双臂 + 头部字段")
    else:
        dataset_features = full_features
        print("[INFO] full_body 模式：数据集包含全身字段")

    # 新增 control_mode 字段
    dataset_features["control_mode"] = {"dtype": "int64", "shape": (1,), "names": None}

    if resume:
        dataset = LeRobotDataset(
            repo_id,
            root=dataset_root,
            batch_encoding_size=1,
        )
        dataset.start_image_writer(num_threads=4)
        sanity_check_dataset_robot_compatibility(dataset, robot, fps, dataset_features)
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
            root=dataset_root,
        )

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="XLerobot bimanual policy inference with human intervention recording"
    )

    # 模型相关
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
        "--device",
        type=str,
        default="auto",
        help="推理设备: auto/cuda/mps/cpu（默认 auto）",
    )

    # 机器人 / ZMQ 相关
    parser.add_argument(
        "--remote_ip",
        type=str,
        required=True,
        help="Orin IP 地址",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        default="left,right,head",
        help="相机名称，逗号分隔（默认 left,right,head）",
    )
    parser.add_argument("--camera_width", type=int, default=640, help="相机图像宽度")
    parser.add_argument("--camera_height", type=int, default=480, help="相机图像高度")

    # 遥操作硬件相关
    parser.add_argument(
        "--left_arm_port",
        type=str,
        default=None,
        help="左主臂串口稳定路径（如 /dev/serial/by-id/...）",
    )
    parser.add_argument(
        "--right_arm_port",
        type=str,
        default=None,
        help="右主臂串口稳定路径（如 /dev/serial/by-id/...）",
    )
    parser.add_argument(
        "--head_step_deg",
        type=float,
        default=2.0,
        help="头部电机每帧步进角度（默认 2.0）",
    )

    # 录制相关
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="采集数据集标识名称",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full_body",
        choices=["upper_body", "full_body"],
        help="录制模式：upper_body 采集双臂+头部数据，full_body 采集全身数据",
    )
    parser.add_argument("--fps", type=int, default=FPS, help="控制频率 Hz（默认 30）")
    parser.add_argument(
        "--num_episodes", type=int, default=NUM_EPISODES, help="录制 episode 数量"
    )
    parser.add_argument(
        "--episode_time_s", type=int, default=EPISODE_TIME_SEC, help="每 episode 最大时长（秒）"
    )
    parser.add_argument(
        "--reset_time_s", type=int, default=RESET_TIME_SEC, help="episode 间重置时间（秒）"
    )
    parser.add_argument("--task_description", type=str, default=TASK_DESCRIPTION, help="任务描述")
    parser.add_argument(
        "--resume", action="store_true", help="在已有数据集上继续录制"
    )
    parser.add_argument(
        "--display_data", action="store_true", help="启用 rerun 可视化"
    )

    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # ------------------------------------------------------------------
    # 1. 加载策略模型
    # ------------------------------------------------------------------
    policy, preprocessor, postprocessor, ds_features, device = load_policy_and_processors(
        model_path=args.model_path,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        device_arg=args.device,
    )

    # ------------------------------------------------------------------
    # 2. 初始化机器人客户端
    # ------------------------------------------------------------------
    camera_configs = build_camera_configs(
        camera_names=args.camera_names,
        fps=args.fps,
        width=args.camera_width,
        height=args.camera_height,
    )
    robot = make_robot_client(
        remote_ip=args.remote_ip,
        camera_configs=camera_configs,
        client_id="xlerobot_infer_record",
    )

    # ------------------------------------------------------------------
    # 3. 初始化遥操作硬件
    # ------------------------------------------------------------------
    left_port = resolve_arm_port(args.left_arm_port, "left_arm_port")
    right_port = resolve_arm_port(args.right_arm_port, "right_arm_port")
    leader = init_leader_arms(left_port, right_port)
    keyboard_teleop, head_controller = init_keyboard_controllers(args.head_step_deg)

    if not robot.is_connected or not leader.is_connected:
        raise RuntimeError("Failed to connect one or more devices!")

    # ------------------------------------------------------------------
    # 4. 创建 / 加载数据集（带 control_mode）
    # ------------------------------------------------------------------
    dataset = create_or_load_dataset(
        repo_id=args.repo_id,
        fps=args.fps,
        robot=robot,
        mode=args.mode,
        dataset_root=args.dataset_root,
        resume=args.resume,
    )

    # ------------------------------------------------------------------
    # 5. 初始化各种监听器
    # ------------------------------------------------------------------
    kb_listener = EpisodeKeyboardListener()
    kb_listener.start()

    mode_listener = ModeToggleListener()
    mode_listener.start()

    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "discard_current_episode": False,
    }

    if args.display_data:
        init_rerun(session_name="xlerobot_infer_and_record")

    # 处理管线
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    print("\n[INFO] All devices connected. Starting inference + recording loop...")
    print("  Mode toggle: Space = 切到 intervention（单向，本 episode 内不再切回）")
    print("  每个 episode 从 autonomous 开始")
    print("  Episode controls:")
    print("    1 = 开始/跳过重置")
    print("    2 = 结束当前 episode")
    print("    3 = 重新录制")
    print("    4 = 完全退出")
    print("  Exit: Ctrl+C\n")

    # ------------------------------------------------------------------
    # 6. build_action 回调：核心决策逻辑
    # ------------------------------------------------------------------
    def build_action(obs: dict) -> dict:
        # autonomous 期间让头部控制器内部目标跟随真实角度，避免切入介入时跳变
        if mode_listener.control_mode == 0:
            head_controller.sync_to_observation(obs)

        # 处理 episode 控制事件
        kb_ev = kb_listener.consume_events()
        sync_episode_events(kb_ev, events)

        # 采集遥操作动作
        teleop_action = build_teleop_action(
            leader=leader,
            head_controller=head_controller,
            keyboard_teleop=keyboard_teleop,
            robot=robot,
            observation=obs,
            mode=args.mode,
        )

        # 根据当前模式决定最终动作
        final_action = decide_action(
            mode=mode_listener.control_mode,
            observation=obs,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            ds_features=ds_features,
            device=device,
            teleop_action=teleop_action,
            robot=robot,
            task_description=args.task_description,
        )

        return final_action

    def episode_start_callback() -> None:
        """每个 episode 开始前重置策略状态，并回到 autonomous 模式。"""
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        mode_listener.reset()
        print("[INFO] 新 episode 开始，已重置 policy / preprocessor / postprocessor，模式回到 autonomous")

    # 可选：在 rerun 里实时绘制 control_mode 曲线
    def post_frame_callback() -> None:
        if args.display_data:
            import rerun as rr

            rr.log("control_mode", rr.Scalars(float(mode_listener.control_mode)))

    # ------------------------------------------------------------------
    # 7. 启动录制会话
    # ------------------------------------------------------------------
    try:
        run_recording_session(
            robot=robot,
            leader=leader,
            events=events,
            dataset=dataset,
            args=args,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            build_action=build_action,
            control_mode_provider=lambda: mode_listener.control_mode,
            episode_start_callback=episode_start_callback,
            post_frame_callback=post_frame_callback,
        )
    finally:
        print("[INFO] Disconnecting...")
        kb_listener.stop()
        mode_listener.stop()
        head_controller.stop()
        if keyboard_teleop.is_connected:
            keyboard_teleop.disconnect()
        if leader.is_connected:
            leader.disconnect()
        if robot.is_connected:
            robot.disconnect()
        print("[INFO] Done")


if __name__ == "__main__":
    main()
