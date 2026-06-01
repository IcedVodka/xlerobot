#!/usr/bin/env python

"""
XLerobot 双主臂远程遥操脚本 (Bimanual Leader + Keyboard + ZMQ)

在 Orin 上先启动 Host:
    PYTHONPATH=src python -m lerobot.robots.xlerobot.xlerobot_host --robot.id=my_xlerobot

在 PC 上运行本脚本:
    PYTHONPATH=src python examples/xlerobot/teleop_bimanual_zmq.py \
        --remote_ip=192.168.1.100 \
        --left_arm_port=/dev/ttyACM0 \
        --right_arm_port=/dev/ttyACM1

控制说明:
    主臂遥操: 直接移动两个 SO-101 主臂，从臂会跟随
    头部控制: T/G = 抬头/低头 (pitch), F/H = 左转/右转 (yaw)
    底盘移动: I/K = 前进/后退, J/L = 左/右平移, U/O = 左转/右转
    速度调节: N/M = 底盘速度加/减档
    退出:     ESC 键
"""

import argparse
import time

import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.xlerobot.config_xlerobot import XLerobotClientConfig
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.xlebi_so101_leader import XleBiSO101Leader, XleBiSO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30

# 头部按键映射 (独立按键模式)
HEAD_KEYMAP = {
    "head_motor_1+": "t",
    "head_motor_1-": "g",
    "head_motor_2+": "f",
    "head_motor_2-": "h",
}


def _from_keyboard_to_head_action(
    pressed_keys: set[str], current_head_pos: dict[str, float], step_deg: float
) -> dict[str, float]:
    """根据当前按键状态更新头部目标位置。

    头部使用位置控制：按下按键时改变目标角度，松开后保持当前角度。
    """
    if HEAD_KEYMAP["head_motor_1+"] in pressed_keys:
        current_head_pos["head_motor_1.pos"] += step_deg
    elif HEAD_KEYMAP["head_motor_1-"] in pressed_keys:
        current_head_pos["head_motor_1.pos"] -= step_deg

    if HEAD_KEYMAP["head_motor_2+"] in pressed_keys:
        current_head_pos["head_motor_2.pos"] += step_deg
    elif HEAD_KEYMAP["head_motor_2-"] in pressed_keys:
        current_head_pos["head_motor_2.pos"] -= step_deg

    return {
        "head_motor_1.pos": current_head_pos["head_motor_1.pos"],
        "head_motor_2.pos": current_head_pos["head_motor_2.pos"],
    }


def main():
    parser = argparse.ArgumentParser(description="XLerobot bimanual teleoperation via ZMQ")
    parser.add_argument("--remote_ip", type=str, required=True, help="Orin IP address")
    parser.add_argument(
        "--left_arm_port", type=str, default="/dev/ttyACM0", help="Left leader arm serial port"
    )
    parser.add_argument(
        "--right_arm_port", type=str, default="/dev/ttyACM1", help="Right leader arm serial port"
    )
    parser.add_argument("--fps", type=int, default=FPS, help="Control loop frequency (Hz)")
    parser.add_argument(
        "--head_step_deg", type=float, default=2.0, help="Head motor step size in degrees per keypress"
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        default="",
        help="Comma-separated camera names from host (e.g. 'head,left_wrist') for image streaming",
    )
    parser.add_argument(
        "--camera_width", type=int, default=640, help="Expected camera image width"
    )
    parser.add_argument(
        "--camera_height", type=int, default=480, help="Expected camera image height"
    )
    args = parser.parse_args()

    # 根据命令行参数动态构建相机配置，使客户端能解码图像
    camera_configs = {}
    for cam_name in args.camera_names.split(","):
        cam_name = cam_name.strip()
        if cam_name:
            camera_configs[cam_name] = OpenCVCameraConfig(
                index_or_path="",  # 仅用于 shape 元数据，客户端不实际控制相机
                fps=args.fps,
                width=args.camera_width,
                height=args.camera_height,
            )

    # 初始化远程机器人客户端 (ZMQ)
    robot_config = XLerobotClientConfig(
        remote_ip=args.remote_ip, id="xlerobot_teleop", cameras=camera_configs
    )
    robot = XLerobotClient(robot_config)

    # 初始化双手主臂
    leader_config = XleBiSO101LeaderConfig(
        id="bimanual_leader",
        left_arm_port=args.left_arm_port,
        right_arm_port=args.right_arm_port,
    )
    leader = XleBiSO101Leader(leader_config)

    # 初始化键盘遥操
    keyboard = KeyboardTeleop(KeyboardTeleopConfig(id="keyboard"))

    # 连接所有设备
    print("[INFO] Connecting to remote robot...")
    robot.connect()
    print("[INFO] Connecting to leader arms...")
    leader.connect()
    print("[INFO] Connecting to keyboard...")
    keyboard.connect()

    if not robot.is_connected or not leader.is_connected or not keyboard.is_connected:
        raise RuntimeError("Failed to connect one or more devices!")

    # 启动 rerun 可视化界面（图像、状态曲线等会自动显示）
    init_rerun(session_name="xlerobot_teleop_bimanual")

    print("\n[INFO] All devices connected. Starting teleop loop...")
    print("  Arms:  Move the leader arms directly")
    print("  Head:  T/G = pitch up/down,  F/H = yaw left/right")
    print("  Base:  I/K = forward/back,   J/L = left/right,  U/O = rotate")
    print("  Speed: N/M = speed +/-")
    print("  Exit:  ESC\n")

    # 头部当前目标位置状态 (会在按键时累加/累减)
    current_head_pos = {"head_motor_1.pos": 0.0, "head_motor_2.pos": 0.0}

    try:
        while True:
            t0 = time.perf_counter()

            # 1. 获取主臂动作 (包含 left_arm_*.pos / right_arm_*.pos，头部和底盘为占位 0)
            leader_action = leader.get_action()

            # 2. 获取键盘按键状态
            keys = keyboard.get_action()
            pressed_keys = set(keys.keys())

            # 3. 键盘 → 底盘动作 (复用 XLerobotClient 内置方法)
            keyboard_keys_array = np.array(list(pressed_keys))
            base_action = robot._from_keyboard_to_base_action(keyboard_keys_array) or {}

            # 4. 键盘 → 头部动作
            head_action = _from_keyboard_to_head_action(pressed_keys, current_head_pos, args.head_step_deg)

            # 5. 合并动作：手臂来自主臂，底盘和头部来自键盘
            #    base_action / head_action 会覆盖 leader_action 中的占位 0 值
            action = {**leader_action, **base_action, **head_action}

            # 6. 通过 ZMQ 发送到 Orin 端的机器人
            robot.send_action(action)

            # 7. 接收观测（包含图像，由 rerun 自动显示）
            obs = robot.get_observation()
            log_rerun_data(observation=obs, action=action)

            # 8. 维持目标频率
            dt = time.perf_counter() - t0
            precise_sleep(max(1.0 / args.fps - dt, 0.0))

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        print("[INFO] Disconnecting...")
        if leader.is_connected:
            leader.disconnect()
        if robot.is_connected:
            robot.disconnect()
        if keyboard.is_connected:
            keyboard.disconnect()
        print("[INFO] Done")


if __name__ == "__main__":
    main()
