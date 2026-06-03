#!/usr/bin/env python

"""
XLerobot 双主臂远程遥操脚本 (Bimanual Leader + Keyboard + ZMQ)

========================================================================
当前 PC 上检测到的遥操臂串口（/dev/serial/by-id/）
========================================================================

    usb-1a86_USB_Single_Serial_5A46084903-if00  ->  ../../ttyACM0
    usb-1a86_USB_Single_Serial_58FA093104-if00  ->  ../../ttyACM1

**注意**：ttyACM0 和 ttyACM1 由内核按枚举顺序分配，插拔或重启后编号
可能互换。下方完整命令已改用 /dev/serial/by-id/ 稳定路径，不再变化。

左右对应关系验证方法（只需做一次）：
    1. 两个臂都插上，执行:  ls -la /dev/serial/by-id/
    2. 拔掉物理上的"左臂"USB，再次执行，看哪个 symlink 消失
    3. 消失的 symlink 对应的就是左臂，填入 --left_arm_port

如果遥操时发现左右反了，直接交换 --left_arm_port 和 --right_arm_port
的值即可，不需要重新标定。

========================================================================
完整启动命令（已填好 IP、相机、串口，确认左右后直接运行）
========================================================================

1. Orin 端先启动 Host:
    PYTHONPATH=src python -m lerobot.robots.xlerobot.xlerobot_host --robot.id=my_xlerobot

2. PC 端运行本脚本（请根据上面的验证结果确认左右后再执行）：
    PYTHONPATH=src python examples/xlerobot/teleop_bimanual_zmq.py \
        --remote_ip=10.42.0.192 \
        --left_arm_port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46084903-if00 \
        --right_arm_port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_58FA093104-if00 \
        --camera_names=left,right,head

3. 仅查看可用串口：
    PYTHONPATH=src python examples/xlerobot/teleop_bimanual_zmq.py --list_ports

========================================================================
重新标定（手臂行程范围或跟随不准时执行）
========================================================================

Step 1: 删除旧标定文件
    rm ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/bimanual_leader_left.json
    rm ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/bimanual_leader_right.json

Step 2: 运行上面的完整命令，按终端提示操作
    - 先松开所有关节，把手臂移到各关节中间位置 → 按 ENTER
    - 再逐个关节缓慢移动全范围（shoulder_pan → shoulder_lift →
      elbow_flex → wrist_flex → wrist_roll → gripper）
    - 全部走完 → 按 ENTER 停止记录
    - 标定数据自动保存，下次启动会复用

========================================================================
控制说明
========================================================================
    主臂遥操: 直接移动两个 SO-101 主臂，从臂会跟随
    头部控制: T/G = 抬头/低头 (pitch), F/H = 左转/右转 (yaw)
    底盘移动: I/K = 前进/后退, J/L = 左/右平移, U/O = 左转/右转
    速度调节: N/M = 底盘速度加/减档
    退出:     ESC 键
"""

import argparse
import glob
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


def find_stable_serial_ports() -> list[str]:
    """查找 /dev/serial/by-id/ 下的稳定串口路径，避免 ttyACM* 插拔变化问题."""
    by_id_paths = sorted(glob.glob("/dev/serial/by-id/*"))
    return by_id_paths


def resolve_arm_port(port_arg: str | None, fallback_label: str) -> str:
    """解析串口参数：优先使用用户指定的稳定路径，未指定时自动探测并提示."""
    if port_arg:
        return port_arg

    stable_ports = find_stable_serial_ports()
    tty_ports = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))

    print(f"\n[WARN] --{fallback_label} 未指定，尝试自动检测串口...")
    if stable_ports:
        print("[INFO] 检测到以下稳定串口路径 (/dev/serial/by-id/)：")
        for i, p in enumerate(stable_ports, 1):
            print(f"       {i}. {p}")
        print(f"[INFO] 建议固定使用以上路径，避免插拔后设备名变化。")
        print(f"       例如: --{fallback_label}={stable_ports[0]}")
        if len(stable_ports) >= 2:
            print(f"       左右臂分别指定不同的路径，不要重复。")
        return stable_ports[0]
    elif tty_ports:
        print(f"[WARN] 未找到 /dev/serial/by-id/ 稳定路径，回退到动态路径: {tty_ports[0]}")
        print(f"       建议将串口芯片插到固定的 USB 口，或使用 udev 规则创建固定别名。")
        return tty_ports[0]
    else:
        raise RuntimeError("未检测到任何串口设备，请检查 USB 连接。")


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
        "--left_arm_port",
        type=str,
        default=None,
        help=(
            "Left leader arm serial port. "
            "强烈推荐使用 /dev/serial/by-id/ 下的稳定路径，"
            "避免 ttyACM* 插拔后变化。"
        ),
    )
    parser.add_argument(
        "--right_arm_port",
        type=str,
        default=None,
        help=(
            "Right leader arm serial port. "
            "强烈推荐使用 /dev/serial/by-id/ 下的稳定路径，"
            "避免 ttyACM* 插拔后变化。"
        ),
    )
    parser.add_argument("--list_ports", action="store_true", help="仅列出可用串口并退出")
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

    # 仅列出串口并退出
    if args.list_ports:
        print("=" * 50)
        print("可用稳定串口路径 (/dev/serial/by-id/):")
        print("=" * 50)
        for p in find_stable_serial_ports():
            print(f"  {p}")
        print("\n动态串口路径 (ttyACM/ttyUSB):")
        for p in sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")):
            print(f"  {p}")
        return

    # 解析并固定串口路径
    left_port = resolve_arm_port(args.left_arm_port, "left_arm_port")
    right_port = resolve_arm_port(args.right_arm_port, "right_arm_port")

    if left_port == right_port:
        raise ValueError(
            f"左右臂使用了相同的串口路径: {left_port}\n"
            "请分别指定不同的 --left_arm_port 和 --right_arm_port"
        )

    print(f"\n[INFO] 左臂串口: {left_port}")
    print(f"[INFO] 右臂串口: {right_port}")

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
        left_arm_port=left_port,
        right_arm_port=right_port,
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
