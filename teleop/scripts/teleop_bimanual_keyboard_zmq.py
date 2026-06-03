#!/usr/bin/env python
"""
XLerobot 双主臂 + 键盘遥操脚本（ZMQ 远程 + 可选数据录制）

控制说明:
    主臂遥操: 直接移动两个 SO-101 主臂，从臂会跟随
    头部控制: T/G = 抬头/低头 (pitch), F/H = 左转/右转 (yaw)
    底盘移动: I/K = 前进/后退, J/L = 左/右平移, U/O = 左转/右转
    速度调节: N/M = 底盘速度加/减档

录制控制:
    Space = 开始新 episode
    R     = 重录当前 episode
    S     = 停止录制
    ESC   = 退出程序

启动命令:
    1. Orin 端先启动 Host:
        PYTHONPATH=src python -m lerobot.robots.xlerobot.xlerobot_host 

    2. PC 端运行本脚本:
        PYTHONPATH=src python teleop/scripts/teleop_bimanual_keyboard_zmq.py \
            --remote_ip=10.42.0.192 \
            --left_arm_port=/dev/serial/by-id/usb-... \
            --right_arm_port=/dev/serial/by-id/usb-... \
            --camera_names=left,right,head

    3. 录制模式（固定底盘+移动底盘两套数据集）:
        PYTHONPATH=src python teleop/scripts/teleop_bimanual_keyboard_zmq.py \
            --remote_ip=10.42.0.192 \
            --left_arm_port=/dev/serial/by-id/usb-... \
            --right_arm_port=/dev/serial/by-id/usb-... \
            --camera_names=left,right,head \
            --record \
            --dataset_repo_id_fixed=user/xlerobot_fixed \
            --dataset_repo_id_mobile=user/xlerobot_mobile \
            --single_task="Pick and place task"
"""

import argparse
import glob
import logging
import sys
import time
from pathlib import Path

import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.xlerobot.config_xlerobot import XLerobotClientConfig
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.xlebi_so101_leader import XleBiSO101Leader, XleBiSO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from teleop.src.teleop_record import TeleopRecordManager, make_head_action

logger = logging.getLogger(__name__)
FPS = 30


def find_stable_serial_ports() -> list[str]:
    """查找 /dev/serial/by-id/ 下的稳定串口路径。"""
    by_id_paths = sorted(glob.glob("/dev/serial/by-id/*"))
    return by_id_paths


def resolve_arm_port(port_arg: str | None, fallback_label: str) -> str:
    """解析串口参数：优先使用用户指定的稳定路径，未指定时自动探测并提示。"""
    if port_arg:
        return port_arg
    stable_ports = find_stable_serial_ports()
    tty_ports = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    print(f"\n[WARN] --{fallback_label} 未指定，尝试自动检测串口...")
    if stable_ports:
        print("[INFO] 检测到以下稳定串口路径 (/dev/serial/by-id/)：")
        for i, p in enumerate(stable_ports, 1):
            print(f"       {i}. {p}")
        print(f"       例如: --{fallback_label}={stable_ports[0]}")
        return stable_ports[0]
    elif tty_ports:
        print(f"[WARN] 回退到动态路径: {tty_ports[0]}")
        return tty_ports[0]
    else:
        raise RuntimeError("未检测到任何串口设备，请检查 USB 连接。")


def parse_args():
    parser = argparse.ArgumentParser(description="XLerobot bimanual teleoperation via ZMQ with recording")
    # 网络
    parser.add_argument("--remote_ip", type=str, required=True, help="Orin IP address")
    parser.add_argument("--port_zmq_cmd", type=int, default=5555, help="ZMQ command port")
    parser.add_argument("--port_zmq_obs", type=int, default=5556, help="ZMQ observation port")
    # 主臂串口
    parser.add_argument("--left_arm_port", type=str, default=None, help="Left leader arm serial port")
    parser.add_argument("--right_arm_port", type=str, default=None, help="Right leader arm serial port")
    parser.add_argument("--list_ports", action="store_true", help="仅列出可用串口并退出")
    # 相机
    parser.add_argument("--camera_names", type=str, default="", help="Comma-separated camera names")
    parser.add_argument("--camera_width", type=int, default=640, help="Camera image width")
    parser.add_argument("--camera_height", type=int, default=480, help="Camera image height")
    # 录制
    parser.add_argument("--record", action="store_true", help="启用数据录制")
    parser.add_argument("--dataset_repo_id_fixed", type=str, default=None, help="固定底盘数据集 repo_id")
    parser.add_argument("--dataset_repo_id_mobile", type=str, default=None, help="移动底盘数据集 repo_id")
    parser.add_argument("--single_task", type=str, default="xlerobot teleop task", help="任务描述")
    parser.add_argument("--fps", type=int, default=FPS, help="Control loop frequency")
    parser.add_argument("--head_step_deg", type=float, default=2.0, help="Head motor step size (degrees)")
    parser.add_argument("--display_data", action="store_true", help="Display rerun visualization")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def main():
    args = parse_args()

    # 日志
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 仅列出串口
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

    # 解析串口
    left_port = resolve_arm_port(args.left_arm_port, "left_arm_port")
    right_port = resolve_arm_port(args.right_arm_port, "right_arm_port")
    if left_port == right_port:
        raise ValueError(f"左右臂使用了相同的串口路径: {left_port}")
    print(f"\n[INFO] 左臂串口: {left_port}")
    print(f"[INFO] 右臂串口: {right_port}")

    # 相机配置
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

    # 初始化远程机器人客户端 (ZMQ)
    robot_config = XLerobotClientConfig(
        remote_ip=args.remote_ip,
        port_zmq_cmd=args.port_zmq_cmd,
        port_zmq_observations=args.port_zmq_obs,
        cameras=camera_configs,
    )
    robot = XLerobotClient(robot_config)

    # 初始化双主臂
    leader_config = XleBiSO101LeaderConfig(
        id="bimanual_leader",
        left_arm_port=left_port,
        right_arm_port=right_port,
    )
    leader = XleBiSO101Leader(leader_config)

    # 初始化键盘遥操
    keyboard = KeyboardTeleop(KeyboardTeleopConfig(id="keyboard"))

    # 连接设备
    print("[INFO] Connecting to remote robot...")
    robot.connect()
    print("[INFO] Connecting to leader arms...")
    leader.connect()
    print("[INFO] Connecting to keyboard...")
    keyboard.connect()

    if not robot.is_connected or not leader.is_connected or not keyboard.is_connected:
        raise RuntimeError("Failed to connect one or more devices!")

    # 初始化录制管理器
    recorder = None
    if args.record:
        if not args.dataset_repo_id_fixed and not args.dataset_repo_id_mobile:
            raise ValueError("--record 需要至少指定 --dataset_repo_id_fixed 或 --dataset_repo_id_mobile")
        recorder = TeleopRecordManager(
            repo_id_fixed=args.dataset_repo_id_fixed,
            repo_id_mobile=args.dataset_repo_id_mobile,
            robot=robot,
            fps=args.fps,
            single_task=args.single_task,
        )
        print(f"\n[INFO] Recording enabled:")
        print(f"       Fixed-base: {args.dataset_repo_id_fixed or 'N/A'}")
        print(f"       Mobile:     {args.dataset_repo_id_mobile or 'N/A'}")
        print(f"       Task:       {args.single_task}")

    # 初始化 rerun
    if args.display_data:
        init_rerun(session_name="xlerobot_teleop_keyboard")

    print("\n[INFO] All devices connected. Starting teleop loop...")
    print("  Arms:  Move the leader arms directly")
    print("  Head:  T/G = pitch up/down,  F/H = yaw left/right")
    print("  Base:  I/K = forward/back,   J/L = left/right,  U/O = rotate")
    print("  Speed: N/M = speed +/-")
    if args.record:
        print("\n  Recording:")
        print("    Space = Start new episode")
        print("    R     = Re-record current episode")
        print("    S     = Stop recording")
    print("  Exit:  ESC\n")

    # 头部当前目标位置
    current_head_pos = {"head_motor_1.pos": 0.0, "head_motor_2.pos": 0.0}

    # 录制按键状态（用于检测按键按下事件）
    prev_pressed = set()
    prev_status = ""

    try:
        while True:
            t0 = time.perf_counter()

            # 1. 获取主臂动作
            leader_action = leader.get_action()

            # 2. 获取键盘按键状态
            keys = keyboard.get_action()
            pressed_keys = set(keys.keys())

            # 3. 检测按键按下事件（从 0→1）
            newly_pressed = pressed_keys - prev_pressed

            # 4. 录制控制
            if recorder is not None:
                if " " in newly_pressed and not recorder.active_recorder.is_recording:
                    recorder.start_episode()
                elif "r" in newly_pressed and recorder.active_recorder.is_recording:
                    recorder.rerecord_episode()
                elif "s" in newly_pressed and recorder.active_recorder.is_recording:
                    recorder.save_episode()
                elif "1" in newly_pressed and recorder.recorder_fixed:
                    recorder.switch_dataset(fixed_base=True)
                    print("[INFO] Switched to fixed-base dataset")
                elif "2" in newly_pressed and recorder.recorder_mobile:
                    recorder.switch_dataset(fixed_base=False)
                    print("[INFO] Switched to mobile dataset")

            # 5. 退出检测
            if "esc" in pressed_keys or "\x1b" in pressed_keys:
                print("\n[INFO] ESC pressed, exiting...")
                break

            # 6. 键盘 → 底盘动作
            keyboard_keys_array = np.array(list(pressed_keys))
            base_action = robot._from_keyboard_to_base_action(keyboard_keys_array) or {}

            # 7. 键盘 → 头部动作
            head_action = make_head_action(pressed_keys, current_head_pos, args.head_step_deg)

            # 8. 合并动作
            action = {**leader_action, **base_action, **head_action}

            # 9. 发送动作到机器人
            robot.send_action(action)

            # 10. 接收观测
            obs = robot.get_observation()

            # 11. 录制帧
            if recorder is not None:
                recorder.record_frame(obs, action)

            # 12. 可视化
            if args.display_data:
                log_rerun_data(observation=obs, action=action)

            # 13. 打印录制状态
            if recorder is not None:
                status = recorder.get_status()
                if status != prev_status:
                    print(f"\r{status}", end="", flush=True)
                    prev_status = status

            # 14. 维持频率
            prev_pressed = pressed_keys
            dt = time.perf_counter() - t0
            precise_sleep(max(1.0 / args.fps - dt, 0.0))

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        print("\n[INFO] Disconnecting...")
        if recorder is not None:
            recorder.finalize()
        if leader.is_connected:
            leader.disconnect()
        if robot.is_connected:
            robot.disconnect()
        if keyboard.is_connected:
            keyboard.disconnect()
        print("[INFO] Done")


if __name__ == "__main__":
    main()
