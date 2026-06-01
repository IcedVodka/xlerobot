#!/usr/bin/env python

"""
XLerobot 双主臂遥操本地调试脚本 (Bimanual Leader + Keyboard + Mock ZMQ)

纯本地运行，不连接远程机器人，用于检查发送数据格式是否正确。

运行:
    PYTHONPATH=src python examples/xlerobot/teleop_bimanual_zmq_mock.py \
        --left_arm_port=/dev/ttyACM0 \
        --right_arm_port=/dev/ttyACM1

控制说明:
    主臂遥操: 直接移动两个 SO-101 主臂
    头部控制: T/G = 抬头/低头 (pitch), F/H = 左转/右转 (yaw)
    底盘移动: I/K = 前进/后退, J/L = 左/右平移, U/O = 左转/右转
    速度调节: N/M = 底盘速度加/减档
    退出:     ESC 键
"""

import argparse
import json
import time

import numpy as np

from lerobot.robots.xlerobot.config_xlerobot import XLerobotConfig
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.xlebi_so101_leader import XleBiSO101Leader, XleBiSO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep

FPS = 30

# 头部按键映射 (独立按键模式)
HEAD_KEYMAP = {
    "head_motor_1+": "t",
    "head_motor_1-": "g",
    "head_motor_2+": "f",
    "head_motor_2-": "h",
}

STATE_ORDER = (
    "left_arm_shoulder_pan.pos",
    "left_arm_shoulder_lift.pos",
    "left_arm_elbow_flex.pos",
    "left_arm_wrist_flex.pos",
    "left_arm_wrist_roll.pos",
    "left_arm_gripper.pos",
    "right_arm_shoulder_pan.pos",
    "right_arm_shoulder_lift.pos",
    "right_arm_elbow_flex.pos",
    "right_arm_wrist_flex.pos",
    "right_arm_wrist_roll.pos",
    "right_arm_gripper.pos",
    "head_motor_1.pos",
    "head_motor_2.pos",
    "x.vel",
    "y.vel",
    "theta.vel",
)


def _from_keyboard_to_head_action(
    pressed_keys: set[str], current_head_pos: dict[str, float], step_deg: float
) -> dict[str, float]:
    """根据当前按键状态更新头部目标位置。"""
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


class MockXLerobotClient(XLerobotClient):
    """模拟远程机器人客户端：只打印/记录发送的 action，不连接真实 ZMQ。"""

    def __init__(self):
        # 构造一个假的 config，跳过父类 __init__ 中的 ZMQ 初始化
        self.config = XLerobotConfig()
        self.config.id = "xlerobot_mock"
        self.config.remote_ip = "127.0.0.1"
        self.config.port_zmq_cmd = 5555
        self.config.port_zmq_observations = 5556
        self.config.teleop_keys = self.config.teleop_keys
        self.config.polling_timeout_ms = 15
        self.config.connect_timeout_s = 5
        self.config.cameras = {}

        self.id = self.config.id
        self.remote_ip = self.config.remote_ip
        self.port_zmq_cmd = self.config.port_zmq_cmd
        self.port_zmq_observations = self.config.port_zmq_observations
        self.teleop_keys = self.config.teleop_keys
        self.polling_timeout_ms = self.config.polling_timeout_ms
        self.connect_timeout_s = self.config.connect_timeout_s

        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None
        self.last_frames = {}
        self.last_remote_state = {}

        self.speed_levels = [
            {"xy": 0.1, "theta": 30},  # slow
            {"xy": 0.2, "theta": 60},  # medium
            {"xy": 0.3, "theta": 90},  # fast
        ]
        self.speed_index = 0

        self._is_connected = False
        self.logs = {}
        self._action_counter = 0

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> None:
        print("[MOCK] 模拟连接远程机器人 (不连接真实 ZMQ)")
        self._is_connected = True

    def disconnect(self) -> None:
        print("[MOCK] 模拟断开远程机器人")
        self._is_connected = False

    def send_action(self, action: dict) -> dict:
        """打印 action 到终端，方便调试。"""
        self._action_counter += 1

        # 按固定顺序提取数值向量
        action_vec = [action.get(k, 0.0) for k in STATE_ORDER]

        # 打印简化版：只显示非零值，避免刷屏
        nonzero = {k: round(v, 4) for k, v in zip(STATE_ORDER, action_vec) if abs(v) > 1e-6}

        print(f"\n{'='*60}")
        print(f"[MOCK] Action #{self._action_counter}  @ {time.strftime('%H:%M:%S')}")
        print(f"{'-'*60}")
        # 打印完整 JSON，方便复制查看
        print(f"[完整 JSON] {json.dumps(action, ensure_ascii=False, default=str)}")
        print(f"{'-'*60}")
        # 打印各关节/速度值，格式化对齐
        print(f"{'Key':<35} {'Value':>12}")
        print(f"{'-'*48}")
        for k, v in zip(STATE_ORDER, action_vec):
            marker = "  <--" if abs(v) > 1e-6 else ""
            print(f"  {k:<33} {v:>12.4f}{marker}")
        print(f"{'='*60}")

        return {**action, "action": np.array(action_vec, dtype=np.float32)}

    def get_observation(self) -> dict:
        """返回模拟的观测数据（全零状态 + 空帧）。"""
        obs = {"observation.state": np.zeros(len(STATE_ORDER), dtype=np.float32)}
        for k in STATE_ORDER:
            obs[k] = 0.0
        return obs


def main():
    parser = argparse.ArgumentParser(description="XLerobot bimanual teleoperation (MOCK debug mode)")
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
    args = parser.parse_args()

    # 初始化模拟远程机器人客户端
    robot = MockXLerobotClient()

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
    print("[INFO] Connecting mock remote robot...")
    robot.connect()
    print("[INFO] Connecting to leader arms...")
    leader.connect()
    print("[INFO] Connecting to keyboard...")
    keyboard.connect()

    if not robot.is_connected or not leader.is_connected or not keyboard.is_connected:
        raise RuntimeError("Failed to connect one or more devices!")

    print("\n[INFO] All devices connected. Starting MOCK teleop loop...")
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

            # 1. 获取主臂动作
            leader_action = leader.get_action()

            # 2. 获取键盘按键状态
            keys = keyboard.get_action()
            pressed_keys = set(keys.keys())

            # 3. 键盘 → 底盘动作
            keyboard_keys_array = np.array(list(pressed_keys))
            base_action = robot._from_keyboard_to_base_action(keyboard_keys_array) or {}

            # 4. 键盘 → 头部动作
            head_action = _from_keyboard_to_head_action(pressed_keys, current_head_pos, args.head_step_deg)

            # 5. 合并动作
            action = {**leader_action, **base_action, **head_action}

            # 6. 发送 action（Mock 模式下只打印）
            robot.send_action(action)

            # 7. 接收模拟观测
            obs = robot.get_observation()

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
