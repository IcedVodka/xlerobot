#!/usr/bin/env python

"""
XLerobot 遥操作硬件公共工具模块

从 ``teleop/scripts/teleop_bimanual_keyboard_zmq.py`` 中抽取的可复用逻辑：
- 串口探测与稳定路径解析
- 键盘头部控制器
- 双臂主臂初始化
- 键盘控制器初始化
- 遥操作动作合并

该模块保持与现有遥操脚本行为一致，供新脚本组合使用。
"""

from __future__ import annotations

import glob
import logging
from typing import TYPE_CHECKING

import numpy as np

from lerobot.robots.xlerobot.config_xlerobot import XLerobotClientConfig
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.xlebi_so101_leader import XleBiSO101Leader, XleBiSO101LeaderConfig

if TYPE_CHECKING:
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. 串口工具
# ---------------------------------------------------------------------------


def find_stable_serial_ports() -> list[str]:
    """查找 ``/dev/serial/by-id/`` 下的稳定串口路径，避免 ttyACM* 插拔变化问题."""
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


# ---------------------------------------------------------------------------
# 2. 键盘头部控制器
# ---------------------------------------------------------------------------


class KeyboardHeadController:
    """使用 pynput 监听方向键，控制头部电机位置。

    注意：与 ``KeyboardTeleop`` 是独立的监听器，不共享按键状态。
    """

    def __init__(self, head_step_deg: float = 2.0):
        self.head_motor_1 = 0.0  # yaw (左右转)
        self.head_motor_2 = 0.0  # pitch (抬低头)
        self.head_step_deg = head_step_deg
        self._pressed: dict = {}
        self._listener = None

    def start(self) -> None:
        from pynput import keyboard

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _on_press(self, key) -> None:
        from pynput import keyboard

        if key in {
            keyboard.Key.up,
            keyboard.Key.down,
            keyboard.Key.left,
            keyboard.Key.right,
        }:
            self._pressed[key] = True

    def _on_release(self, key) -> None:
        self._pressed.pop(key, None)

    def get_head_action(self) -> dict[str, float]:
        from pynput import keyboard

        if self._pressed.get(keyboard.Key.up):
            self.head_motor_2 -= self.head_step_deg
        if self._pressed.get(keyboard.Key.down):
            self.head_motor_2 += self.head_step_deg
        if self._pressed.get(keyboard.Key.left):
            self.head_motor_1 -= self.head_step_deg
        if self._pressed.get(keyboard.Key.right):
            self.head_motor_1 += self.head_step_deg

        return {
            "head_motor_1.pos": self.head_motor_1,
            "head_motor_2.pos": self.head_motor_2,
        }


# ---------------------------------------------------------------------------
# 3. 硬件初始化
# ---------------------------------------------------------------------------


def init_leader_arms(left_arm_port: str, right_arm_port: str) -> XleBiSO101Leader:
    """初始化双臂 SO-101 主臂并连接。"""
    if left_arm_port == right_arm_port:
        raise ValueError(
            f"左右臂使用了相同的串口路径: {left_arm_port}\n"
            "请分别指定不同的 --left_arm_port 和 --right_arm_port"
        )

    print(f"\n[INFO] 左臂串口: {left_arm_port}")
    print(f"[INFO] 右臂串口: {right_arm_port}")

    leader_config = XleBiSO101LeaderConfig(
        id="bimanual_leader",
        left_arm_port=left_arm_port,
        right_arm_port=right_arm_port,
    )
    leader = XleBiSO101Leader(leader_config)
    print("[INFO] Connecting to leader arms...")
    leader.connect()
    return leader


def init_keyboard_controllers(head_step_deg: float = 2.0) -> tuple[KeyboardTeleop, KeyboardHeadController]:
    """初始化键盘控制器（方向键控制头部，IKJL/UOMN 控制底盘）。"""
    print("[INFO] 初始化键盘控制器（方向键=头部，IKJL=底盘）...")
    keyboard_teleop = KeyboardTeleop(KeyboardTeleopConfig(id="my_laptop_keyboard"))
    head_controller = KeyboardHeadController(head_step_deg=head_step_deg)
    keyboard_teleop.connect()
    head_controller.start()
    return keyboard_teleop, head_controller


def make_robot_client(
    remote_ip: str,
    camera_configs: dict[str, OpenCVCameraConfig],
    client_id: str = "xlerobot_infer_record",
) -> XLerobotClient:
    """初始化并连接 ZMQ 机器人客户端。"""
    robot_config = XLerobotClientConfig(
        remote_ip=remote_ip,
        id=client_id,
        cameras=camera_configs,
    )
    robot = XLerobotClient(robot_config)
    print(f"[INFO] 连接 Orin ({remote_ip})...")
    robot.connect()
    print("[INFO] 已连接!")
    if not robot.is_connected:
        raise RuntimeError("Failed to connect to robot host!")
    return robot


# ---------------------------------------------------------------------------
# 4. 遥操作动作构建
# ---------------------------------------------------------------------------


def build_teleop_action(
    leader: XleBiSO101Leader,
    head_controller: KeyboardHeadController,
    keyboard_teleop: KeyboardTeleop,
    robot: XLerobotClient,
    observation: dict,
    mode: str = "full_body",
) -> dict[str, float]:
    """合并 leader + head + keyboard base 动作，并补齐缺失字段。

    Args:
        leader: 已连接的 XleBiSO101Leader 实例。
        head_controller: 已启动的 KeyboardHeadController 实例。
        keyboard_teleop: 已连接的 KeyboardTeleop 实例。
        robot: 已连接的 XLerobotClient 实例。
        observation: 当前观测字典。
        mode: ``"upper_body"`` 时底盘动作置空，``"full_body"`` 时包含底盘。

    Returns:
        完整的机器人动作字典。
    """
    from teleop_record_utils import merge_actions

    leader_action = leader.get_action()
    head_action = head_controller.get_head_action()
    pressed_keys = np.array(list(keyboard_teleop.get_action().keys()))
    base_action = robot._from_keyboard_to_base_action(pressed_keys) or {}

    if mode == "upper_body":
        base_action = {}

    return merge_actions(
        leader_action=leader_action,
        head_action=head_action,
        base_action=base_action,
        observation=observation,
        action_features=robot.action_features,
    )
