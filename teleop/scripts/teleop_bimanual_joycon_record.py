#!/usr/bin/env python

"""
XLerobot 双臂遥操 + 双 Joy-Con 控制头部和底盘 + 数据采集

========================================================================
控制说明
========================================================================

主臂遥操: 直接移动两个 SO-101 主臂，从臂会跟随

左 Joy-Con (头部):
    摇杆 = 抬头/低头/左转/右转
    D-pad 上  = 开始一轮 / 跳过重置 进入下一轮
    D-pad 下  = 结束当前 episode
    D-pad 左  = 重新录制当前 episode
    D-pad 右  = 完全退出录制流程

右 Joy-Con (底盘):
    摇杆 = 前进/后退/左移/右移
    Y    = 左转,  A = 右转
    X    = 底盘速度加档,  B = 底盘速度减档

后备键盘控制（当 Joy-Con 不可用时）:
    1    = 开始一轮 / 跳过重置
    2    = 结束当前 episode
    3    = 重新录制
    4    = 完全退出

========================================================================
两种模式
========================================================================

--mode arms_only :
    控制双臂+头部，底盘不动（置零），数据集中只记录双臂数据。
    适合训练仅控制双臂的策略（头部作为场景变化来源）。

--mode full_body (默认):
    全身控制（双臂+头部+底盘），数据集中记录全身数据。

========================================================================
启动命令
========================================================================

1. Orin 端先启动 Host:
    PYTHONPATH=src python -m lerobot.robots.xlerobot.xlerobot_host --robot.id=my_xlerobot

2. PC 端运行本脚本：
    PYTHONPATH=src python teleop/scripts/teleop_bimanual_joycon_record.py \
        --remote_ip=10.42.0.192 \
        --left_arm_port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46084903-if00 \
        --right_arm_port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_58FA093104-if00 \
        --repo_id=my_bimanual_dataset \
        --mode=full_body \
        --camera_names=left,right,head

        PYTHONPATH=src python teleop/scripts/teleop_bimanual_joycon_record.py \
        --remote_ip=10.42.0.192 \
        --left_arm_port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46084903-if00 \
        --right_arm_port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_58FA093104-if00 \
        --repo_id=my_bimanual_dataset \
        --mode=arms_only \
        --camera_names=left,right,head \
        --display_data


3. 仅查看可用串口：
    PYTHONPATH=src python teleop/scripts/teleop_bimanual_joycon_record.py --list_ports
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
import time
from pathlib import Path

import numpy as np

# 把 teleop/src 加入路径，以便导入共用工具
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from teleop_record_utils import (
    EpisodeKeyboardListener,
    clear_phase_exit_event,
    filter_arm_only_features,
    make_round_prompt,
    merge_actions,
    record_loop,
    run_recording_session,
    sync_episode_events,
)

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.robots.xlerobot.config_xlerobot import XLerobotClientConfig
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient
from lerobot.teleoperators.xlebi_so101_leader import XleBiSO101Leader, XleBiSO101LeaderConfig
from lerobot.utils.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = logging.getLogger(__name__)

FPS = 30
NUM_EPISODES = 50
EPISODE_TIME_SEC = 300
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "My task description"

BASE_SPEED_LEVELS = [
    {"xy": 0.05, "theta": 15},
    {"xy": 0.10, "theta": 30},
    {"xy": 0.15, "theta": 45},
    {"xy": 0.20, "theta": 60},
    {"xy": 0.30, "theta": 90},
]

STICK_UP_THRESHOLD = 3000
STICK_DOWN_THRESHOLD = 1000
STICK_LEFT_THRESHOLD = 1000
STICK_RIGHT_THRESHOLD = 3000


# ---------------------------------------------------------------------------
# 串口工具（与 teleop_bimanual_zmq.py 保持一致）
# ---------------------------------------------------------------------------


def find_stable_serial_ports() -> list[str]:
    by_id_paths = sorted(glob.glob("/dev/serial/by-id/*"))
    return by_id_paths


def resolve_arm_port(port_arg: str | None, fallback_label: str) -> str:
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
            print("       左右臂分别指定不同的路径，不要重复。")
        return stable_ports[0]
    elif tty_ports:
        print(f"[WARN] 未找到稳定路径，回退到动态路径: {tty_ports[0]}")
        return tty_ports[0]
    else:
        raise RuntimeError("未检测到任何串口设备，请检查 USB 连接。")


# ---------------------------------------------------------------------------
# Joy-Con 控制器（扩展 episode 控制）
# ---------------------------------------------------------------------------


class RecordingDualJoyconController:
    """双 Joy-Con 控制器 — 左手柄控制头部，右手柄控制底盘，D-pad 控制 episode。"""

    def __init__(self, head_step_deg: float = 2.0):
        try:
            from joyconrobotics import JoyconRobotics
        except ImportError:
            print("错误: 未安装 joyconrobotics 库")
            print("请执行: cd joycon-robotics && pip install -e . && sudo make install")
            raise

        self.left_joycon = JoyconRobotics(
            device="left",
            dof_speed=[2, 2, 2, 1, 1, 1],
            without_rest_init=True,
        )
        self.right_joycon = JoyconRobotics(
            device="right",
            dof_speed=[2, 2, 2, 1, 1, 1],
            without_rest_init=True,
        )

        self.head_motor_1 = 0.0
        self.head_motor_2 = 0.0
        self.head_step_deg = head_step_deg
        self.speed_level = 1
        self.prev_x = 0
        self.prev_b = 0

        # Episode 控制去抖
        self.prev_episode_buttons = {"up": 0, "down": 0, "right": 0, "left": 0}
        self.episode_events = {"end": False, "rerecord": False, "stop": False}

    def _read_left_buttons(self) -> dict:
        j = self.left_joycon.joycon
        return {
            "StickH": j.get_stick_left_horizontal(),
            "StickV": j.get_stick_left_vertical(),
        }

    def _read_right_buttons(self) -> dict:
        j = self.right_joycon.joycon
        return {
            "X": j.get_button_x(),
            "B": j.get_button_b(),
            "Y": j.get_button_y(),
            "A": j.get_button_a(),
            "StickH": j.get_stick_right_horizontal(),
            "StickV": j.get_stick_right_vertical(),
        }

    def _get_head_action(self, buttons: dict) -> dict:
        stick_v = buttons["StickV"]
        stick_h = buttons["StickH"]

        if stick_v > STICK_UP_THRESHOLD:
            self.head_motor_2 -= self.head_step_deg
        elif stick_v < STICK_DOWN_THRESHOLD:
            self.head_motor_2 += self.head_step_deg

        if stick_h < STICK_LEFT_THRESHOLD:
            self.head_motor_1 -= self.head_step_deg
        elif stick_h > STICK_RIGHT_THRESHOLD:
            self.head_motor_1 += self.head_step_deg

        return {
            "head_motor_1.pos": self.head_motor_1,
            "head_motor_2.pos": self.head_motor_2,
        }

    def _get_base_action(self, buttons: dict) -> dict:
        speed = BASE_SPEED_LEVELS[self.speed_level]
        x_cmd = y_cmd = theta_cmd = 0.0

        stick_v = buttons["StickV"]
        stick_h = buttons["StickH"]

        if stick_v > STICK_UP_THRESHOLD:
            x_cmd += speed["xy"]
        elif stick_v < STICK_DOWN_THRESHOLD:
            x_cmd -= speed["xy"]

        if stick_h < STICK_LEFT_THRESHOLD:
            y_cmd += speed["xy"]
        elif stick_h > STICK_RIGHT_THRESHOLD:
            y_cmd -= speed["xy"]

        if buttons["Y"]:
            theta_cmd += speed["theta"]
        if buttons["A"]:
            theta_cmd -= speed["theta"]

        return {"x.vel": x_cmd, "y.vel": y_cmd, "theta.vel": theta_cmd}

    def _update_speed(self, buttons: dict) -> None:
        if buttons["X"] and not self.prev_x:
            self.speed_level = min(self.speed_level + 1, len(BASE_SPEED_LEVELS) - 1)
            print(f"[JOYCON] 底盘速度加档: level {self.speed_level + 1}/{len(BASE_SPEED_LEVELS)}")
        if buttons["B"] and not self.prev_b:
            self.speed_level = max(self.speed_level - 1, 0)
            print(f"[JOYCON] 底盘速度减档: level {self.speed_level + 1}/{len(BASE_SPEED_LEVELS)}")
        self.prev_x = buttons["X"]
        self.prev_b = buttons["B"]

    def update(self) -> tuple[dict, dict]:
        """更新双 Joy-Con 状态，返回 (base_action, head_action)。"""
        left_buttons = self._read_left_buttons()
        right_buttons = self._read_right_buttons()

        head_action = self._get_head_action(left_buttons)
        base_action = self._get_base_action(right_buttons)
        self._update_speed(right_buttons)

        return base_action, head_action

    def update_episode_controls(self) -> None:
        """检测左手柄 D-pad 上升沿，设置 episode 事件。

        映射：上=开始/跳过重置，下=结束，左=重录，右=退出。
        """
        j = self.left_joycon.joycon
        buttons = {
            "up": j.get_button_up(),
            "down": j.get_button_down(),
            "left": j.get_button_left(),
            "right": j.get_button_right(),
        }
        for name, val in buttons.items():
            if val and not self.prev_episode_buttons[name]:
                if name == "up":
                    self.episode_events["start_next"] = True
                    print("[JOYCON] D-pad 上：开始/跳过重置")
                elif name == "down":
                    self.episode_events["end_current"] = True
                    print("[JOYCON] D-pad 下：结束当前 episode")
                elif name == "left":
                    self.episode_events["rerecord"] = True
                    print("[JOYCON] D-pad 左：重新录制")
                elif name == "right":
                    self.episode_events["stop"] = True
                    print("[JOYCON] D-pad 右：完全退出")
            self.prev_episode_buttons[name] = val

    def consume_episode_events(self) -> dict:
        """读取并清空当前帧的 episode 事件。"""
        ev = self.episode_events.copy()
        self.episode_events = {
            "start_next": False,
            "end_current": False,
            "rerecord": False,
            "stop": False,
        }
        return ev

    def disconnect(self) -> None:
        if hasattr(self.left_joycon, "disconnect"):
            self.left_joycon.disconnect()
        if hasattr(self.right_joycon, "disconnect"):
            self.right_joycon.disconnect()


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="XLerobot bimanual teleoperation + dataset recording with dual Joy-Con"
    )
    parser.add_argument("--remote_ip", type=str, default=None, help="Orin IP address")
    parser.add_argument(
        "--left_arm_port",
        type=str,
        default=None,
        help="左主臂串口，强烈推荐使用 /dev/serial/by-id/ 下的稳定路径",
    )
    parser.add_argument(
        "--right_arm_port",
        type=str,
        default=None,
        help="右主臂串口，强烈推荐使用 /dev/serial/by-id/ 下的稳定路径",
    )
    parser.add_argument("--list_ports", action="store_true", help="仅列出可用串口并退出")
    parser.add_argument("--fps", type=int, default=FPS, help="Control loop frequency (Hz)")
    parser.add_argument(
        "--head_step_deg", type=float, default=2.0, help="头部电机每帧步进角度"
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        default="",
        help="逗号分隔的相机名称（如 'left,right,head'）",
    )
    parser.add_argument("--camera_width", type=int, default=640, help="相机图像宽度")
    parser.add_argument("--camera_height", type=int, default=480, help="相机图像高度")

    # 录制相关参数
    parser.add_argument(
        "--mode",
        type=str,
        default="full_body",
        choices=["arms_only", "full_body"],
        help="录制模式：arms_only 只采双臂数据，full_body 采集全身数据",
    )
    parser.add_argument("--repo_id", type=str, default=None, help="数据集标识名称")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="数据集本地存储根目录（默认 ~/.cache/huggingface/lerobot/<repo_id>）",
    )
    parser.add_argument("--num_episodes", type=int, default=NUM_EPISODES, help="录制 episode 数量")
    parser.add_argument(
        "--episode_time_s", type=int, default=EPISODE_TIME_SEC, help="每 episode 最大时长（秒）"
    )
    parser.add_argument(
        "--reset_time_s", type=int, default=RESET_TIME_SEC, help="episode 间重置时间（秒）"
    )
    parser.add_argument("--task_description", type=str, default=TASK_DESCRIPTION, help="任务描述")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="在已有数据集上继续录制",
    )
    parser.add_argument("--display_data", action="store_true", help="启用 rerun 可视化")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 仅列出串口时不需要 remote_ip/repo_id
    if not args.list_ports:
        if not args.remote_ip:
            parser.error("--remote_ip is required (unless using --list_ports)")
        if not args.repo_id:
            parser.error("--repo_id is required (unless using --list_ports)")

    # 仅列出串口并退出
    if args.list_ports:
        print("=" * 50)
        print("可用稳定串口路径 (/dev/serial/by-id/)：")
        print("=" * 50)
        for p in find_stable_serial_ports():
            print(f"  {p}")
        print("\n动态串口路径 (ttyACM/ttyUSB)：")
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
    print(f"[INFO] 录制模式: {args.mode}")

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

    # 初始化设备
    robot_config = XLerobotClientConfig(
        remote_ip=args.remote_ip, id="xlerobot_teleop", cameras=camera_configs
    )
    robot = XLerobotClient(robot_config)
    leader_config = XleBiSO101LeaderConfig(
        id="bimanual_leader",
        left_arm_port=left_port,
        right_arm_port=right_port,
    )
    leader = XleBiSO101Leader(leader_config)
    joycon = RecordingDualJoyconController(head_step_deg=args.head_step_deg)

    # 处理管线
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # 数据集 features
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    full_features = {**action_features, **obs_features}

    if args.mode == "arms_only":
        dataset_features = filter_arm_only_features(full_features)
        print("[INFO] arms_only 模式：数据集仅包含双臂字段")
    else:
        dataset_features = full_features
        print("[INFO] full_body 模式：数据集包含全身字段")

    # 连接设备
    robot.connect()
    leader.connect()

    # 创建/加载数据集
    if args.resume:
        dataset = LeRobotDataset(
            args.repo_id,
            root=args.dataset_root,
            batch_encoding_size=1,
        )
        dataset.start_image_writer(num_threads=4)
        sanity_check_dataset_robot_compatibility(dataset, robot, args.fps, dataset_features)
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
            root=args.dataset_root,
        )

    # 键盘监听器（后备 episode 控制：1=开始 2=结束 3=重录 4=退出）
    kb_listener = EpisodeKeyboardListener()
    kb_listener.start()
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "discard_current_episode": False,
    }

    if args.display_data:
        init_rerun(session_name="xlerobot_teleop_joycon_record")

    if not robot.is_connected or not leader.is_connected:
        raise RuntimeError("Failed to connect one or more devices!")

    print("\n[INFO] All devices connected. Starting recording loop...")
    print("  Arms:       Move the leader arms directly")
    print("  Left Joy-Con (头部):")
    print("    摇杆 = 抬头/低头/左转/右转")
    print("    D-pad 上 = 开始/跳过重置")
    print("    D-pad 下 = 结束当前 episode")
    print("    D-pad 左 = 重新录制")
    print("    D-pad 右 = 完全退出")
    print("  Right Joy-Con (底盘):")
    print("    摇杆 = 前进/后退/左移/右移")
    print("    Y/A  = 左转/右转")
    print("    X/B  = 速度加档/减档")
    print("  Keyboard backup: 1=start, 2=end, 3=rerecord, 4=quit\n")

    # -----------------------------------------------------------------------
    # build_action 回调：每帧构造完整动作
    # -----------------------------------------------------------------------

    def build_action(obs: dict) -> dict:
        leader_action = leader.get_action()
        base_action, head_action = joycon.update()

        # 更新 episode 控制状态（Joy-Con D-pad）
        joycon.update_episode_controls()
        joycon_ev = joycon.consume_episode_events()
        sync_episode_events(joycon_ev, events)

        # 后备键盘控制
        kb_ev = kb_listener.consume_events()
        sync_episode_events(kb_ev, events)

        # arms_only 模式下底盘不动，但允许控制头部
        if args.mode == "arms_only":
            base_action = {}

        return merge_actions(
            leader_action=leader_action,
            head_action=head_action,
            base_action=base_action,
            observation=obs,
            action_features=robot.action_features,
        )

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
        )
    finally:
        print("[INFO] Disconnecting...")
        kb_listener.stop()
        joycon.disconnect()
        if leader.is_connected:
            leader.disconnect()
        if robot.is_connected:
            robot.disconnect()
        print("[INFO] Done")


if __name__ == "__main__":
    main()
