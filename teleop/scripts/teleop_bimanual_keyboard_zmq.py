#!/usr/bin/env python

"""
XLerobot 双主臂远程遥操脚本 (Bimanual Leader + Keyboard + ZMQ) + 可选录制

========================================================================
键盘控制底盘与头部
========================================================================

控制说明:
    主臂遥操: 直接移动两个 SO-101 主臂，从臂会跟随

    头部控制（方向键）:
        ↑ = 抬头,  ↓ = 低头
        ← = 左转,  → = 右转

    底盘控制:
        I = 前进,  K = 后退
        J = 左移,  L = 右移
        U = 左转,  O = 右转
        N = 速度加档,  M = 速度减档

录制控制（数字键）：
    1 = 开始 / 跳过重置 并进入下一轮
    2 = 结束当前 episode
    3 = 重新录制当前 episode
    4 = 完全退出录制流程

========================================================================
两种模式
========================================================================

--mode arms_only :
    控制双臂+头部，底盘不动（置零），数据集中记录双臂+头部数据。
    适合训练仅控制双臂的策略（头部作为场景变化来源）。

--mode full_body (默认):
    全身控制（双臂+头部+底盘），数据集中记录全身数据。

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
完整启动命令
========================================================================

1. Orin 端先启动 Host:
    PYTHONPATH=src python -m lerobot.robots.xlerobot.xlerobot_host --robot.id=my_xlerobot

2. PC 端运行本脚本（纯遥操，不录制）：
    PYTHONPATH=src python teleop/scripts/teleop_bimanual_keyboard_zmq.py \
        --remote_ip=10.42.0.192 \
        --left_arm_port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46084903-if00 \
        --right_arm_port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_58FA093104-if00 \
        --camera_names=left,right,head

3. PC 端运行本脚本（录制数据 + 可视化）：
    PYTHONPATH=src python teleop/scripts/teleop_bimanual_keyboard_zmq.py \
        --remote_ip=10.42.0.192 \
        --left_arm_port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46084903-if00 \
        --right_arm_port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_58FA093104-if00 \
        --camera_names=left,right,head \
        --repo_id=my_keyboard_dataset \
        --mode=full_body \
        --display_data

4. 仅查看可用串口：
    PYTHONPATH=src python teleop/scripts/teleop_bimanual_keyboard_zmq.py --list_ports

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
"""

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
    filter_arm_head_features,
    merge_actions,
    run_recording_session,
    sync_episode_events,
)

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.robots.xlerobot.config_xlerobot import XLerobotClientConfig
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
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


# ---------------------------------------------------------------------------
# 串口工具
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 键盘头部控制器（方向键）
# ---------------------------------------------------------------------------


class KeyboardHeadController:
    """使用 pynput 监听方向键，控制头部电机位置。

    注意：与 KeyboardTeleop 是独立的监听器，不共享按键状态。
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
# 主函数
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="XLerobot bimanual teleoperation via ZMQ with keyboard")
    parser.add_argument("--remote_ip", type=str, default=None, help="Orin IP address")
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
        "--head_step_deg", type=float, default=2.0, help="Head motor step size in degrees per frame"
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

    # 录制相关参数
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="数据集标识名称，指定时进入录制模式，不指定时仅遥操",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full_body",
        choices=["arms_only", "full_body"],
        help="录制模式：arms_only 采集双臂+头部数据（底盘不动），full_body 采集全身数据",
    )
    parser.add_argument("--dataset_root", type=str, default=None, help="数据集本地存储根目录")
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

    # 仅列出串口时不需要 remote_ip/repo_id
    if not args.list_ports:
        if not args.remote_ip:
            parser.error("--remote_ip is required (unless using --list_ports)")

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

    # 初始化键盘控制器
    print("[INFO] 初始化键盘控制器（方向键=头部，IKJL=底盘）...")
    keyboard_teleop = KeyboardTeleop(KeyboardTeleopConfig(id="my_laptop_keyboard"))
    head_controller = KeyboardHeadController(head_step_deg=args.head_step_deg)

    # 连接所有设备
    print("[INFO] Connecting to remote robot...")
    robot.connect()
    print("[INFO] Connecting to leader arms...")
    leader.connect()
    print("[INFO] Connecting to keyboard...")
    keyboard_teleop.connect()
    head_controller.start()

    if not robot.is_connected or not leader.is_connected:
        raise RuntimeError("Failed to connect one or more devices!")

    # 条件启动 rerun 可视化
    if args.display_data:
        init_rerun(session_name="xlerobot_teleop_keyboard")

    # ========================================================================
    # 录制模式
    # ========================================================================
    if args.repo_id:
        print(f"[INFO] 录制模式: {args.mode}")

        # 处理管线
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

        # 数据集 features
        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, "observation")
        full_features = {**action_features, **obs_features}

        if args.mode == "arms_only":
            dataset_features = filter_arm_head_features(full_features)
            print("[INFO] arms_only 模式：数据集包含双臂 + 头部字段")
        else:
            dataset_features = full_features
            print("[INFO] full_body 模式：数据集包含全身字段")

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

        # 键盘 episode 控制器（1=开始 2=结束 3=重录 4=退出）
        kb_listener = EpisodeKeyboardListener()
        kb_listener.start()
        events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "discard_current_episode": False,
        }

        print("\n[INFO] All devices connected. Starting recording loop...")
        print("  Arms:       Move the leader arms directly")
        print("  Head (方向键):")
        print("    ↑/↓ = 抬头/低头,  ←/→ = 左转/右转")
        print("  Base (键盘):")
        print("    I/K = 前进/后退, J/L = 左移/右移")
        print("    U/O = 左转/右转, N/M = 速度加档/减档")
        print("  Recording controls:")
        print("    1 = 开始/跳过重置")
        print("    2 = 结束当前 episode")
        print("    3 = 重新录制")
        print("    4 = 完全退出")
        print("  Exit:       Ctrl+C\n")

        # -------------------------------------------------------------------
        # build_action 回调
        # -------------------------------------------------------------------
        def build_action(obs: dict) -> dict:
            leader_action = leader.get_action()
            head_action = head_controller.get_head_action()
            pressed_keys = np.array(list(keyboard_teleop.get_action().keys()))
            base_action = robot._from_keyboard_to_base_action(pressed_keys) or {}

            # 处理键盘 episode 控制事件
            kb_ev = kb_listener.consume_events()
            sync_episode_events(kb_ev, events)

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
            head_controller.stop()
            if keyboard_teleop.is_connected:
                keyboard_teleop.disconnect()
            if leader.is_connected:
                leader.disconnect()
            if robot.is_connected:
                robot.disconnect()
            print("[INFO] Done")
        return

    # ========================================================================
    # 纯遥操模式（不录制）
    # ========================================================================
    print("\n[INFO] All devices connected. Starting teleop loop...")
    print("  Arms:       Move the leader arms directly")
    print("  Head (方向键):")
    print("    ↑/↓ = 抬头/低头,  ←/→ = 左转/右转")
    print("  Base (键盘):")
    print("    I/K = 前进/后退, J/L = 左移/右移")
    print("    U/O = 左转/右转, N/M = 速度加档/减档")
    print("  Exit:       Ctrl+C\n")

    try:
        while True:
            t0 = time.perf_counter()

            # 1. 获取主臂动作
            leader_action = leader.get_action()

            # 2. 获取头部动作（方向键）
            head_action = head_controller.get_head_action()

            # 3. 获取底盘动作（键盘 I/K/J/L/U/O/N/M）
            pressed_keys = np.array(list(keyboard_teleop.get_action().keys()))
            base_action = robot._from_keyboard_to_base_action(pressed_keys) or {}

            # 4. 合并动作：手臂来自主臂，底盘和头部来自键盘
            action = {**leader_action, **base_action, **head_action}

            # 5. 通过 ZMQ 发送到 Orin 端的机器人
            robot.send_action(action)

            # 6. 接收观测（包含图像，由 rerun 自动显示）
            obs = robot.get_observation()
            if args.display_data:
                log_rerun_data(observation=obs, action=action)

            # 7. 维持目标频率
            dt = time.perf_counter() - t0
            precise_sleep(max(1.0 / args.fps - dt, 0.0))

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        print("[INFO] Disconnecting...")
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
