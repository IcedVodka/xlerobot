#!/usr/bin/env python
"""
XLerobot 双主臂 + Joy-Con 单控遥操脚本（ZMQ 远程 + 可选数据录制）

控制说明:
    主臂遥操: 直接移动两个 SO-101 主臂，从臂会跟随
    右 Joy-Con:
        方向键 = 头部控制 (上/下=pitch, 左/右=yaw)
        X = 前进, B = 后退
        Y = 左转, A = 右转
        +/- = 底盘速度加/减
        Home = 退出程序

录制控制:
    右 Joy-Con + = 保存并开始下一 episode
    右 Joy-Con - = 重录当前 episode
    右 Joy-Con Home (长按2秒) = 停止录制

启动命令:
    1. Orin 端先启动 Host:
        PYTHONPATH=src python -m lerobot.robots.xlerobot.xlerobot_host --robot.id=my_xlerobot

    2. PC 端运行本脚本:
        PYTHONPATH=src python teleop/scripts/teleop_bimanual_joycon_single_zmq.py \
            --remote_ip=10.42.0.192 \
            --left_arm_port=/dev/serial/by-id/usb-... \
            --right_arm_port=/dev/serial/by-id/usb-... \
            --camera_names=left,right,head

    3. 录制模式:
        PYTHONPATH=src python teleop/scripts/teleop_bimanual_joycon_single_zmq.py \
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.xlerobot.config_xlerobot import XLerobotClientConfig
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient
from lerobot.teleoperators.xlebi_so101_leader import XleBiSO101Leader, XleBiSO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from teleop.src.teleop_record import TeleopRecordManager

try:
    from joyconrobotics import JoyconRobotics
except ImportError:
    JoyconRobotics = None
    logging.warning("joyconrobotics not installed. Joy-Con control will not work.")

logger = logging.getLogger(__name__)
FPS = 30

# 底盘速度控制参数
BASE_ACCELERATION_RATE = 2.0
BASE_DECELERATION_RATE = 2.5
BASE_MAX_SPEED = 3.0


def find_stable_serial_ports() -> list[str]:
    return sorted(glob.glob("/dev/serial/by-id/*"))


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
        return stable_ports[0]
    elif tty_ports:
        print(f"[WARN] 回退到动态路径: {tty_ports[0]}")
        return tty_ports[0]
    else:
        raise RuntimeError("未检测到任何串口设备，请检查 USB 连接。")


class SingleJoyconController:
    """单右 Joy-Con 控制器 — 同时负责头部和底盘控制。"""

    def __init__(self):
        if JoyconRobotics is None:
            raise ImportError("joyconrobotics not installed. Install with: pip install joyconrobotics")
        self.joycon = JoyconRobotics("right", dof_speed=[2, 2, 2, 1, 1, 1])
        self.head_step_deg = 2.0

        # 头部目标位置
        self.head_motor_1 = 0.0
        self.head_motor_2 = 0.0

        # 底盘速度控制状态
        self.current_base_speed = 0.0
        self.last_update_time = time.time()
        self.is_accelerating = False

        # 录制按钮状态
        self.prev_plus = 0
        self.prev_minus = 0
        self.prev_home = 0
        self.home_press_time = 0.0

    def update(self) -> dict:
        """更新 Joy-Con 状态，返回控制数据。"""
        result = {
            "head_action": {},
            "base_action": {},
            "record_start": False,
            "record_rerecord": False,
            "record_stop": False,
            "exit": False,
        }

        # 获取 Joy-Con 数据（pose 和 gripper 用于主臂夹爪控制，但主臂由物理移动控制）
        _pose, _gripper, _control = self.joycon.get_control()

        j = self.joycon.joycon

        # ===== 头部控制 (方向键) =====
        if j.get_button_up():
            self.head_motor_2 += self.head_step_deg
        if j.get_button_down():
            self.head_motor_2 -= self.head_step_deg
        if j.get_button_left():
            self.head_motor_1 += self.head_step_deg
        if j.get_button_right():
            self.head_motor_1 -= self.head_step_deg

        result["head_action"] = {
            "head_motor_1.pos": self.head_motor_1,
            "head_motor_2.pos": self.head_motor_2,
        }

        # ===== 底盘控制 (X/B/Y/A) =====
        button_x = j.get_button_x()  # forward
        button_b = j.get_button_b()  # backward
        button_y = j.get_button_y()  # rotate left
        button_a = j.get_button_a()  # rotate right

        pressed_keys = set()
        if button_x:
            pressed_keys.add("i")
        if button_b:
            pressed_keys.add("k")
        if button_y:
            pressed_keys.add("u")
        if button_a:
            pressed_keys.add("o")

        keyboard_keys = np.array(list(pressed_keys))
        base_action = self._get_base_action_with_speed(keyboard_keys)
        result["base_action"] = base_action

        # ===== 录制控制 =====
        plus = j.get_button_plus()
        minus = j.get_button_minus()
        home = j.get_button_home()

        if plus and not self.prev_plus:
            result["record_start"] = True
        if minus and not self.prev_minus:
            result["record_rerecord"] = True

        # Home 键: 短按=退出，长按2秒=停止录制
        if home and not self.prev_home:
            self.home_press_time = time.time()
        elif not home and self.prev_home:
            press_duration = time.time() - self.home_press_time
            if press_duration >= 2.0:
                result["record_stop"] = True
            else:
                result["exit"] = True

        self.prev_plus = plus
        self.prev_minus = minus
        self.prev_home = home

        return result

    def _get_base_action_with_speed(self, keyboard_keys: np.ndarray) -> dict:
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        any_base_button = any(k in keyboard_keys for k in ["i", "k", "j", "l", "u", "o"])

        if any_base_button:
            if not self.is_accelerating:
                self.is_accelerating = True
            self.current_base_speed += BASE_ACCELERATION_RATE * dt
            self.current_base_speed = min(self.current_base_speed, BASE_MAX_SPEED)
        else:
            if self.is_accelerating:
                self.is_accelerating = False
            self.current_base_speed -= BASE_DECELERATION_RATE * dt
            self.current_base_speed = max(self.current_base_speed, 0.0)

        base_action = self._from_keyboard_to_base_action(keyboard_keys)

        if self.current_base_speed > 0.01:
            for key in base_action:
                if "vel" in key:
                    base_action[key] *= self.current_base_speed

        return base_action

    def _from_keyboard_to_base_action(self, pressed_keys: np.ndarray) -> dict:
        speed_levels = [
            {"xy": 0.1, "theta": 30},
            {"xy": 0.2, "theta": 60},
            {"xy": 0.3, "theta": 90},
        ]
        speed = speed_levels[0]

        x_cmd = y_cmd = theta_cmd = 0.0
        if "i" in pressed_keys:
            x_cmd += speed["xy"]
        if "k" in pressed_keys:
            x_cmd -= speed["xy"]
        if "j" in pressed_keys:
            y_cmd += speed["xy"]
        if "l" in pressed_keys:
            y_cmd -= speed["xy"]
        if "u" in pressed_keys:
            theta_cmd += speed["theta"]
        if "o" in pressed_keys:
            theta_cmd -= speed["theta"]

        return {"x.vel": x_cmd, "y.vel": y_cmd, "theta.vel": theta_cmd}

    def disconnect(self):
        if hasattr(self.joycon, "disconnect"):
            self.joycon.disconnect()


def parse_args():
    parser = argparse.ArgumentParser(description="XLerobot bimanual teleoperation via ZMQ with single Joy-Con control")
    parser.add_argument("--remote_ip", type=str, required=True, help="Orin IP address")
    parser.add_argument("--port_zmq_cmd", type=int, default=5555)
    parser.add_argument("--port_zmq_obs", type=int, default=5556)
    parser.add_argument("--left_arm_port", type=str, default=None)
    parser.add_argument("--right_arm_port", type=str, default=None)
    parser.add_argument("--list_ports", action="store_true")
    parser.add_argument("--camera_names", type=str, default="")
    parser.add_argument("--camera_width", type=int, default=640)
    parser.add_argument("--camera_height", type=int, default=480)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--dataset_repo_id_fixed", type=str, default=None)
    parser.add_argument("--dataset_repo_id_mobile", type=str, default=None)
    parser.add_argument("--single_task", type=str, default="xlerobot teleop task")
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--display_data", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if JoyconRobotics is None:
        print("[ERROR] joyconrobotics not installed. Install with: pip install joyconrobotics")
        return

    if args.list_ports:
        print("=" * 50)
        print("可用稳定串口路径:")
        for p in find_stable_serial_ports():
            print(f"  {p}")
        return

    left_port = resolve_arm_port(args.left_arm_port, "left_arm_port")
    right_port = resolve_arm_port(args.right_arm_port, "right_arm_port")
    if left_port == right_port:
        raise ValueError(f"左右臂使用了相同的串口: {left_port}")
    print(f"\n[INFO] 左臂串口: {left_port}")
    print(f"[INFO] 右臂串口: {right_port}")

    camera_configs = {}
    for cam_name in args.camera_names.split(","):
        cam_name = cam_name.strip()
        if cam_name:
            camera_configs[cam_name] = OpenCVCameraConfig(
                index_or_path="", fps=args.fps, width=args.camera_width, height=args.camera_height,
            )

    robot_config = XLerobotClientConfig(
        remote_ip=args.remote_ip,
        port_zmq_cmd=args.port_zmq_cmd,
        port_zmq_observations=args.port_zmq_obs,
        cameras=camera_configs,
    )
    robot = XLerobotClient(robot_config)

    leader_config = XleBiSO101LeaderConfig(
        id="bimanual_leader", left_arm_port=left_port, right_arm_port=right_port,
    )
    leader = XleBiSO101Leader(leader_config)

    print("[INFO] Initializing single Joy-Con controller (right only)...")
    joycon = SingleJoyconController()

    print("[INFO] Connecting to remote robot...")
    robot.connect()
    print("[INFO] Connecting to leader arms...")
    leader.connect()

    recorder = None
    if args.record:
        if not args.dataset_repo_id_fixed and not args.dataset_repo_id_mobile:
            raise ValueError("--record 需要至少指定一个 dataset_repo_id")
        recorder = TeleopRecordManager(
            repo_id_fixed=args.dataset_repo_id_fixed,
            repo_id_mobile=args.dataset_repo_id_mobile,
            robot=robot,
            fps=args.fps,
            single_task=args.single_task,
        )
        print(f"\n[INFO] Recording enabled")
        print(f"       Fixed: {args.dataset_repo_id_fixed or 'N/A'}")
        print(f"       Mobile: {args.dataset_repo_id_mobile or 'N/A'}")

    if args.display_data:
        init_rerun(session_name="xlerobot_teleop_joycon_single")

    print("\n[INFO] All devices connected. Starting teleop loop...")
    print("  Arms:   Move leader arms directly")
    print("  Head:   Right Joy-Con D-pad (up/down/left/right)")
    print("  Base:   Right Joy-Con X/B/Y/A (forward/back/rotL/rotR)")
    print("  Speed:  Auto acceleration/deceleration")
    if args.record:
        print("\n  Recording:")
        print("    + = Next episode")
        print("    - = Re-record")
        print("    Home (hold 2s) = Stop")
    print("  Exit:   Home (short press)\n")

    try:
        while True:
            t0 = time.perf_counter()

            leader_action = leader.get_action()
            joycon_data = joycon.update()

            # 录制控制
            if recorder is not None:
                if joycon_data.get("record_start") and not recorder.active_recorder.is_recording:
                    recorder.start_episode()
                elif joycon_data.get("record_rerecord") and recorder.active_recorder.is_recording:
                    recorder.rerecord_episode()
                elif joycon_data.get("record_stop") and recorder.active_recorder.is_recording:
                    recorder.save_episode()

            # 退出
            if joycon_data.get("exit"):
                print("\n[INFO] Home pressed, exiting...")
                break

            # 合并动作
            action = {**leader_action}
            action.update(joycon_data.get("base_action", {}))
            action.update(joycon_data.get("head_action", {}))

            robot.send_action(action)
            obs = robot.get_observation()

            if recorder is not None:
                recorder.record_frame(obs, action)

            if args.display_data:
                log_rerun_data(observation=obs, action=action)

            dt = time.perf_counter() - t0
            precise_sleep(max(1.0 / args.fps - dt, 0.0))

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        print("\n[INFO] Disconnecting...")
        if recorder is not None:
            recorder.finalize()
        joycon.disconnect()
        if leader.is_connected:
            leader.disconnect()
        if robot.is_connected:
            robot.disconnect()
        print("[INFO] Done")


if __name__ == "__main__":
    main()
