#!/usr/bin/env python
"""
XLerobot 双主臂远程遥操脚本 (Bimanual Leader + Single Joy-Con + ZMQ)

========================================================================
单 Joy-Con 控制底盘与头部（可选左右，默认右 Joy-Con）
========================================================================

控制说明:
    主臂遥操: 直接移动两个 SO-101 主臂，从臂会跟随

    Joy-Con 按键 (默认右 Joy-Con):
      底盘模式 (默认):
        摇杆 = 前后左右平移 (上=前 下=后 左=左移 右=右移)
        Y = 左转,  A = 右转
        X = 底盘速度加档,  B = 底盘速度减档
        ZR = 切换至头部控制模式

      头部模式 (按 ZR 切换):
        摇杆 = 头部控制 (上=抬头 下=低头 左=左转 右=右转)
        X = 抬头,  B = 低头
        Y = 左转,  A = 右转
        ZR = 切换至底盘控制模式

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

2. PC 端运行本脚本：
    PYTHONPATH=src python examples/xlerobot/teleop_bimanual_joycon_single_zmq.py \
        --remote_ip=10.42.0.192 \
        --left_arm_port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46084903-if00 \
        --right_arm_port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_58FA093104-if00 \
        --camera_names=left,right,head

3. 使用左 Joy-Con：
    PYTHONPATH=src python examples/xlerobot/teleop_bimanual_joycon_single_zmq.py \
        --remote_ip=10.42.0.192 \
        --left_arm_port=/dev/serial/by-id/usb-... \
        --right_arm_port=/dev/serial/by-id/usb-... \
        --joycon_side=left \
        --camera_names=left,right,head

4. 仅查看可用串口：
    PYTHONPATH=src python examples/xlerobot/teleop_bimanual_joycon_single_zmq.py --list_ports

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
import time

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.xlerobot.config_xlerobot import XLerobotClientConfig
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient
from lerobot.teleoperators.xlebi_so101_leader import XleBiSO101Leader, XleBiSO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30

# 底盘速度档位
BASE_SPEED_LEVELS = [
    {"xy": 0.05, "theta": 15},
    {"xy": 0.10, "theta": 30},
    {"xy": 0.15, "theta": 45},
    {"xy": 0.20, "theta": 60},
    {"xy": 0.30, "theta": 90},
]

# 摇杆阈值（Joy-Con 摇杆原始值范围 0-4095，中心约 2048）
STICK_UP_THRESHOLD = 3000     # > 此值认为摇杆向上推
STICK_DOWN_THRESHOLD = 1000   # < 此值认为摇杆向下推
STICK_LEFT_THRESHOLD = 1000   # < 此值认为摇杆向左推
STICK_RIGHT_THRESHOLD = 3000  # > 此值认为摇杆向右推


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


class SingleJoyconController:
    """单 Joy-Con 控制器 — 负责头部和底盘控制。

    支持左右 Joy-Con，默认右 Joy-Con。
    ZR 键切换底盘/头部控制模式。
    """

    def __init__(self, side: str = "right"):
        try:
            from joyconrobotics import JoyconRobotics
        except ImportError:
            print("错误: 未安装 joyconrobotics 库")
            print("请执行: git clone https://github.com/box2ai-robotics/joycon-robotics.git && cd joycon-robotics && pip install -e . && sudo make install")
            raise

        self.side = side.lower()
        if self.side not in ("left", "right"):
            raise ValueError(f"joycon_side 必须是 left 或 right， got '{self.side}'")

        self.joycon = JoyconRobotics(
            device=self.side,
            dof_speed=[2, 2, 2, 1, 1, 1],
            without_rest_init=True,
        )

        # 控制模式: "base" = 底盘, "head" = 头部
        self.control_mode = "base"

        # 头部目标位置
        self.head_motor_1 = 0.0  # yaw (左右转)
        self.head_motor_2 = 0.0  # pitch (抬低头)
        self.head_step_deg = 2.0

        # 底盘速度档位
        self.speed_level = 1  # 默认中档

        # 按键去抖状态（记录上一帧按键，用于上升沿检测）
        self.prev_zr = 0

    def _read_buttons(self) -> dict:
        """读取当前 Joy-Con 的按钮状态。"""
        j = self.joycon.joycon
        is_r = j.is_right()

        if is_r:
            return {
                "X": j.get_button_x(),
                "B": j.get_button_b(),
                "Y": j.get_button_y(),
                "A": j.get_button_a(),
                "Home": j.get_button_home(),
                "R": j.get_button_r(),
                "ZR": j.get_button_zr(),
                "StickH": j.get_stick_right_horizontal(),
                "StickV": j.get_stick_right_vertical(),
            }
        else:
            # 左 Joy-Con 只有方向键，映射如下：
            #   上=前进(X)  下=后退(B)  左=左移(Y)  右=右移(A)
            return {
                "X": j.get_button_up(),
                "B": j.get_button_down(),
                "Y": j.get_button_left(),
                "A": j.get_button_right(),
                "Home": j.get_button_minus(),
                "R": j.get_button_l(),
                "ZR": j.get_button_zl(),
                "StickH": j.get_stick_left_horizontal(),
                "StickV": j.get_stick_left_vertical(),
            }

    def _get_base_action(self, buttons: dict) -> dict:
        """根据按钮状态生成底盘动作。

        底盘模式映射：
          摇杆上/下 = 前进/后退，摇杆左/右 = 左移/右移
          Y/A = 旋转左/右
        """
        speed = BASE_SPEED_LEVELS[self.speed_level]

        x_cmd = y_cmd = theta_cmd = 0.0

        # 摇杆控制平移 (Joy-Con 摇杆原始值 0-4095)
        stick_v = buttons["StickV"]
        stick_h = buttons["StickH"]

        if stick_v > STICK_UP_THRESHOLD:
            x_cmd += speed["xy"]      # 上推 = 前进
        elif stick_v < STICK_DOWN_THRESHOLD:
            x_cmd -= speed["xy"]      # 下推 = 后退

        if stick_h < STICK_LEFT_THRESHOLD:
            y_cmd += speed["xy"]      # 左推 = 左移
        elif stick_h > STICK_RIGHT_THRESHOLD:
            y_cmd -= speed["xy"]      # 右推 = 右移

        # Y/A 按钮控制旋转
        if buttons["Y"]:
            theta_cmd += speed["theta"]   # 左转
        if buttons["A"]:
            theta_cmd -= speed["theta"]   # 右转

        return {"x.vel": x_cmd, "y.vel": y_cmd, "theta.vel": theta_cmd}

    def _update_head_from_buttons(self, buttons: dict) -> None:
        """根据按钮状态更新头部目标位置（位置控制）。

        注意：实际电机方向与代码命名假设相反
              motor_2 (pitch): '-='=抬头, '+='=低头
              motor_1 (yaw):   '+='=左转, '-='=右转
        """
        if buttons["X"]:
            self.head_motor_2 -= self.head_step_deg   # 实际抬头
        if buttons["B"]:
            self.head_motor_2 += self.head_step_deg   # 实际低头
        if buttons["Y"]:
            self.head_motor_1 += self.head_step_deg   # 实际左转
        if buttons["A"]:
            self.head_motor_1 -= self.head_step_deg   # 实际右转

    def _get_head_action(self, buttons: dict) -> dict:
        """根据按钮状态生成头部动作（含摇杆 + 按钮双重控制）。"""
        # 摇杆控制头部
        stick_v = buttons["StickV"]
        stick_h = buttons["StickH"]

        # 注意：实际电机方向与代码命名假设相反
        # motor_2 (pitch): '-='=抬头, '+='=低头
        if stick_v > STICK_UP_THRESHOLD:
            self.head_motor_2 -= self.head_step_deg   # 摇杆上推 = 实际抬头
        elif stick_v < STICK_DOWN_THRESHOLD:
            self.head_motor_2 += self.head_step_deg   # 摇杆下推 = 实际低头

        # motor_1 (yaw):  '+='=左转, '-='=右转
        if stick_h < STICK_LEFT_THRESHOLD:
            self.head_motor_1 += self.head_step_deg   # 摇杆左推 = 实际左转
        elif stick_h > STICK_RIGHT_THRESHOLD:
            self.head_motor_1 -= self.head_step_deg   # 摇杆右推 = 实际右转

        # 按钮也控制头部（作为摇杆的备选）
        self._update_head_from_buttons(buttons)

        return {
            "head_motor_1.pos": self.head_motor_1,
            "head_motor_2.pos": self.head_motor_2,
        }

    def update(self) -> tuple[dict, dict, bool]:
        """更新 Joy-Con 状态，返回 (base_action, head_action, should_exit)。"""
        buttons = self._read_buttons()

        # ZR 键切换模式 (上升沿检测)
        zr_pressed = buttons["ZR"]
        if zr_pressed and not self.prev_zr:
            if self.control_mode == "base":
                self.control_mode = "head"
                print(f"[JOYCON] 切换至头部控制模式 ({self.side})")
            else:
                self.control_mode = "base"
                print(f"[JOYCON] 切换至底盘控制模式 ({self.side})")
        self.prev_zr = zr_pressed

        base_action = {}
        head_action = {}

        if self.control_mode == "base":
            base_action = self._get_base_action(buttons)

            # 底盘模式下 X/B 控制速度档位 (上升沿触发)
            if buttons["X"] and not getattr(self, "_prev_speed_x", 0):
                self.speed_level = min(self.speed_level + 1, len(BASE_SPEED_LEVELS) - 1)
                print(f"[JOYCON] 底盘速度加档: level {self.speed_level + 1}/{len(BASE_SPEED_LEVELS)}")
            if buttons["B"] and not getattr(self, "_prev_speed_b", 0):
                self.speed_level = max(self.speed_level - 1, 0)
                print(f"[JOYCON] 底盘速度减档: level {self.speed_level + 1}/{len(BASE_SPEED_LEVELS)}")
            self._prev_speed_x = buttons["X"]
            self._prev_speed_b = buttons["B"]

            # 头部保持当前位置
            head_action = {
                "head_motor_1.pos": self.head_motor_1,
                "head_motor_2.pos": self.head_motor_2,
            }
        else:
            head_action = self._get_head_action(buttons)

        return base_action, head_action, False

    def disconnect(self):
        if hasattr(self.joycon, "disconnect"):
            self.joycon.disconnect()


def main():
    parser = argparse.ArgumentParser(description="XLerobot bimanual teleoperation via ZMQ with single Joy-Con")
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
    parser.add_argument(
        "--joycon_side",
        type=str,
        default="right",
        choices=["left", "right"],
        help="使用哪个 Joy-Con 控制底盘和头部: left(左), right(右，默认)",
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

    # 初始化单 Joy-Con 控制器
    print(f"[INFO] 初始化 {args.joycon_side} Joy-Con 控制器...")
    joycon = SingleJoyconController(side=args.joycon_side)

    # 连接所有设备
    print("[INFO] Connecting to remote robot...")
    robot.connect()
    print("[INFO] Connecting to leader arms...")
    leader.connect()

    if not robot.is_connected or not leader.is_connected:
        raise RuntimeError("Failed to connect one or more devices!")

    # 启动 rerun 可视化界面（图像、状态曲线等会自动显示）
    init_rerun(session_name="xlerobot_teleop_joycon_single")

    side_label = "右" if args.joycon_side == "right" else "左"
    print(f"\n[INFO] All devices connected. Starting teleop loop...")
    print(f"  Arms:      Move the leader arms directly")
    print(f"  Joy-Con:   {side_label} Joy-Con ({args.joycon_side})")
    print(f"  Base mode: 摇杆=平移(前/后/左/右)  Y/A=旋转  X/B=速度+/-")
    print(f"  Head mode: 摇杆=头部(抬/低/左/右)  X=抬头 B=低头 Y=左转 A=右转")
    print(f"  Switch:    ZR = 切换底盘/头部模式")
    print(f"  Exit:      Ctrl+C\n")

    # 设置头部步长
    joycon.head_step_deg = args.head_step_deg

    try:
        while True:
            t0 = time.perf_counter()

            # 1. 获取主臂动作
            leader_action = leader.get_action()

            # 2. 获取 Joy-Con 控制数据
            base_action, head_action, should_exit = joycon.update()

            if should_exit:
                break

            # 3. 合并动作：手臂来自主臂，底盘和头部来自 Joy-Con
            action = {**leader_action, **base_action, **head_action}

            # 4. 通过 ZMQ 发送到 Orin 端的机器人
            robot.send_action(action)

            # 5. 接收观测（包含图像，由 rerun 自动显示）
            obs = robot.get_observation()
            log_rerun_data(observation=obs, action=action)

            # 6. 维持目标频率
            dt = time.perf_counter() - t0
            precise_sleep(max(1.0 / args.fps - dt, 0.0))

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        print("[INFO] Disconnecting...")
        joycon.disconnect()
        if leader.is_connected:
            leader.disconnect()
        if robot.is_connected:
            robot.disconnect()
        print("[INFO] Done")


if __name__ == "__main__":
    main()
