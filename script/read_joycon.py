#!/usr/bin/env python
"""
Joy-Con 手柄输入读取工具

连接左/右/左右 Joy-Con，实时打印摇杆、按钮和姿态数据。

用法:
    python read_joycon.py --side left      # 只读左 Joy-Con
    python read_joycon.py --side right     # 只读右 Joy-Con
    python read_joycon.py --side both      # 同时读左右（默认）
    python read_joycon.py --side both --rate 60   # 60Hz 刷新
"""

import argparse
import sys
import time


def _fmt_bool(val: int) -> str:
    """把 0/1 格式化成 ●/○"""
    return "●" if val == 1 else "○"


class JoyconReader:
    """封装 Joy-Con 读取，支持左/右单个或同时读取"""

    def __init__(self, side: str, rate: int = 30):
        self.side = side.lower()
        self.rate = rate
        self.interval = 1.0 / rate
        self.controllers = {}  # key: "left" | "right", value: JoyconRobotics
        self._connected = False

    def connect(self) -> bool:
        try:
            from joyconrobotics import JoyconRobotics
        except ImportError:
            print("错误: 未安装 joyconrobotics 库")
            print("请执行: git clone https://github.com/box2ai-robotics/joycon-robotics.git && cd joycon-robotics && pip install -e . && sudo make install")
            return False

        sides = []
        if self.side in ("left", "both"):
            sides.append("left")
        if self.side in ("right", "both"):
            sides.append("right")

        if not sides:
            print(f"错误: 不合法的 side 参数 '{self.side}'，请使用 left/right/both")
            return False

        for s in sides:
            try:
                ctrl = JoyconRobotics(
                    device=s,
                    dof_speed=[2, 2, 2, 1, 1, 1],
                    without_rest_init=True,  # 不执行复位，避免初始化抖动
                )
                self.controllers[s] = ctrl
                print(f"✅ {s} Joy-Con 已连接")
            except Exception as e:
                print(f"❌ {s} Joy-Con 连接失败: {e}")
                self.disconnect()
                return False

        self._connected = bool(self.controllers)
        return self._connected

    def disconnect(self):
        for s, ctrl in list(self.controllers.items()):
            try:
                ctrl.disconnect()
                print(f"🔌 {s} Joy-Con 已断开")
            except Exception:
                pass
        self.controllers.clear()
        self._connected = False

    def _read_one(self, side: str, ctrl) -> dict:
        """读取单个 Joy-Con 的所有输入状态"""
        j = ctrl.joycon
        is_r = j.is_right()

        # 摇杆原始值
        if is_r:
            stick_v = j.get_stick_right_vertical()
            stick_h = j.get_stick_right_horizontal()
        else:
            stick_v = j.get_stick_left_vertical()
            stick_h = j.get_stick_left_horizontal()

        # 按钮状态
        if is_r:
            buttons = {
                "X": j.get_button_x(),
                "B": j.get_button_b(),
                "Y": j.get_button_y(),
                "A": j.get_button_a(),
                "+": j.get_button_plus(),
                "Home": j.get_button_home(),
                "ZR": j.get_button_zr(),
                "R": j.get_button_r(),
                "StickR": j.get_button_r_stick(),
            }
        else:
            buttons = {
                "↑": j.get_button_up(),
                "↓": j.get_button_down(),
                "←": j.get_button_left(),
                "→": j.get_button_right(),
                "-": j.get_button_minus(),
                "Capture": j.get_button_capture(),
                "ZL": j.get_button_zl(),
                "L": j.get_button_l(),
                "StickL": j.get_button_l_stick(),
            }

        # 姿态（通过 get_control 获取）
        pose, gripper, btn_ctrl = ctrl.get_control()
        x, y, z, roll, pitch, yaw = pose

        return {
            "side": side,
            "stick_v": stick_v,
            "stick_h": stick_h,
            "pose": (x, y, z, roll, pitch, yaw),
            "gripper": gripper,
            "btn_ctrl": btn_ctrl,
            "buttons": buttons,
        }

    def _format_line(self, data: dict) -> str:
        """把读取结果格式化成一行字符串"""
        side = data["side"]
        sv, sh = data["stick_v"], data["stick_h"]
        x, y, z, r, p, yaw = data["pose"]
        btns = data["buttons"]

        # 找出当前按下的按钮
        pressed = [name for name, val in btns.items() if val == 1]
        pressed_str = " ".join(pressed) if pressed else "-"

        return (
            f"[{side:5s}] "
            f" Stick(v={sv:4d},h={sh:4d}) |"
            f" Pose(x={x:+.3f},y={y:+.3f},z={z:+.3f},"
            f"r={r:+.2f},p={p:+.2f},y={yaw:+.2f}) |"
            f" Gripper={data['gripper']:.1f} |"
            f" Buttons=[{pressed_str}]"
        )

    def run(self):
        """主循环：持续读取并打印"""
        if not self._connected:
            print("未连接任何 Joy-Con，退出")
            return

        print(f"\n🎮 开始读取 (目标 {self.rate}Hz)，按 Ctrl+C 退出\n")

        try:
            while True:
                t0 = time.perf_counter()

                lines = []
                for side, ctrl in self.controllers.items():
                    data = self._read_one(side, ctrl)
                    lines.append(self._format_line(data))

                # 清除行并打印（both 模式时两行）
                output = "\n".join(lines)
                n_lines = output.count("\n") + 1
                if n_lines == 1:
                    # 单行模式：回车 + 清到行尾，避免旧字符残留
                    print(f"\r\033[K{output}", end="", flush=True)
                else:
                    # 多行模式：先清当前行，打印（最后一行自带清尾），再把光标移回第一行开头
                    print(f"\r\033[K{output}\033[K", end="", flush=True)
                    print(f"\033[{n_lines - 1}A\r", end="", flush=True)

                # 精确睡眠
                elapsed = time.perf_counter() - t0
                sleep_time = self.interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\n⏹ 已停止")
        finally:
            self.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Joy-Con 手柄输入读取工具")
    parser.add_argument(
        "--side",
        type=str,
        default="both",
        choices=["left", "right", "both"],
        help="连接哪个手柄: left(左), right(右), both(左右同时，默认)",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=30,
        help="刷新频率 Hz (默认 30)",
    )
    args = parser.parse_args()

    reader = JoyconReader(side=args.side, rate=args.rate)

    if not reader.connect():
        sys.exit(1)

    reader.run()


if __name__ == "__main__":
    main()
