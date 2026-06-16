#!/usr/bin/env python

"""
Joy-Con IMU (陀螺仪 + 加速度计) 校准工具

========================================================================
使用说明
========================================================================

1. 确保两个 Joy-Con 已通过蓝牙连接到 PC
2. 运行本脚本:
    PYTHONPATH=src python teleop/scripts/calibrate_joycon.py

3. 按提示选择操作:
    l  - 校准左 Joy-Con (陀螺仪 + 加速度计重力标定)
    r  - 校准右 Joy-Con (陀螺仪 + 加速度计重力标定)
    b  - 同时校准两个
    ml - 监测左 Joy-Con 实时读数
    mr - 监测右 Joy-Con 实时读数
    q  - 退出

4. 校准时将 Joy-Con 平放在桌面上保持静止，等待完成

========================================================================
"""

import math
import time

from joyconrobotics import JoyconRobotics


class CalibratableJoycon(JoyconRobotics):
    """跳过自动校准的 JoyconRobotics，让用户手动控制校准时机。"""

    def __init__(self, device, **kwargs):
        super().__init__(device, without_rest_init=True, **kwargs)
        self._accel_scale_compensation = 1.0  # 加速度计全局比例补偿因子

    def _get_compensated_accel(self):
        """获取经过重力补偿后的加速度计读数。"""
        raw = self.gyro.accel_in_g[0] if hasattr(self.gyro, 'accel_in_g') else [0, 0, 0]
        return tuple(v * self._accel_scale_compensation for v in raw)

    def calibrate_gyro(self):
        """校准陀螺仪零偏 (2 秒累加窗口)。"""
        print("  → 陀螺仪校准中，请保持静止...")

        pre = self.gyro.gyro_in_rad[0] if hasattr(self.gyro, 'gyro_in_rad') else None
        if pre:
            print(f"     校准前: x={pre[0]:+.4f}  y={pre[1]:+.4f}  z={pre[2]:+.4f} rad/s")

        self.gyro.calibrate(seconds=2)
        for i in range(2, 0, -1):
            print(f"     剩余 {i} 秒...")
            time.sleep(1)

        self.gyro.reset_orientation()
        self.orientation_sensor.reset_yaw()
        self.position = self.offset_position_m.copy()
        self.yaw_diff = 0.0
        self.orientation_sensor.set_yaw_diff(0.0)

        time.sleep(0.3)
        post = self.gyro.gyro_in_rad[0] if hasattr(self.gyro, 'gyro_in_rad') else None
        if post and pre:
            print(f"     校准后: x={post[0]:+.4f}  y={post[1]:+.4f}  z={post[2]:+.4f} rad/s")
            print(f"     零偏变化: Δx={abs(post[0]-pre[0]):.4f} Δy={abs(post[1]-pre[1]):.4f} Δz={abs(post[2]-pre[2]):.4f}")
        print("  ✓ 陀螺仪校准完成")

    def calibrate_accel_scale(self, sample_seconds=2.0):
        """通过重力测量标定加速度计比例因子。

        原理: Joy-Con 静止时，合加速度应等于 1g (重力加速度)。
        如果读出的合加速度 ≠ 1g，说明比例因子有偏差，计算补偿系数。
        """
        print("  → 加速度计重力标定中，请将 Joy-Con 平放在桌面上保持静止...")

        # 采集多帧数据取平均
        samples = []
        start = time.time()
        while time.time() - start < sample_seconds:
            a = self.gyro.accel_in_g[0] if hasattr(self.gyro, 'accel_in_g') else [0, 0, 0]
            samples.append(a)
            time.sleep(0.05)

        # 计算平均值
        n = len(samples)
        avg_x = sum(s[0] for s in samples) / n
        avg_y = sum(s[1] for s in samples) / n
        avg_z = sum(s[2] for s in samples) / n

        # 合加速度
        magnitude = math.sqrt(avg_x**2 + avg_y**2 + avg_z**2)

        print(f"     平均读数: x={avg_x:+.4f}  y={avg_y:+.4f}  z={avg_z:+.4f} g")
        print(f"     合加速度: {magnitude:.4f} g (理论值: 1.0000 g)")

        # 计算补偿因子
        if magnitude > 0.1:
            self._accel_scale_compensation = 1.0 / magnitude
            error_pct = (magnitude - 1.0) * 100
            print(f"     偏差: {error_pct:+.1f}%")
            print(f"     补偿因子: {self._accel_scale_compensation:.4f} (读数将乘以该值)")

            if abs(error_pct) > 10:
                print(f"     ⚠ 警告: 偏差超过 10%，该 Joy-Con 的出厂加速度计校准数据可能不准确！")
            else:
                print(f"     ✓ 偏差在可接受范围内")
        else:
            print(f"     ✗ 读数异常，无法计算补偿因子")

        print("  ✓ 加速度计标定完成")
        return magnitude

    def calibrate_with_feedback(self):
        """执行完整的 IMU 校准流程。"""
        print(f"\n  [{'左' if self.joycon.is_left() else '右'} Joy-Con]")
        self.calibrate_gyro()
        self.calibrate_accel_scale(sample_seconds=2.0)

    def get_calibrated_accel(self):
        """获取经过校准和补偿后的加速度计读数。"""
        return self._get_compensated_accel()


def show_live_readings(joycon, label="", duration=3.0):
    """显示实时读数，帮助用户观察漂移情况。"""
    print(f"\n  [{label}] 实时读数监测 ({duration}s):")
    print("  " + "-" * 65)
    print(f"  {'时间':>6s}  {'Gx':>9s} {'Gy':>9s} {'Gz':>9s} {'Ax':>9s} {'Ay':>9s} {'Az':>9s} {'|A|':>7s}")
    print("  " + "-" * 65)

    t0 = time.time()
    while time.time() - t0 < duration:
        gyro = joycon.gyro.gyro_in_rad[0] if hasattr(joycon.gyro, 'gyro_in_rad') else [0, 0, 0]
        accel = joycon.get_calibrated_accel()
        mag = math.sqrt(accel[0]**2 + accel[1]**2 + accel[2]**2)
        elapsed = time.time() - t0
        print(f"  {elapsed:6.2f}s  {gyro[0]:+9.4f} {gyro[1]:+9.4f} {gyro[2]:+9.4f}  "
              f"{accel[0]:+9.4f} {accel[1]:+9.4f} {accel[2]:+9.4f} {mag:7.4f}")
        time.sleep(0.5)


def main():
    print("=" * 60)
    print("Joy-Con IMU 校准工具")
    print("=" * 60)
    print("\n说明:")
    print("  • 陀螺仪校准: 通过 2 秒静止累加计算零偏")
    print("  • 加速度计标定: 通过重力测量计算比例补偿因子")
    print("  • 官方 Joy-Con 的加速度计出厂校准可能存在偏差，")
    print("    本工具会自动检测并补偿")

    print("\n[INFO] 正在连接 Joy-Con...")
    try:
        joycon_left = CalibratableJoycon("left", dof_speed=[2, 2, 2, 1, 1, 1])
        print("  ✓ 左 Joy-Con 已连接")
    except Exception as e:
        print(f"  ✗ 左 Joy-Con 连接失败: {e}")
        joycon_left = None

    try:
        joycon_right = CalibratableJoycon("right", dof_speed=[2, 2, 2, 1, 1, 1])
        print("  ✓ 右 Joy-Con 已连接")
    except Exception as e:
        print(f"  ✗ 右 Joy-Con 连接失败: {e}")
        joycon_right = None

    if joycon_left is None and joycon_right is None:
        print("\n[ERROR] 两个 Joy-Con 都未连接，请检查蓝牙配对后重试。")
        return

    while True:
        print("\n" + "=" * 60)
        print("请选择操作:")
        print("  l  - 校准左 Joy-Con")
        print("  r  - 校准右 Joy-Con")
        print("  b  - 同时校准两个 Joy-Con")
        print("  ml - 监测左 Joy-Con 实时读数")
        print("  mr - 监测右 Joy-Con 实时读数")
        print("  q  - 退出")
        print("=" * 60)

        choice = input("> ").strip().lower()

        if choice == "q":
            break

        elif choice == "l":
            if joycon_left is None:
                print("[WARN] 左 Joy-Con 未连接")
                continue
            joycon_left.calibrate_with_feedback()

        elif choice == "r":
            if joycon_right is None:
                print("[WARN] 右 Joy-Con 未连接")
                continue
            joycon_right.calibrate_with_feedback()

        elif choice == "b":
            if joycon_left is None or joycon_right is None:
                print("[WARN] 需要两个 Joy-Con 都连接")
                continue
            print("\n[同时校准两个 Joy-Con]")
            print("  请将两个 Joy-Con 都平放在桌面上...")
            joycon_left.calibrate_with_feedback()
            joycon_right.calibrate_with_feedback()

        elif choice == "ml":
            if joycon_left is None:
                print("[WARN] 左 Joy-Con 未连接")
                continue
            show_live_readings(joycon_left, "左 Joy-Con", duration=3.0)

        elif choice == "mr":
            if joycon_right is None:
                print("[WARN] 右 Joy-Con 未连接")
                continue
            show_live_readings(joycon_right, "右 Joy-Con", duration=3.0)

        else:
            print(f"[WARN] 未知选项: '{choice}'")

    print("\n[INFO] 断开连接...")
    if joycon_left is not None:
        joycon_left.disconnect()
    if joycon_right is not None:
        joycon_right.disconnect()
    print("[INFO] 已退出")


if __name__ == "__main__":
    main()
