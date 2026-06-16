#!/usr/bin/env python

"""
XLerobot 双臂 Joy-Con IMU 姿态遥操脚本 (ZMQ 远程版本)

========================================================================
纯 Joy-Con 控制 — 无遥操臂，利用 IMU 姿态控制双臂末端执行器
========================================================================

控制说明:
    右 Joy-Con (右臂):
        IMU 姿态(倾斜) = 右臂末端执行器姿态 (roll/pitch/yaw)
        垂直摇杆       = X/Z 平移 (前后)
        水平摇杆       = Y 平移 (左右)
        R 按钮         = Z 轴上升
        Stick 按钮     = Z 轴下降
        ZR 按钮        = 右夹爪线性开合 (按住持续动作，首次按下切换方向)
        X/B/Y/A 按钮   = 底盘 前/后/左旋转/右旋转

    左 Joy-Con (左臂 + 头部):
        IMU 姿态(倾斜) = 左臂末端执行器姿态 (roll/pitch/yaw)
        垂直摇杆       = X/Z 平移 (前后)
        水平摇杆       = Y 平移 (左右)
        L 按钮         = Z 轴上升
        Stick 按钮     = Z 轴下降
        ZL 按钮        = 左夹爪线性开合
        D-pad 上/下    = 头部俯仰 (head_motor_2)
        D-pad 左/右    = 头部偏航 (head_motor_1)

    Home 按钮 (右) / Capture 按钮 (左):
        重置双臂+头部到零位

    Plus 按钮 (右) / Minus 按钮 (左):
        重新校准 Joy-Con IMU (放置水平桌面)

========================================================================
底盘速度控制说明:
    - 按住任意底盘控制按钮时，速度线性加速到最大
    - 松开按钮后，速度线性减速到 0
    - 可通过修改以下参数调整加减速斜率:
      * BASE_ACCELERATION_RATE: 加速度斜率 (speed/second)
      * BASE_DECELERATION_RATE: 减速度斜率 (speed/second)
      * BASE_MAX_SPEED: 最大速度倍率

========================================================================
启动命令
========================================================================

1. Orin 端先启动 Host:
    PYTHONPATH=src python -m lerobot.robots.xlerobot.xlerobot_host --robot.id=my_xlerobot

2. PC 端运行本脚本：
    PYTHONPATH=src python teleop/scripts/teleop_bimanual_joycon_zmq.py \
        --remote_ip=10.42.0.192

========================================================================
"""

import argparse
import math
import time

import numpy as np

from lerobot.model.SO101Robot import SO101Kinematics
from lerobot.robots.xlerobot.config_xlerobot import XLerobotClientConfig
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient
from lerobot.utils.robot_utils import precise_sleep
from joyconrobotics import JoyconRobotics

FPS = 30

LEFT_JOINT_MAP = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
}
RIGHT_JOINT_MAP = {
    "shoulder_pan": "right_arm_shoulder_pan",
    "shoulder_lift": "right_arm_shoulder_lift",
    "elbow_flex": "right_arm_elbow_flex",
    "wrist_flex": "right_arm_wrist_flex",
    "wrist_roll": "right_arm_wrist_roll",
    "gripper": "right_arm_gripper",
}

HEAD_MOTOR_MAP = {
    "head_motor_1": "head_motor_1",
    "head_motor_2": "head_motor_2",
}


# =============================================================================
# Joy-Con 控制器 (固定轴向控制)
# =============================================================================

class FixedAxesJoyconRobotics(JoyconRobotics):
    def __init__(self, device, **kwargs):
        super().__init__(device, **kwargs)

        # 为左右 Joy-Con 设置不同的摇杆中心值
        if self.joycon.is_right():
            self.joycon_stick_v_0 = 1900
            self.joycon_stick_h_0 = 2100
        else:  # left Joy-Con
            self.joycon_stick_v_0 = 2300
            self.joycon_stick_h_0 = 2000

        # 夹爪控制相关变量
        self.gripper_speed = 0.4  # 夹爪开合速度 (度/帧)
        self.gripper_direction = 1  # 1 表示打开, -1 表示关闭
        self.gripper_min = 0  # 最小角度 (完全闭合)
        self.gripper_max = 90  # 最大角度 (完全张开)
        self.last_gripper_button_state = 0  # 记录上一帧按钮状态用于检测按下事件

    def common_update(self):
        # 修改后的更新逻辑：摇杆只控制固定轴向
        speed_scale = 0.001

        # 获取当前姿态数据
        orientation_rad = self.get_orientation()
        roll, pitch, yaw = orientation_rad

        # 垂直摇杆：控制 X 和 Z 轴 (前后)
        joycon_stick_v = self.joycon.get_stick_right_vertical() if self.joycon.is_right() else self.joycon.get_stick_left_vertical()
        joycon_stick_v_threshold = 300
        joycon_stick_v_range = 1000
        if joycon_stick_v > joycon_stick_v_threshold + self.joycon_stick_v_0:
            self.position[0] += speed_scale * (joycon_stick_v - self.joycon_stick_v_0) / joycon_stick_v_range * self.dof_speed[0] * self.direction_reverse[0] * math.cos(pitch)
            self.position[2] += speed_scale * (joycon_stick_v - self.joycon_stick_v_0) / joycon_stick_v_range * self.dof_speed[1] * self.direction_reverse[1] * math.sin(pitch)
        elif joycon_stick_v < self.joycon_stick_v_0 - joycon_stick_v_threshold:
            self.position[0] += speed_scale * (joycon_stick_v - self.joycon_stick_v_0) / joycon_stick_v_range * self.dof_speed[0] * self.direction_reverse[0] * math.cos(pitch)
            self.position[2] += speed_scale * (joycon_stick_v - self.joycon_stick_v_0) / joycon_stick_v_range * self.dof_speed[1] * self.direction_reverse[1] * math.sin(pitch)

        # 水平摇杆：只控制 Y 轴 (左右)
        joycon_stick_h = self.joycon.get_stick_right_horizontal() if self.joycon.is_right() else self.joycon.get_stick_left_horizontal()
        joycon_stick_h_threshold = 300
        joycon_stick_h_range = 1000
        if joycon_stick_h > joycon_stick_h_threshold + self.joycon_stick_h_0:
            self.position[1] += speed_scale * (joycon_stick_h - self.joycon_stick_h_0) / joycon_stick_h_range * self.dof_speed[1] * self.direction_reverse[1]
        elif joycon_stick_h < self.joycon_stick_h_0 - joycon_stick_h_threshold:
            self.position[1] += speed_scale * (joycon_stick_h - self.joycon_stick_h_0) / joycon_stick_h_range * self.dof_speed[1] * self.direction_reverse[1]

        # Z 轴按钮控制
        joycon_button_up = self.joycon.get_button_r() if self.joycon.is_right() else self.joycon.get_button_l()
        if joycon_button_up == 1:
            self.position[2] += speed_scale * self.dof_speed[2] * self.direction_reverse[2]

        joycon_button_down = self.joycon.get_button_r_stick() if self.joycon.is_right() else self.joycon.get_button_l_stick()
        if joycon_button_down == 1:
            self.position[2] -= speed_scale * self.dof_speed[2] * self.direction_reverse[2]

        # Home 按钮重置逻辑 (简化版)
        joycon_button_home = self.joycon.get_button_home() if self.joycon.is_right() else self.joycon.get_button_capture()
        if joycon_button_home == 1:
            self.position = self.offset_position_m.copy()

        # 夹爪控制逻辑 (按住线性增减模式)
        for event_type, status in self.button.events():
            if (self.joycon.is_right() and event_type == 'plus' and status == 1) or (self.joycon.is_left() and event_type == 'minus' and status == 1):
                self.reset_button = 1
                self.reset_joycon()
            elif self.joycon.is_right() and event_type == 'a':
                self.next_episode_button = status
            elif self.joycon.is_right() and event_type == 'y':
                self.restart_episode_button = status
            else:
                self.reset_button = 0

        # 夹爪按钮状态检测与方向控制
        gripper_button_pressed = False
        if self.joycon.is_right():
            # 右 Joy-Con 使用 ZR 按钮
            if not self.change_down_to_gripper:
                gripper_button_pressed = self.joycon.get_button_zr() == 1
            else:
                gripper_button_pressed = self.joycon.get_button_stick_r_btn() == 1
        else:
            # 左 Joy-Con 使用 ZL 按钮
            if not self.change_down_to_gripper:
                gripper_button_pressed = self.joycon.get_button_zl() == 1
            else:
                gripper_button_pressed = self.joycon.get_button_stick_l_btn() == 1

        # 检测按钮按下事件 (从 0 到 1) 来切换方向
        if gripper_button_pressed and self.last_gripper_button_state == 0:
            # 按钮刚按下，切换方向
            self.gripper_direction *= -1
            print(f"[GRIPPER] Direction changed to: {'Open' if self.gripper_direction == 1 else 'Close'}")

        # 更新按钮状态记录
        self.last_gripper_button_state = gripper_button_pressed

        # 按住夹爪按钮时线性控制夹爪开合
        if gripper_button_pressed:
            new_gripper_state = self.gripper_state + self.gripper_direction * self.gripper_speed
            # 如果超出限制，停止移动
            if self.gripper_min <= new_gripper_state <= self.gripper_max:
                self.gripper_state = new_gripper_state

        # 按钮控制状态
        if self.joycon.is_right():
            if self.next_episode_button == 1:
                self.button_control = 1
            elif self.restart_episode_button == 1:
                self.button_control = -1
            elif self.reset_button == 1:
                self.button_control = 8
            else:
                self.button_control = 0

        return self.position, self.gripper_state, self.button_control


# =============================================================================
# 臂控制器 (P-control + 逆运动学)
# =============================================================================

class SimpleTeleopArm:
    def __init__(self, joint_map, initial_obs, kinematics, prefix="right", kp=1):
        self.joint_map = joint_map
        self.prefix = prefix
        self.kp = kp
        self.kinematics = kinematics

        # 初始关节位置
        self.joint_positions = {
            "shoulder_pan": initial_obs[f"{prefix}_arm_shoulder_pan.pos"],
            "shoulder_lift": initial_obs[f"{prefix}_arm_shoulder_lift.pos"],
            "elbow_flex": initial_obs[f"{prefix}_arm_elbow_flex.pos"],
            "wrist_flex": initial_obs[f"{prefix}_arm_wrist_flex.pos"],
            "wrist_roll": initial_obs[f"{prefix}_arm_wrist_roll.pos"],
            "gripper": initial_obs[f"{prefix}_arm_gripper.pos"],
        }

        # 设置初始 x/y 为固定值
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0

        # 设置步长
        self.degree_step = 2
        self.xy_step = 0.005

        # P 控制目标位置，设置为零位
        self.target_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
        self.zero_pos = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }

    def move_to_zero_position(self, robot):
        print(f"[{self.prefix}] Moving to Zero Position: {self.zero_pos} ......")
        self.target_positions = self.zero_pos.copy()

        # 重置运动学变量到初始状态
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0

        # 显式设置 wrist_flex
        self.target_positions["wrist_flex"] = 0.0

        action = self.p_control_action(robot)
        robot.send_action(action)

    def handle_joycon_input(self, joycon_pose, gripper_state):
        """处理 Joy-Con 输入，更新臂控制"""
        x, y, z, roll_, pitch_, yaw = joycon_pose

        # 计算 pitch 控制
        pitch = -pitch_ * 60 + 10

        # 设置坐标
        current_x = 0.1629 + x
        current_y = 0.1131 + z

        # 计算 roll
        roll = roll_ * 45

        print(f"[{self.prefix}] pitch: {pitch}")

        # 添加 y 值控制 shoulder_pan 关节
        y_scale = 250.0
        self.target_positions["shoulder_pan"] = y * y_scale

        # 使用逆运动学计算关节角度
        try:
            joint2_target, joint3_target = self.kinematics.inverse_kinematics(current_x, current_y)
            self.target_positions["shoulder_lift"] = joint2_target
            self.target_positions["elbow_flex"] = joint3_target
        except Exception as e:
            print(f"[{self.prefix}] IK failed: {e}")

        # 设置 wrist_flex
        self.target_positions["wrist_flex"] = -self.target_positions["shoulder_lift"] - self.target_positions["elbow_flex"] + pitch

        # 设置 wrist_roll
        self.target_positions["wrist_roll"] = roll

    def p_control_action(self, robot):
        obs = robot.get_observation()
        current = {j: obs[f"{self.prefix}_arm_{j}.pos"] for j in self.joint_map}
        action = {}
        for j in self.target_positions:
            error = self.target_positions[j] - current[j]
            control = self.kp * error
            action[f"{self.joint_map[j]}.pos"] = current[j] + control
        return action


# =============================================================================
# 头部控制器
# =============================================================================

class SimpleHeadControl:
    def __init__(self, initial_obs, kp=1):
        self.kp = kp
        self.degree_step = 2  # 每次移动 2 度
        # 初始化头部电机位置
        self.target_positions = {
            "head_motor_1": initial_obs.get("head_motor_1.pos", 0.0),
            "head_motor_2": initial_obs.get("head_motor_2.pos", 0.0),
        }
        self.zero_pos = {"head_motor_1": 0.0, "head_motor_2": 0.0}

    def move_to_zero_position(self, robot):
        print("[HEAD] Moving to Zero Position: {self.zero_pos} ......")
        self.target_positions = self.zero_pos.copy()
        action = self.p_control_action(robot)
        robot.send_action(action)

    def handle_joycon_input(self, joycon):
        """处理左 Joy-Con 方向键输入控制头部电机"""
        # 获取左 Joy-Con 方向键状态
        button_up = joycon.joycon.get_button_up()      # 上: head_motor_2+
        button_down = joycon.joycon.get_button_down()  # 下: head_motor_2-
        button_left = joycon.joycon.get_button_left()  # 左: head_motor_1+
        button_right = joycon.joycon.get_button_right() # 右: head_motor_1-

        if button_up == 1:
            self.target_positions["head_motor_2"] += self.degree_step
            print(f"[HEAD] head_motor_2: {self.target_positions['head_motor_2']}")
        if button_down == 1:
            self.target_positions["head_motor_2"] -= self.degree_step
            print(f"[HEAD] head_motor_2: {self.target_positions['head_motor_2']}")
        if button_left == 1:
            self.target_positions["head_motor_1"] += self.degree_step
            print(f"[HEAD] head_motor_1: {self.target_positions['head_motor_1']}")
        if button_right == 1:
            self.target_positions["head_motor_1"] -= self.degree_step
            print(f"[HEAD] head_motor_1: {self.target_positions['head_motor_1']}")

    def p_control_action(self, robot):
        obs = robot.get_observation()
        action = {}
        for motor in self.target_positions:
            current = obs.get(f"{HEAD_MOTOR_MAP[motor]}.pos", 0.0)
            error = self.target_positions[motor] - current
            control = self.kp * error
            action[f"{HEAD_MOTOR_MAP[motor]}.pos"] = current + control
        return action


# =============================================================================
# 底盘控制
# =============================================================================

def get_joycon_base_action(joycon, robot):
    """
    从 Joy-Con 获取底盘控制指令
    X: 前进, B: 后退, Y: 左转, A: 右转
    """
    # 获取按钮状态
    button_x = joycon.joycon.get_button_x()  # 前进
    button_b = joycon.joycon.get_button_b()  # 后退
    button_y = joycon.joycon.get_button_y()  # 左转
    button_a = joycon.joycon.get_button_a()  # 右转

    # 构建按键集合 (模拟键盘输入)
    pressed_keys = set()

    if button_x == 1:
        pressed_keys.add('k')  # 前进
        print("[BASE] Forward")
    if button_b == 1:
        pressed_keys.add('i')  # 后退
        print("[BASE] Backward")
    if button_y == 1:
        pressed_keys.add('u')  # 左转
        print("[BASE] Left turn")
    if button_a == 1:
        pressed_keys.add('o')  # 右转
        print("[BASE] Right turn")

    # 转换为 numpy 数组并获取底盘动作
    keyboard_keys = np.array(list(pressed_keys))
    base_action = robot._from_keyboard_to_base_action(keyboard_keys) or {}

    return base_action


# 底盘速度控制参数 —— 可调整斜率
BASE_ACCELERATION_RATE = 2.0  # 加速度斜率 (speed/second)
BASE_DECELERATION_RATE = 2.5  # 减速度斜率 (speed/second)
BASE_MAX_SPEED = 3.0          # 最大速度倍率


def get_joycon_speed_control(joycon):
    """
    从 Joy-Con 获取速度控制 —— 线性加速和减速
    按住任意底盘控制按钮时线性加速到最大速度，松开时线性减速到 0
    """
    global current_base_speed, last_update_time, is_accelerating

    # 初始化全局变量
    if 'current_base_speed' not in globals():
        current_base_speed = 0.0
        last_update_time = time.time()
        is_accelerating = False

    current_time = time.time()
    dt = current_time - last_update_time
    last_update_time = current_time

    # 检查是否有底盘控制按钮被按下
    button_x = joycon.joycon.get_button_x()  # 前进
    button_b = joycon.joycon.get_button_b()  # 后退
    button_y = joycon.joycon.get_button_y()  # 左转
    button_a = joycon.joycon.get_button_a()  # 右转

    any_base_button_pressed = any([button_x, button_b, button_y, button_a])

    if any_base_button_pressed:
        # 按钮按下 - 加速
        if not is_accelerating:
            is_accelerating = True
            print("[BASE] Starting acceleration")

        # 线性加速
        current_base_speed += BASE_ACCELERATION_RATE * dt
        current_base_speed = min(current_base_speed, BASE_MAX_SPEED)

    else:
        # 无按钮按下 - 减速
        if is_accelerating:
            is_accelerating = False
            print("[BASE] Starting deceleration")

        # 线性减速
        current_base_speed -= BASE_DECELERATION_RATE * dt
        current_base_speed = max(current_base_speed, 0.0)

    # 打印当前速度 (可选，用于调试)
    if abs(current_base_speed) > 0.01:  # 仅在速度不为 0 时打印
        print(f"[BASE] Current speed: {current_base_speed:.2f}")

    return current_base_speed


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="XLerobot bimanual Joy-Con IMU pose teleoperation via ZMQ (no leader arms, no viz, no recording)"
    )
    parser.add_argument("--remote_ip", type=str, required=True, help="Orin IP address")
    parser.add_argument("--fps", type=int, default=FPS, help="Control loop frequency (Hz)")
    parser.add_argument("--kp", type=float, default=1.0, help="P-control proportional gain")
    args = parser.parse_args()

    # 初始化远程机器人客户端 (ZMQ)
    print(f"[MAIN] Connecting to remote robot at {args.remote_ip}...")
    robot_config = XLerobotClientConfig(
        remote_ip=args.remote_ip,
        id="xlerobot_teleop_joycon",
        cameras={},  # 无需图像解码
    )
    robot = XLerobotClient(robot_config)

    try:
        robot.connect()
        print("[MAIN] Successfully connected to remote robot")
    except Exception as e:
        print(f"[MAIN] Failed to connect to remote robot: {e}")
        return

    # 初始化右 Joy-Con 控制器
    print("[MAIN] Initializing right Joy-Con controller...")
    joycon_right = FixedAxesJoyconRobotics(
        "right",
        dof_speed=[2, 2, 2, 1, 1, 1]
    )
    print("[MAIN] Right Joy-Con controller connected")

    # 初始化左 Joy-Con 控制器
    print("[MAIN] Initializing left Joy-Con controller...")
    joycon_left = FixedAxesJoyconRobotics(
        "left",
        dof_speed=[2, 2, 2, 1, 1, 1]
    )
    print("[MAIN] Left Joy-Con controller connected")

    # 初始化臂和头部实例
    obs = robot.get_observation()
    kin_left = SO101Kinematics()
    kin_right = SO101Kinematics()
    left_arm = SimpleTeleopArm(LEFT_JOINT_MAP, obs, kin_left, prefix="left", kp=args.kp)
    right_arm = SimpleTeleopArm(RIGHT_JOINT_MAP, obs, kin_right, prefix="right", kp=args.kp)
    head_control = SimpleHeadControl(obs, kp=args.kp)

    # 启动时双臂和头部移动到零位
    print("[MAIN] Moving arms and head to zero position...")
    left_arm.move_to_zero_position(robot)
    right_arm.move_to_zero_position(robot)
    head_control.move_to_zero_position(robot)

    print("\n[INFO] All devices connected. Starting teleop loop...")
    print("  Right Joy-Con (右臂 + 底盘):")
    print("    IMU 姿态 = 右臂末端姿态")
    print("    摇杆     = X/Y/Z 平移")
    print("    R/Stick  = Z 上升/下降")
    print("    ZR       = 右夹爪")
    print("    X/B/Y/A  = 底盘 前/后/左转/右转")
    print("  Left Joy-Con (左臂 + 头部):")
    print("    IMU 姿态 = 左臂末端姿态")
    print("    摇杆     = X/Y/Z 平移")
    print("    L/Stick  = Z 上升/下降")
    print("    ZL       = 左夹爪")
    print("    D-pad    = 头部俯仰/偏航")
    print("  Home/Capture = 归零")
    print("  Plus/Minus   = 重新校准 Joy-Con")
    print("  Exit: Ctrl+C\n")

    try:
        while True:
            t0 = time.perf_counter()

            # 1. 获取左右 Joy-Con 姿态和夹爪状态
            pose_right, gripper_right, control_button_right = joycon_right.get_control()
            print(f"pose_right: {pose_right}, gripper_right: {gripper_right}, control_button_right: {control_button_right}")
            pose_left, gripper_left, control_button_left = joycon_left.get_control()
            print(f"pose_left: {pose_left}, gripper_left: {gripper_left}, control_button_left: {control_button_left}")

            # 2. 处理重置 (Home/Capture 按钮)
            if control_button_right == 8:
                print("[MAIN] Reset to zero position!")
                right_arm.move_to_zero_position(robot)
                left_arm.move_to_zero_position(robot)
                head_control.move_to_zero_position(robot)
                continue

            # 3. 更新双臂目标位置
            right_arm.target_positions["gripper"] = gripper_right
            left_arm.target_positions["gripper"] = gripper_left

            right_arm.handle_joycon_input(pose_right, gripper_right)
            right_action = right_arm.p_control_action(robot)
            left_arm.handle_joycon_input(pose_left, gripper_left)
            left_action = left_arm.p_control_action(robot)

            # 4. 更新头部
            head_control.handle_joycon_input(joycon_left)
            head_action = head_control.p_control_action(robot)

            # 5. 获取底盘动作 (带加减速)
            base_action = get_joycon_base_action(joycon_right, robot)
            speed_multiplier = get_joycon_speed_control(joycon_right)

            if base_action:
                for key in base_action:
                    if 'vel' in key or 'velocity' in key:
                        base_action[key] *= speed_multiplier

            # 6. 合并并发送
            action = {**left_action, **right_action, **head_action, **base_action}
            robot.send_action(action)

            # 7. 接收观测 (维持 ZMQ 连接，但忽略数据)
            obs = robot.get_observation()

            # 8. 频率控制
            dt = time.perf_counter() - t0
            precise_sleep(max(1.0 / args.fps - dt, 0.0))

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        print("[INFO] Disconnecting...")
        joycon_right.disconnect()
        joycon_left.disconnect()
        if robot.is_connected:
            robot.disconnect()
        print("[INFO] Done")


if __name__ == "__main__":
    main()
