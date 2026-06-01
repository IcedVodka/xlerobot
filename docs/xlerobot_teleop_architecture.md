# XLerobot 遥操与远程控制代码架构分析

> 本文档梳理了代码库中可用于 XLerobot 主从臂遥操和远程 ZMQ 控制的全部相关代码，为后续实现"Orin 连接本体 + PC 连接主臂"的遥操方案提供参考。

---

## 1. 整体架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          遥操控制端 (PC)                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────────┐ │
│  │ XleBiSO101Leader │  │  KeyboardTeleop  │  │     XLerobotClient         │ │
│  │   (双手主臂)      │  │   (键盘输入)      │  │   (ZMQ 远程客户端)          │ │
│  │  USB /dev/ttyACM*│  │   pynput 监听     │  │   PUSH cmd → Orin:5555     │ │
│  │  读关节位置       │  │   返回按键集合     │  │   PULL obs ← Orin:5556     │ │
│  └────────┬─────────┘  └────────┬─────────┘  └────────────┬───────────────┘ │
│           │                     │                         │                 │
│           └──────────┬──────────┘                         │                 │
│                      ▼                                    │                 │
│            合并为完整 action dict                          │                 │
│     {left_arm_*.pos, right_arm_*.pos,                     │                 │
│      head_motor_*.pos, x.vel, y.vel, theta.vel}          │                 │
│                      │                                    │                 │
│                      └────────────────────────────────────┘                 │
│                                                           ZMQ (TCP)         │
└─────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          机器人本体端 (Orin)                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     XLerobotHost (ZMQ 服务端)                           │ │
│  │   PULL cmd ← PC:5555     PUSH obs → PC:5556                            │ │
│  │   看门狗: 500ms 无命令 → 自动 stop_base()                               │ │
│  └────────────────────────────────┬───────────────────────────────────────┘ │
│                                   │                                         │
│                                   ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        XLerobot (本体机器人)                            │ │
│  │                                                                        │ │
│  │   bus1 (/dev/ttyACM0): 左臂(6 motors) + 头部(2 motors)                  │ │
│  │   bus2 (/dev/ttyACM1): 右臂(6 motors) + 底盘(3 wheels)                  │ │
│  │                                                                        │ │
│  │   connect() → 校准恢复 → configure() → POSITION mode(臂/头)            │ │
│  │                                          VELOCITY mode(底盘)           │ │
│  │   get_observation() → sync_read 位置/速度 → 轮速→体速度转换            │ │
│  │   send_action() → 解析 action dict → sync_write Goal_Position/Velocity │ │
│  │   _body_to_wheel_raw() → 全向轮运动学逆解                              │ │
│  │   _from_keyboard_to_base_action() → 键盘→底盘速度映射                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 关键文件索引

### 2.1 机器人本体 (Orin 端)

| 文件 | 作用 |
|------|------|
| `src/lerobot/robots/xlerobot/xlerobot.py` | XLerobot 本体类。双 bus、双臂+头+全向轮底盘控制 |
| `src/lerobot/robots/xlerobot/xlerobot_host.py` | ZMQ Host。PULL 接收命令，PUSH 发送观测(含图片base64) |
| `src/lerobot/robots/xlerobot/xlerobot_client.py` | ZMQ Client。远程控制端使用 |
| `src/lerobot/robots/xlerobot/config_xlerobot.py` | 配置类：XLerobotConfig / XLerobotHostConfig / XLerobotClientConfig |
| `src/lerobot/motors/feetech/feetech.py` | FeetechMotorsBus 实现，scservo_sdk 驱动 |
| `src/lerobot/motors/motors_bus.py` | MotorsBus 抽象基类，sync_read / sync_write |

### 2.2 遥操输入 (PC 端)

| 文件 | 作用 |
|------|------|
| `src/lerobot/teleoperators/xlebi_so101_leader/xlebi_so101_leader.py` | **双手主臂**。封装两个 SO101Leader，自动加 `left_arm_` / `right_arm_` 前缀 |
| `src/lerobot/teleoperators/so101_leader/so101_leader.py` | 单臂 SO-101 主臂。读 Present_Position |
| `src/lerobot/teleoperators/keyboard/teleop_keyboard.py` | 键盘遥操。`KeyboardTeleop` 返回当前按键集合；`KeyboardRoverTeleop` 返回速度指令 |
| `src/lerobot/teleoperators/teleoperator.py` | Teleoperator 抽象基类 |

### 2.3 遥操主循环与示例

| 文件 | 作用 |
|------|------|
| `src/lerobot/scripts/lerobot_teleoperate.py` | 通用遥操脚本。`teleop_loop()`：get_obs → get_action → process → send_action |
| `examples/lekiwi/teleoperate.py` | **最佳参考示例**。Leader臂 + 键盘 + ZMQ Client 组合遥操 |
| `examples/so100_to_so100_EE/teleoperate.py` | 主从臂直接连接示例（无远程） |
| `examples/xlerobot/4_xlerobot_teleop_keyboard.py` | XLerobot 键盘遥操示例（本地/远程均可） |

### 2.4 ZMQ 示例（非实际控制）

| 文件 | 作用 |
|------|------|
| `script/zmq_server.py` / `zmq_client.py` | 简单 REQ/REP ZMQ 示例 |
| `script/zmq_pub.py` / `zmq_sub.py` | 简单 PUB/SUB ZMQ 示例 |

---

## 3. 核心逻辑详解

### 3.1 ZMQ 远程控制通信协议

**通信模式**: PUSH/PULL（非 REQ/REP）
- **命令通道** (Port 5555): PC(Client, PUSH) → Orin(Host, PULL)
- **观测通道** (Port 5556): Orin(Host, PUSH) → PC(Client, PULL)
- **CONFLATE=1**: 只保留最新一帧，避免队列堆积

**数据格式**:
```python
# Command (PC → Orin): JSON dict
{
    "left_arm_shoulder_pan.pos": 0.5,
    "left_arm_shoulder_lift.pos": -0.3,
    ...,
    "head_motor_1.pos": 0.1,
    "head_motor_2.pos": 0.0,
    "x.vel": 0.2,
    "y.vel": 0.0,
    "theta.vel": 30.0
}

# Observation (Orin → PC): JSON dict，图片字段为 base64 JPEG
{
    "left_arm_shoulder_pan.pos": 0.51,
    ...,
    "camera_name": "base64_jpeg_string"
}
```

**看门狗保护** (`xlerobot_host.py:84-89`):
- 若 500ms 内未收到新命令，自动调用 `robot.stop_base()` 停止底盘
- 防止网络中断时机器人继续运动

### 3.2 XLerobot 本体控制逻辑

**硬件拓扑** (`xlerobot.py:81-127`):
```
bus1 (/dev/ttyACM0):  IDs 1-8
  ├── left_arm_shoulder_pan    (ID 1, sts3215, POSITION)
  ├── left_arm_shoulder_lift   (ID 2, sts3215, POSITION)
  ├── left_arm_elbow_flex      (ID 3, sts3215, POSITION)
  ├── left_arm_wrist_flex      (ID 4, sts3215, POSITION)
  ├── left_arm_wrist_roll      (ID 5, sts3215, POSITION)
  ├── left_arm_gripper         (ID 6, sts3215, RANGE_0_100)
  ├── head_motor_1             (ID 7, sts3215, POSITION)
  └── head_motor_2             (ID 8, sts3215, POSITION)

bus2 (/dev/ttyACM1):  IDs 1-9
  ├── right_arm_shoulder_pan   (ID 1, sts3215, POSITION)
  ├── right_arm_shoulder_lift  (ID 2, sts3215, POSITION)
  ├── right_arm_elbow_flex     (ID 3, sts3215, POSITION)
  ├── right_arm_wrist_flex     (ID 4, sts3215, POSITION)
  ├── right_arm_wrist_roll     (ID 5, sts3215, POSITION)
  ├── right_arm_gripper        (ID 6, sts3215, RANGE_0_100)
  ├── base_left_wheel          (ID 7, sts3215, VELOCITY)
  ├── base_back_wheel          (ID 8, sts3215, VELOCITY)
  └── base_right_wheel         (ID 9, sts3215, VELOCITY)
```

**send_action 流程** (`xlerobot.py:571-636`):
1. 按前缀解析 action dict：`left_arm_*.pos`, `right_arm_*.pos`, `head_*.pos`, `*.vel`
2. 若配置 `max_relative_target`，读取当前位置并做安全限幅
3. 臂/头目标位置 → `sync_write("Goal_Position", ...)` 到 bus1/bus2
4. 底盘体速度 `(x.vel, y.vel, theta.vel)` → `_body_to_wheel_raw()` 全向轮逆解 → `sync_write("Goal_Velocity", ...)` 到 bus2

**全向轮运动学** (`xlerobot.py:381-444`):
- 三个轮子安装角度：240°, 0°, 120°（-90° 偏移后）
- 输入：体坐标系速度 (x m/s, y m/s, theta deg/s)
- 输出：三个轮子的角速度 raw 值

**键盘→底盘映射** (`xlerobot.py:495-528`):
- `i/k`: 前进/后退 (x.vel)
- `j/l`: 左/右平移 (y.vel)
- `u/o`: 左转/右转 (theta.vel)
- `n/m`: 加速/减速 (3 档速度)

### 3.3 双手主臂 XleBiSO101Leader

**结构** (`xlebi_so101_leader.py:38-67`):
- 内部封装两个 `SO101Leader` 实例：`left_arm` 和 `right_arm`
- 各自独立的 USB 串口 (`left_arm_port`, `right_arm_port`)
- 各自独立的校准数据

**get_action 流程** (`xlebi_so101_leader.py:103-119`):
1. 读取左臂 → 键名前加 `left_arm_`
2. 读取右臂 → 键名前加 `right_arm_`
3. 填充头部和底盘占位值（全为 0.0）
```python
# 返回示例
{
    "left_arm_shoulder_pan.pos": 0.5,
    "left_arm_shoulder_lift.pos": -0.3,
    ...,
    "right_arm_shoulder_pan.pos": 0.2,
    ...,
    "head_motor_1.pos": 0.0,   # 占位
    "head_motor_2.pos": 0.0,
    "x.vel": 0.0,
    "y.vel": 0.0,
    "theta.vel": 0.0
}
```

### 3.4 KeyboardTeleop 输入逻辑

**KeyboardTeleop** (`teleop_keyboard.py:50-161`):
- 使用 `pynput.keyboard.Listener` 后台监听按键
- `get_action()` 返回当前所有被按下的键的集合：`{"i": None, "j": None, ...}`
- 不直接输出速度值，只输出原始按键状态

**在 LeKiwi 示例中的用法** (`examples/lekiwi/teleoperate.py:65-66`):
```python
keyboard_keys = keyboard.get_action()          # 获取按键集合
base_action = robot._from_keyboard_to_base_action(keyboard_keys)  # Client 端转换为速度
```

注意：`XLerobotClient` 也实现了 `_from_keyboard_to_base_action()` 方法 (`xlerobot_client.py:282-315`)，与 `XLerobot` 本体的实现完全一致。

---

## 4. 最佳参考：LeKiwi 遥操示例

`examples/lekiwi/teleoperate.py` 是用户需求的**最接近实现**。其逻辑如下：

```python
# 配置
robot_config = LeKiwiClientConfig(remote_ip="192.168.31.165", id="LK12252710")
teleop_arm_config = SO101LeaderConfig(port="COM69", id="R07252710")
keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

# 初始化
robot = LeKiwiClient(robot_config)      # ZMQ 远程客户端
leader_arm = SO101Leader(teleop_arm_config)  # 主臂（单臂）
keyboard = KeyboardTeleop(keyboard_config)   # 键盘

# 连接
robot.connect()
leader_arm.connect()
keyboard.connect()

# 主循环
while True:
    observation = robot.get_observation()       # 从 Orin 获取观测
    arm_action = leader_arm.get_action()        # 读取主臂位置
    arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
    keyboard_keys = keyboard.get_action()       # 读取按键状态
    base_action = robot._from_keyboard_to_base_action(keyboard_keys)  # 转为底盘速度
    action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
    robot.send_action(action)                   # 通过 ZMQ 发送到 Orin
```

---

## 5. 现有能力 vs 用户需求对照

| 需求 | 现有支持 | 状态 |
|------|---------|------|
| Orin 连接 XLerobot 本体 | `XLerobot` + `XLerobotHost` | ✅ 已有 |
| PC 通过 ZMQ 远程控制 | `XLerobotClient` | ✅ 已有 |
| PC 连接两个主臂 | `XleBiSO101Leader` | ✅ 已有 |
| 主臂遥操控制从臂 | `teleop_loop()` / LeKiwi 示例 | ✅ 已有 |
| 键盘控制底盘 | `KeyboardTeleop` + `_from_keyboard_to_base_action()` | ✅ 已有 |
| **键盘控制头部** | 无直接支持 | ⚠️ 需扩展 |
| 一键启动脚本 | 无 | ⚠️ 需新增 |

---

## 6. 实现方案建议

基于以上分析，实现用户需求的**改动最小**的方案如下：

### 方案：复用 LeKiwi 模式 + 扩展头部键盘控制

**Orin 端（机器人本体）**:
```bash
python -m lerobot.robots.xlerobot.xlerobot_host
```
- 运行 `XLerobotHost`，监听 5555/5556 端口
- 内部使用 `XLerobot` 控制双臂+头+底盘

**PC 端（遥操端）**:
- 新建一个遥操脚本（或扩展 `lerobot_teleoperate.py` 的配置能力）
- 使用 `XleBiSO101Leader` 读取双手主臂
- 使用 `KeyboardTeleop` 监听键盘
- 使用 `XLerobotClient` 连接 Orin

**需要新增/修改的代码**:

1. **头部键盘控制映射**: 现有键盘只能控制底盘，需要增加头部电机控制
   - 可选方案 A: 在 `XLerobotClient._from_keyboard_to_base_action()` 中扩展，增加头部按键映射
   - 可选方案 B: 新增一个 `XlerobotKeyboardTeleop` 类，在遥操端统一处理所有输入映射

2. **双手主臂 + 键盘 + ZMQ 的遥操脚本**: 类似 `examples/lekiwi/teleoperate.py`，但使用 `XleBiSO101Leader` + `XLerobotClient`

3. **动作合并逻辑**: `XleBiSO101Leader.get_action()` 返回的 action 包含头部/底盘占位 0，需要与键盘输入合并时正确覆盖

---

*文档生成时间: 2026-05-30*
