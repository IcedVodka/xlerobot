# XLerobot 遥操与数据采集系统 — 项目代码调研报告

> 调研时间：2026-06-02
> 基于代码库：`/home/gml-cwl/code4/xlerobot`
> LeRobot 版本：v3.0

---

## 目录

1. [项目整体架构](#1-项目整体架构)
2. [遥操作系统](#2-遥操作系统)
3. [远程通信网络层](#3-远程通信网络层)
4. [机器人硬件接口](#4-机器人硬件接口)
5. [数据采集与 LeRobot v3 数据集](#5-数据采集与-lerobot-v3-数据集)
6. [模型训练与策略](#6-模型训练与策略)
7. [评估与推理部署](#7-评估与推理部署)
8. [摄像头系统](#8-摄像头系统)
9. [处理器流水线](#9-处理器流水线)
10. [电机与运动控制](#10-电机与运动控制)
11. [可复用资产清单](#11-可复用资产清单)
12. [现有示例脚本汇总](#12-现有示例脚本汇总)
13. [关键结论与建议](#13-关键结论与建议)

---

## 1. 项目整体架构

```
xlerobot/                          # 项目根目录
├── src/lerobot/                   # 核心库
│   ├── cameras/                   # 相机抽象层 (OpenCV, RealSense)
│   ├── datasets/                  # LeRobotDataset v3.0 实现
│   ├── motors/                    # 电机总线抽象 (Feetech, Dynamixel)
│   ├── policies/                  # 策略模型 (ACT, Diffusion, PI0, ...)
│   ├── processor/                 # 数据处理器流水线
│   ├── robots/                    # 机器人硬件接口
│   ├── teleoperators/             # 遥操设备接口
│   ├── utils/                     # 工具函数
│   └── scripts/                   # CLI 脚本 (train, eval, record, replay)
├── examples/                      # 示例代码
│   ├── xlerobot/                  # XLerobot 专用示例
│   ├── training/                  # 训练示例
│   └── tutorial/                  # 教程 (ACT, Diffusion, ...)
├── docs/                          # 文档
└── teleop/                        # 【新建】遥操系统开发目录
    ├── docs/
    ├── src/
    ├── configs/
    ├── scripts/
    └── tests/
```

---

## 2. 遥操作系统

### 2.1 抽象基类

**文件**：`src/lerobot/teleoperators/teleoperator.py`

所有遥操设备继承此抽象基类，定义标准接口：

```python
class Teleoperator(ABC):
    @abstractmethod
    def connect(self) -> None: ...
    @abstractmethod
    def disconnect(self) -> None: ...
    @abstractmethod
    def get_action(self) -> dict[str, float]: ...
    @abstractmethod
    def send_feedback(self, feedback: dict[str, float]) -> None: ...
    @abstractmethod
    def calibrate(self) -> None: ...
```

**可复用价值**：高。新增任何遥操设备只需继承此类并实现接口。

### 2.2 键盘遥操

**核心文件**：
- `src/lerobot/teleoperators/keyboard/teleop_keyboard.py`
- `src/lerobot/teleoperators/keyboard/configuration_keyboard.py`

**类**：`KeyboardTeleop`、`KeyboardEndEffectorTeleop`、`KeyboardRoverTeleop`

**功能**：
- 基于 `pynput` 库实现全局键盘监听
- 支持按键状态查询（按下/释放）
- `KeyboardEndEffectorTeleop`：增量 XYZ + 夹爪控制
- `KeyboardRoverTeleop`：WASD 式移动底盘控制 + 速度调节

**配置类**：`KeyboardTeleopConfig` — 可自定义按键映射

**示例**：
- `examples/xlerobot/0_so100_keyboard_joint_control.py` — SO100 关节级键盘控制
- `examples/xlerobot/1_so100_keyboard_ee_control.py` — SO100 末端执行器键盘控制（含 IK）
- `examples/xlerobot/4_xlerobot_teleop_keyboard.py` — 完整 XLerobot 键盘遥操（双臂 + 头 + 底盘）

**可复用价值**：高。直接用于 PC 端键盘控制头部和底盘运动。

### 2.3 Joy-Con (Switch 手柄) 遥操

**核心文件**：无内置类，示例脚本中直接实现

**示例文件**：
- `examples/xlerobot/6_so100_joycon_ee_control.py` — SO100 单臂 Joy-Con EE 控制
- `examples/xlerobot/7_xlerobot_teleop_joycon.py` — 完整 XLerobot Joy-Con 遥操
- `examples/xlerobot/7_xlerobot_2wheels_teleop_joycon.py` — 两轮版
- `examples/xlerobot/7_xlerobot_2wheels_teleop_joycon_smooth.py` — 平滑加速版

**控制映射**（以 `7_xlerobot_teleop_joycon.py` 为例）：
```
右 Joy-Con:
  - 摇杆: 右臂 XY 平面移动（IK）
  - 上下按钮 (R/StickR): 右臂 Z 轴
  - ZR: 右臂夹爪
  - 方向键: 底盘运动 (X=前, B=后, Y=左转, A=右转)
  - Home: 归零
  - +/-: 下一 episode / 重录 episode

左 Joy-Con:
  - 摇杆: 左臂 XY 平面移动（IK）
  - 上下按钮 (L/StickL): 左臂 Z 轴
  - ZL: 左臂夹爪
  - 方向键: 头部控制 (上/下=pitch, 左/右=yaw)
  - Capture: 归零
```

**依赖**：`joyconrobotics` 库（蓝牙 HID 通信）

**实现细节**：
- `FixedAxesJoyconRobotics` 继承 `JoyconRobotics`，重写 `common_update()`
- 内置加减速曲线：`BASE_ACCELERATION_RATE=2.0`, `BASE_DECELERATION_RATE=2.5`
- 夹爪采用方向切换模式：按一次 ZR/ZL 改变开合方向，持续按住线性增减

**可复用价值**：中。代码在示例脚本中，未封装为 Teleoperator 子类，需要提取封装。

### 2.4 Xbox/Gamepad 遥操

**核心文件**：
- `src/lerobot/teleoperators/gamepad/teleop_gamepad.py`
- `src/lerobot/teleoperators/gamepad/gamepad_utils.py`
- `src/lerobot/teleoperators/gamepad/configuration_gamepad.py`

**类**：`GamepadTeleop`、`GamepadController`、`GamepadControllerHID`

**功能**：
- Linux/Windows 使用 `pygame`，macOS 使用 `hidapi`
- 支持增量 XYZ + 夹爪控制
- 支持 episode 控制事件（开始/停止录制）

**示例**：`examples/xlerobot/5_xlerobot_teleop_xbox.py`

**可复用价值**：高。已封装为 Teleoperator 子类。

### 2.5 VR 遥操

**核心文件**：
- `src/lerobot/teleoperators/xlerobot_vr/xlerobot_vr.py`
- `src/lerobot/teleoperators/xlerobot_vr/vr_monitor.py`
- `src/lerobot/teleoperators/xlerobot_vr/configuration_xlerobot_vr.py`

**类**：`XLerobotVRTeleop`

**XLeVR 子系统**（独立 Web VR）：
- `XLeVR/xlevr/inputs/vr_ws_server.py` — WebSocket 服务器接收 VR 控制器数据
- `XLeVR/web-ui/index.html` — A-Frame 构建的 Web VR 界面
- `XLeVR/web-ui/vr_app.js` — 处理控制器跟踪、WebSocket 发送

**支持**：位置、旋转（四元数）、握力、扳机、摇杆、按钮
- 绝对/相对位置控制模式
- 支持 episode 事件：exit_early、rerecord、stop、reset

**示例**：`examples/xlerobot/8_xlerobot_teleop_vr.py`

**可复用价值**：高。VR 遥操已完整封装。

### 2.6 主臂遥操（物理示教）

**双主臂 SO-101**：
- `src/lerobot/teleoperators/xlebi_so101_leader/xlebi_so101_leader.py`
- `src/lerobot/teleoperators/xlebi_so101_leader/config_xlebi_so101_leader.py`

**类**：`XleBiSO101Leader`

**功能**：
- 封装左右两个 `SO101Leader`
- `get_action()` 自动添加 `left_arm_` / `right_arm_` 前缀
- 自动补齐 head/base 占位动作（值为 0）
- `send_feedback()` 自动分发到对应主臂

**示例**：`examples/xlerobot/teleop_bimanual_zmq.py`

**可复用价值**：高。主臂遥操的核心封装。

---

## 3. 远程通信网络层

### 3.1 ZMQ Host-Client 架构

**文件**：
- `src/lerobot/robots/xlerobot/xlerobot_host.py`
- `src/lerobot/robots/xlerobot/xlerobot_client.py`
- `src/lerobot/robots/xlerobot/config_xlerobot.py`

**端口配置**：
```python
@dataclass
class XLerobotHostConfig:
    port_zmq_cmd: int = 5555          # 命令接收端口
    port_zmq_observations: int = 5556 # 观测发送端口
    connection_time_s: int = 3600     # 连接持续时间
    watchdog_timeout_ms: int = 500    # 看门狗超时
    max_loop_freq_hz: int = 30        # 最大循环频率

@dataclass
class XLerobotClientConfig:
    remote_ip: str                    # Orin IP 地址
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5
```

**Host 端**（运行在 Orin 上）：
```python
class XLerobotHost:
    def __init__(self, config):
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)  # 接收命令
        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)  # 发送观测
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)         # 只保留最新消息
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
```

**Host 主循环**：
1. `zmq.NOBLOCK` 接收命令 → `robot.send_action(data)`
2. 看门狗：500ms 无命令 → `robot.stop_base()`
3. `robot.get_observation()` 获取观测
4. JPEG base64 编码图像 → ZMQ PUSH 发送
5. 控制频率：`max_loop_freq_hz=30`

**Client 端**（运行在 PC 上）：
```python
class XLerobotClient(Robot):
    def __init__(self, config):
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket = self.zmq_context.socket(zmq.PULL)
        # ...

    def get_observation(self):
        frames, obs_dict = self._get_data()  # 轮询 ZMQ PULL
        # base64 JPEG 解码 → numpy 图像数组
        return obs_dict  # 包含状态和图像

    def send_action(self, action):
        self.zmq_cmd_socket.send_string(json.dumps(action))
```

### 3.2 图像传输

**编码方式**：JPEG base64
```python
ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
last_observation[cam_key] = base64.b64encode(buffer).decode("utf-8")
```

**带宽估算**（3 相机 @ 640x480 @ 30fps）：
- 单帧 JPEG 约 50~150KB
- 总计约 150~450KB/帧
- 30fps 时约 4.5~13.5 MB/s
- 千兆局域网完全可行

**优化方向**：H.264 视频流编码可大幅降低带宽

### 3.3 看门狗安全机制

```python
if (now - last_cmd_time > watchdog_timeout_ms / 1000) and not watchdog_active:
    robot.stop_base()
    watchdog_active = True
```

**可复用价值**：高。ZMQ Host-Client 架构已完整实现，可直接使用。

---

## 4. 机器人硬件接口

### 4.1 XLerobot 本体

**文件**：`src/lerobot/robots/xlerobot/xlerobot.py`

**类**：`XLerobot`

**硬件组成**：
```
Bus1 (port1):
  - 左臂: left_arm_shoulder_pan(1) ~ left_arm_gripper(6)
  - 头部: head_motor_1(7), head_motor_2(8)

Bus2 (port2):
  - 右臂: right_arm_shoulder_pan(1) ~ right_arm_gripper(6)
  - 底盘: base_left_wheel(7), base_back_wheel(8), base_right_wheel(9)
```

**观测特征**（`observation_features`）：
```python
{
    # 左臂关节位置
    "left_arm_shoulder_pan.pos", "left_arm_shoulder_lift.pos",
    "left_arm_elbow_flex.pos", "left_arm_wrist_flex.pos",
    "left_arm_wrist_roll.pos", "left_arm_gripper.pos",
    # 右臂关节位置
    "right_arm_shoulder_pan.pos", "right_arm_shoulder_lift.pos",
    "right_arm_elbow_flex.pos", "right_arm_wrist_flex.pos",
    "right_arm_wrist_roll.pos", "right_arm_gripper.pos",
    # 头部关节位置
    "head_motor_1.pos", "head_motor_2.pos",
    # 底盘速度（世界坐标系）
    "x.vel", "y.vel", "theta.vel",
    # 相机图像
    "right": (480, 640, 3), "left": (480, 640, 3), "head": (480, 640, 3)
}
```

**动作特征**（`action_features`）：与状态特征同构

### 4.2 底盘运动学

**正向运动学**（轮速 → 体速度）：
```python
def _wheel_raw_to_body(self, left, back, right):
    # 全向轮安装角度: 240°, 0°, 120°
    # 逆运动学矩阵求解
    return {"x.vel": x, "y.vel": y, "theta.vel": theta}
```

**逆向运动学**（体速度 → 轮速）：
```python
def _body_to_wheel_raw(self, x, y, theta):
    # 运动学矩阵映射
    # 自动缩放防止超速
    return {"base_left_wheel": raw0, "base_back_wheel": raw1, "base_right_wheel": raw2}
```

**速度档位**：
```python
self.speed_levels = [
    {"xy": 0.1, "theta": 30},   # 慢速
    {"xy": 0.2, "theta": 60},   # 中速
    {"xy": 0.3, "theta": 90},   # 快速
]
```

### 4.3 键盘 → 底盘控制

```python
def _from_keyboard_to_base_action(self, pressed_keys):
    # I/K = 前进/后退
    # J/L = 左/右平移
    # U/O = 左转/右转
    # N/M = 速度加/减
    return {"x.vel": x_cmd, "y.vel": y_cmd, "theta.vel": theta_cmd}
```

### 4.4 头部控制

```python
def _from_keyboard_to_head_action(pressed_keys, current_head_pos, step_deg):
    # T/G = head_motor_1 抬头/低头 (pitch)
    # F/H = head_motor_2 左转/右转 (yaw)
    return {"head_motor_1.pos": ..., "head_motor_2.pos": ...}
```

头部采用**位置控制**模式，按键改变目标角度，松开后保持当前角度。

### 4.5 安全控制

```python
def send_action(self, action):
    # 1. 分离 left/right/head/base 动作
    # 2. max_relative_target 安全限幅
    # 3. 转换 key 名（去掉 .pos 后缀）
    # 4. sync_write 到对应总线
    # 5. 返回实际发送的动作
```

### 4.6 配置

**文件**：`src/lerobot/robots/xlerobot/config_xlerobot.py`

```python
@RobotConfig.register_subclass("xlerobot")
@dataclass
class XLerobotConfig(RobotConfig):
    port1: str = "/dev/serial/by-id/..."  # 左臂 + 头部
    port2: str = "/dev/serial/by-id/..."  # 右臂 + 底盘
    cameras: dict = xlerobot_cameras_config()  # right/left(OpenCV) + head(RealSense)
    teleop_keys: dict = {...}  # 默认键盘映射
```

**可复用价值**：高。XLerobot 硬件接口已完整封装。

---

## 5. 数据采集与 LeRobot v3 数据集

### 5.1 核心数据集类

**文件**：`src/lerobot/datasets/lerobot_dataset.py`

**类**：`LeRobotDataset`（1765 行）

**关键方法**：
```python
# 创建新数据集
LeRobotDataset.create(
    repo_id="user/dataset_name",
    fps=30,
    features=dataset_features,
    robot_type="xlerobot_client",
    use_videos=True,
    image_writer_processes=0,
    image_writer_threads=12,
)

# 添加帧
dataset.add_frame(frame_dict)

# 保存 episode
dataset.save_episode()

# 关闭并 finalize
dataset.finalize()

# 上传到 HuggingFace Hub
dataset.push_to_hub()
```

### 5.2 v3.0 数据格式

```
dataset/
├── data/
│   └── chunk-000/
│       ├── file-000.parquet       # 多 episode 合并存储
│       └── file-001.parquet
├── meta/
│   ├── info.json                  # schema, FPS, features, path templates
│   ├── stats.json                 # 归一化统计 (mean, std, q01, q10, q50, q90, q99)
│   ├── tasks.parquet              # 任务描述
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet   # episode 元数据
└── videos/
    └── observation.images.camera/
        └── chunk-000/
            └── file-000.mp4       # 合并视频 chunk
```

**Episode 元数据字段**：
```python
{
    "episode_index": int,
    "tasks": str,
    "length": int,
    "data/chunk_index": int,
    "data/file_index": int,
    "videos/{camera}/chunk_index": int,
    "videos/{camera}/file_index": int,
    "videos/{camera}/from_timestamp": float,
    "videos/{camera}/to_timestamp": float,
    "dataset_from_index": int,
    "dataset_to_index": int,
}
```

### 5.3 录制主脚本

**文件**：`src/lerobot/scripts/lerobot_record.py`

**核心录制循环** `record_loop()`（239~408 行）：

```python
def record_loop(
    robot, events, fps,
    teleop_action_processor, robot_action_processor, robot_observation_processor,
    dataset=None, teleop=None, policy=None,
    control_time_s=None, single_task=None, display_data=False,
):
    while timestamp < control_time_s:
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix="observation")

        # 获取动作 (teleop 或 policy)
        if policy:
            action_values = predict_action(observation_frame, policy, ...)
        else:
            act = teleop.get_action()
            act_processed = teleop_action_processor((act, obs))

        robot_action_to_send = robot_action_processor((act_processed, obs))
        sent_action = robot.send_action(robot_action_to_send)

        # 保存到数据集
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, act_processed, prefix="action")
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(observation=obs_processed, action=act_processed)

        precise_sleep(1/fps - dt)
```

### 5.4 特征转换工具

**文件**：`src/lerobot/datasets/utils.py`

```python
# 硬件特征 → 数据集特征
def hw_to_dataset_features(features, prefix):
    # 例如: "left_arm_shoulder_pan.pos" → "observation.state" 或 "action.left_arm_shoulder_pan"

# 构建数据集帧
def build_dataset_frame(dataset_features, data, prefix="observation"):
    # 将观测/动作数据转换为符合 dataset features 的字典
```

### 5.5 视频编码

**文件**：`src/lerobot/datasets/video_utils.py`

```python
class VideoEncodingManager:
    """上下文管理器，自动批量编码视频"""
    def __enter__(self):
        # 启动编码管理
    def __exit__(self, ...):
        # 批量编码所有 episode 视频

def encode_video_frames(frames, output_path, fps=30):
    # PyAV/ffmpeg 编码

def decode_video_frames(video_path, start_frame, end_frame):
    # 支持 torchcodec, pyav, video_reader
```

### 5.6 异步图像写入

**文件**：`src/lerobot/datasets/image_writer.py`

```python
class AsyncImageWriter:
    """多线程/多进程异步写入 PNG"""
```

**推荐配置**：
```python
num_image_writer_processes = 0       # 0=线程, ≥1=进程
num_image_writer_threads_per_camera = 4  # 每相机线程数
```

### 5.7 数据集工具

**文件**：`src/lerobot/datasets/dataset_tools.py`

**功能**：
- `delete_episodes()` — 删除指定 episode
- `split_dataset()` — 按分数或索引拆分
- `merge_datasets()` — 合并多个数据集
- `modify_features()` — 添加/删除特征
- `remove_feature()` — 删除特征（如相机）

### 5.8 统计计算

**文件**：`src/lerobot/datasets/compute_stats.py`

```python
class RunningQuantileStats:
    """运行时直方图分位数统计 (q01, q10, q50, q90, q99)"""
```

### 5.9 录制配置

**文件**：`src/lerobot/scripts/lerobot_record.py`（163~231 行）

```python
@dataclass
class DatasetRecordConfig:
    repo_id: str
    single_task: str
    fps: int = 30
    episode_time_s: int | float = 60
    reset_time_s: int | float = 60
    num_episodes: int = 50
    video: bool = True
    push_to_hub: bool = True
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1

@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    teleop: TeleoperatorConfig | None = None
    policy: PreTrainedConfig | None = None
    display_data: bool = False
    play_sounds: bool = True
    resume: bool = False
```

**可复用价值**：高。录制流程已完整封装，支持 teleop/policy 双模式。

---

## 6. 模型训练与策略

### 6.1 策略基类

**文件**：`src/lerobot/policies/pretrained.py`

```python
class PreTrainedPolicy:
    def forward(self, batch): ...
    def select_action(self, observation): ...
    def predict_action_chunk(self, observation): ...
    def reset(self): ...
    def get_optim_params(self): ...
```

### 6.2 策略工厂

**文件**：`src/lerobot/policies/factory.py`

```python
def get_policy_class(policy_type): ...
def make_policy(config, ds_meta): ...
def make_pre_post_processors(policy_cfg, pretrained_path, dataset_stats): ...
```

### 6.3 支持策略

| 策略 | 模型文件 | 配置 | 说明 |
|------|---------|------|------|
| ACT | `policies/act/modeling_act.py` | `configuration_act.py` | VAE + Transformer |
| Diffusion Policy | `policies/diffusion/modeling_diffusion.py` | `configuration_diffusion.py` | DDPM + UNet1D |
| PI0 | `policies/pi0/modeling_pi0.py` | `configuration_pi0.py` | PaliGemma VLA |
| PI05 | `policies/pi05/modeling_pi05.py` | `configuration_pi05.py` | |
| VQBeT | `policies/vqbet/modeling_vqbet.py` | `configuration_vqbet.py` | |
| TDMPC | `policies/tdmpc/modeling_tdmpc.py` | `configuration_tdmpc.py` | |
| SAC | `policies/sac/modeling_sac.py` | `configuration_sac.py` | |
| SmolVLA | `policies/smolvla/modeling_smolvla.py` | `configuration_smolvla.py` | 轻量 VLA |
| GROOT | `policies/groot/modeling_groot.py` | `configuration_groot.py` | |
| XVLA | `policies/xvla/modeling_xvla.py` | `configuration_xvla.py` | |
| Wall-X | `policies/wall_x/modeling_wall_x.py` | `configuration_wall_x.py` | |
| SARM | `policies/sarm/modeling_sarm.py` | `configuration_sarm.py` | 子任务奖励模型 |
| RTC | `policies/rtc/modeling_rtc.py` | `configuration_rtc.py` | 实时控制 |

### 6.4 训练配置

**文件**：`src/lerobot/configs/train.py`

```python
@dataclass
class TrainPipelineConfig:
    dataset: DatasetConfig
    policy: PreTrainedConfig
    output_dir: Path | None = None
    batch_size: int = 8
    steps: int = 100_000
    eval_freq: int = 20_000
    save_freq: int = 20_000
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandBConfig = field(default_factory=WandbConfig)
    # RA-BC 参数
    use_rabc: bool = False
    rabc_kappa: float = 0.01
```

### 6.5 训练 CLI

```bash
# ACT 训练
lerobot-train \
    --dataset.repo_id=user/dataset \
    --policy.type=act \
    --policy.dim_model=512 \
    --output_dir=outputs/train/act_run \
    --steps=100000

# 带评估
lerobot-train \
    --dataset.repo_id=user/dataset \
    --policy.type=diffusion \
    --eval.n_episodes=10 \
    --eval_freq=20000
```

### 6.6 训练示例

- `examples/training/train_policy.py` — 策略训练示例
- `examples/training/train_with_streaming.py` — 流式数据集训练
- `examples/tutorial/act/act_training_example.py` — ACT 教程
- `examples/tutorial/diffusion/diffusion_training_example.py` — Diffusion Policy 教程

**可复用价值**：高。训练脚本和配置已完整封装。

---

## 7. 评估与推理部署

### 7.1 评估脚本

**文件**：`src/lerobot/scripts/lerobot_eval.py`

```bash
lerobot-eval \
    --robot.type=xlerobot_client \
    --robot.remote_ip=10.42.0.192 \
    --policy.path=outputs/train/act_run/checkpoint/last/pretrained_model \
    --eval.n_episodes=10 \
    --eval.batch_size=1
```

**功能**：
- 批量策略 rollout
- 计算成功率、奖励
- 保存评估视频
- 支持多任务并行评估

### 7.2 重放脚本

**文件**：`src/lerobot/scripts/lerobot_replay.py`

```bash
lerobot-replay \
    --robot.type=xlerobot_client \
    --robot.remote_ip=10.42.0.192 \
    --dataset.repo_id=user/dataset \
    --episode_index=0
```

**功能**：重放数据集 episode 到机器人上

### 7.3 异步推理（gRPC）

**文件**：`src/lerobot/async_inference/`

```
policy_server.py    # gRPC PolicyServer — 接收观测，运行策略推理
robot_client.py     # gRPC RobotClient — 连接策略服务器，执行动作
configs.py          # 配置
helpers.py          # TimedAction, TimedObservation, FPSTracker
constants.py        # 常量
transport/          # protobuf 定义
```

**PolicyServer**：
- 接收策略指令（类型、预训练路径、设备）
- 接收观测数据
- 运行策略推理并返回动作块
- 支持观测队列、去重、FPS 追踪

**RobotClient**：
- 连接策略服务器
- 流式发送观测
- 接收并队列动作块
- 控制循环中执行动作

### 7.4 数据集可视化

**文件**：`src/lerobot/scripts/lerobot_dataset_viz.py`

```bash
lerobot-dataset-viz \
    --dataset.repo_id=user/dataset \
    --episode_index=0 \
    --mode=local
```

**功能**：使用 Rerun 可视化 episode 的图像、动作、状态

**可复用价值**：高。评估和推理部署已完整实现。

---

## 8. 摄像头系统

### 8.1 相机抽象基类

**文件**：`src/lerobot/cameras/camera.py`

```python
class Camera(ABC):
    @abstractmethod
    def connect(self): ...
    @abstractmethod
    def read(self) -> np.ndarray: ...
    @abstractmethod
    def async_read(self) -> np.ndarray: ...
    @abstractmethod
    def disconnect(self): ...
```

### 8.2 RealSense 相机

**文件**：`src/lerobot/cameras/realsense/camera_realsense.py`

**类**：`RealSenseCamera`

**功能**：
- 序列号识别（避免插拔后设备号变化）
- RGB + Depth 流
- 异步读取（后台线程）
- Pipeline 配置、预热、后处理

**配置**：
```python
RealSenseCameraConfig(
    serial_number_or_name="327122072195",
    fps=30, width=640, height=480,
    color_mode=ColorMode.RGB,
    use_depth=False,
    warmup_s=1
)
```

### 8.3 OpenCV 相机

**文件**：`src/lerobot/cameras/opencv/camera_opencv.py`

**类**：`OpenCVCamera`

**功能**：
- 支持相机索引和设备路径
- 异步读取
- 旋转支持（0°, 90°, 180°, 270°）
- 多后端支持

**配置**：
```python
OpenCVCameraConfig(
    index_or_path="/dev/v4l/by-id/...",
    fps=30, width=640, height=480,
    rotation=Cv2Rotation.NO_ROTATION,
    fourcc="MJPG", warmup_s=0
)
```

### 8.4 相机工厂

**文件**：`src/lerobot/cameras/utils.py`

```python
def make_cameras_from_configs(configs: dict[str, CameraConfig]) -> dict[str, Camera]:
    # 根据配置字典创建相机实例
```

### 8.5 当前 XLerobot 相机配置

```python
def xlerobot_cameras_config() -> dict[str, CameraConfig]:
    return {
        "right": OpenCVCameraConfig(index_or_path="/dev/v4l/by-id/...", fps=30, ...),
        "left":  OpenCVCameraConfig(index_or_path="/dev/v4l/by-id/...", fps=30, ...),
        "head":  RealSenseCameraConfig(serial_number_or_name="327122072195", fps=30, ...),
    }
```

**可复用价值**：高。相机系统已完整抽象。

---

## 9. 处理器流水线

### 9.1 流水线架构

**文件**：`src/lerobot/processor/pipeline.py`

```python
class PolicyProcessorPipeline:
    """策略输入/输出处理链"""

class RobotProcessorPipeline:
    """机器人观测/动作处理链"""
```

### 9.2 处理器类型

| 处理器 | 文件 | 功能 |
|--------|------|------|
| Normalize | `normalize_processor.py` | 基于 stats 的 Z-score 归一化 |
| Device | `device_processor.py` | CPU/GPU 设备放置 |
| Observation | `observation_processor.py` | 观测数据格式化 |
| Delta Action | `delta_action_processor.py` | 绝对/增量动作转换 |
| Rename | `rename_processor.py` | 键名重映射 |
| Tokenizer | `tokenizer_processor.py` | 文本 Tokenize |
| Policy-Robot Bridge | `policy_robot_bridge.py` | 策略输出 → 机器人动作转换 |

### 9.3 默认处理器

**文件**：`src/lerobot/processor/factory.py`

```python
def make_default_processors():
    # 返回: teleop_action_processor, robot_action_processor, robot_observation_processor
    # 默认均为 IdentityProcessor（透传）
```

**可复用价值**：高。流水线架构支持灵活扩展。

---

## 10. 电机与运动控制

### 10.1 电机总线基类

**文件**：`src/lerobot/motors/motors_bus.py`

```python
class MotorsBus(ABC):
    def sync_read(self, register_name, motor_names): ...
    def sync_write(self, register_name, values): ...
    def read_calibration(self): ...
    def write_calibration(self, calibration): ...
    def record_ranges_of_motion(self, motor_names): ...
    def set_half_turn_homings(self, motor_names): ...
```

### 10.2 Feetech 电机

**文件**：`src/lerobot/motors/feetech/feetech.py`

**类**：`FeetechMotorsBus`

**支持型号**：STS3215（SO-101/SO-100 使用）

**功能**：
- 同步读写（减少通信延迟）
- 自动标定管理
- 位置/速度/电流模式
- PID 参数配置

### 10.3 SO101 运动学

**文件**：`src/lerobot/model/SO101Robot.py`

**类**：`SO101Kinematics`

```python
class SO101Kinematics:
    def forward_kinematics(self, joint_angles) -> (x, y, z):
        """正运动学：关节角 → 末端位置"""

    def inverse_kinematics(self, x, y) -> (joint2, joint3):
        """逆运动学：XY 平面位置 → shoulder_lift, elbow_flex"""

    def generate_trajectory(self, start, end, num_points):
        """轨迹生成"""
```

**可复用价值**：高。运动学已封装，IK 用于键盘/Joy-Con 末端执行器控制。

---

## 11. 可复用资产清单

| # | 资产名称 | 文件路径 | 复用方式 | 优先级 |
|---|---------|---------|---------|--------|
| 1 | ZMQ Host | `src/lerobot/robots/xlerobot/xlerobot_host.py` | 直接运行 | P0 |
| 2 | ZMQ Client | `src/lerobot/robots/xlerobot/xlerobot_client.py` | 直接使用 | P0 |
| 3 | 双主臂遥操器 | `src/lerobot/teleoperators/xlebi_so101_leader/` | 直接使用 | P0 |
| 4 | 键盘遥操 | `src/lerobot/teleoperators/keyboard/` | 直接使用 | P0 |
| 5 | 录制循环 | `src/lerobot/scripts/lerobot_record.py:record_loop()` | 复用或继承 | P0 |
| 6 | LeRobotDataset | `src/lerobot/datasets/lerobot_dataset.py` | 直接使用 | P0 |
| 7 | 键盘事件系统 | `src/lerobot/utils/control_utils.py:init_keyboard_listener()` | 直接使用 | P0 |
| 8 | rerun 可视化 | `src/lerobot/utils/visualization_utils.py` | 直接使用 | P1 |
| 9 | Joy-Con 控制逻辑 | `examples/xlerobot/7_xlerobot_teleop_joycon.py` | 提取封装 | P1 |
| 10 | SO101 运动学 | `src/lerobot/model/SO101Robot.py` | 直接使用 | P1 |
| 11 | 特征转换工具 | `src/lerobot/datasets/utils.py` | 直接使用 | P1 |
| 12 | 视频编码 | `src/lerobot/datasets/video_utils.py` | 直接使用 | P1 |
| 13 | 训练脚本 | `src/lerobot/scripts/lerobot_train.py` | 直接使用 | P1 |
| 14 | 评估脚本 | `src/lerobot/scripts/lerobot_eval.py` | 直接使用 | P1 |
| 15 | 重放脚本 | `src/lerobot/scripts/lerobot_replay.py` | 直接使用 | P1 |
| 16 | 异步推理 | `src/lerobot/async_inference/` | 直接使用 | P2 |
| 17 | 数据集工具 | `src/lerobot/datasets/dataset_tools.py` | 直接使用 | P2 |
| 18 | VR 遥操 | `src/lerobot/teleoperators/xlerobot_vr/` | 直接使用 | P2 |
| 19 | Gamepad 遥操 | `src/lerobot/teleoperators/gamepad/` | 直接使用 | P2 |
| 20 | 处理器流水线 | `src/lerobot/processor/` | 直接使用 | P2 |

---

## 12. 现有示例脚本汇总

### 12.1 XLerobot 专用示例

| 文件 | 功能 | 输入设备 | 远程 | 录制 |
|------|------|---------|------|------|
| `0_so100_keyboard_joint_control.py` | SO100 关节级键盘控制 | 键盘 | ❌ | ❌ |
| `1_so100_keyboard_ee_control.py` | SO100 EE 键盘控制 | 键盘+IK | ❌ | ❌ |
| `2_dual_so100_keyboard_ee_control.py` | 双臂 SO100 键盘 | 键盘+IK | ❌ | ❌ |
| `4_xlerobot_teleop_keyboard.py` | 完整 XLerobot 键盘 | 键盘 | ✅ ZMQ | ❌ |
| `4_xlerobot_2wheels_teleop_keyboard.py` | 两轮版键盘 | 键盘 | ✅ ZMQ | ❌ |
| `5_xlerobot_teleop_xbox.py` | Xbox 遥操 | Xbox | ❌ | ❌ |
| `6_so100_joycon_ee_control.py` | SO100 Joy-Con | Joy-Con | ❌ | ❌ |
| `7_xlerobot_teleop_joycon.py` | 完整 XLerobot Joy-Con | Joy-Con | ❌ | ❌ |
| `8_xlerobot_teleop_vr.py` | VR 遥操 | VR | ❌ | ❌ |
| `8_vr_teleop_with_dataset_recording.py` | VR 遥操+录制 | VR | ❌ | ✅ |
| `teleop_bimanual_zmq.py` | 双主臂+键盘 ZMQ | 主臂+键盘 | ✅ ZMQ | ❌ |
| `record_keyboard_teleop.py` | 键盘录制 | 键盘 | ✅ ZMQ | ✅ |
| `record_remote_bi_so101_leader_keyboard.py` | 主臂+键盘录制 | 主臂+键盘 | ✅ ZMQ | ✅ |

### 12.2 其他机器人示例

| 目录 | 功能 |
|------|------|
| `examples/lekiwi/` | LeKiwi 遥操/录制/重放/评估 |
| `examples/so100_to_so100_EE/` | SO100 主臂→从臂 EE 空间映射 |
| `examples/phone_to_so100/` | 手机 → SO100 遥操 |
| `examples/omni_base/` | 全向底盘键盘遥操 |
| `examples/training/` | 训练示例 |
| `examples/tutorial/` | ACT/Diffusion/PI0 教程 |

---

## 13. 关键结论与建议

### 13.1 已有能力总结

| 模块 | 完成度 | 说明 |
|------|--------|------|
| ZMQ 远程通信 | 95% | Host/Client 已完整，仅需优化图像编码 |
| 主臂遥操 | 100% | `XleBiSO101Leader` 已封装 |
| 键盘遥操 | 100% | 支持双臂+头+底盘 |
| Joy-Con 遥操 | 80% | 功能完整但未封装为 Teleoperator |
| 数据采集 | 95% | `lerobot_record.py` 已封装 |
| 数据集格式 | 100% | v3.0 格式完整 |
| 模型训练 | 95% | 12+ 策略，训练脚本完整 |
| 模型评估 | 90% | eval/replay/async 推理 |
| 可视化 | 90% | rerun 支持实时和数据集可视化 |

### 13.2 待建设能力

1. **Joy-Con + ZMQ 远程遥操脚本**：将 `7_xlerobot_teleop_joycon.py` 改造为 ZMQ 远程版
2. **混合遥操脚本**：主臂(双臂) + Joy-Con(头+底盘) + ZMQ 远程
3. **数据采集增强**：episode 标记、多任务支持、自动上传
4. **双臂训练配置**：纯双臂动作（不含底盘）
5. **双臂移动训练配置**：完整动作空间（arm + head + base）

### 13.3 网络优化建议

| 优化项 | 当前状态 | 建议 |
|--------|---------|------|
| 图像编码 | JPEG base64 | 考虑 H.264 视频流降低带宽 |
| 传输协议 | ZMQ PUSH/PULL | 当前足够，可考虑 gRPC 替代 |
| 压缩率 | ~50-150KB/帧 | JPEG quality 可调（当前 90） |
| 帧率 | 30fps | 可根据网络降频至 15-20fps |

### 13.4 推荐实施路径

```
Phase 1: 基础设施
  ├─ Orin 部署 Host（已存在）
  └─ PC 端 Client 测试连通性

Phase 2: 遥操系统
  ├─ 方案 A: 主臂+键盘（已存在 teleop_bimanual_zmq.py）
  ├─ 方案 B: Joy-Con 远程遥操（需开发）
  └─ 方案 C: 键盘 IK 遥操（已存在 record_keyboard_teleop.py）

Phase 3: 数据采集
  ├─ 录制脚本定制
  ├─ 数据集管理工具
  └─ 自动上传 Hub

Phase 4: 模型训练
  ├─ 双臂任务训练配置
  ├─ 双臂移动任务训练配置
  └─ 训练监控（wandb）

Phase 5: 验证部署
  ├─ 评估脚本
  ├─ 策略重放
  └─ 异步推理部署
```

---

## 附录 A：快速启动命令

### Orin 端启动 Host

```bash
PYTHONPATH=src python -m lerobot.robots.xlerobot.xlerobot_host --robot.id=my_xlerobot
```

### PC 端启动双主臂遥操

```bash
PYTHONPATH=src python examples/xlerobot/teleop_bimanual_zmq.py \
    --remote_ip=10.42.0.192 \
    --left_arm_port=/dev/serial/by-id/usb-... \
    --right_arm_port=/dev/serial/by-id/usb-... \
    --camera_names=left,right,head
```

### 录制数据集

```bash
PYTHONPATH=src python examples/xlerobot/record_keyboard_teleop.py \
    --robot_id=my_xlerobot \
    --remote_ip=10.42.0.192 \
    --repo_id=user/xlerobot_dataset \
    --num_episodes=50 \
    --task_description="Pick and place task"
```

### 训练

```bash
lerobot-train \
    --dataset.repo_id=user/xlerobot_dataset \
    --policy.type=act \
    --output_dir=outputs/train/xlerobot_act
```

### 评估

```bash
lerobot-eval \
    --robot.type=xlerobot_client \
    --robot.remote_ip=10.42.0.192 \
    --policy.path=outputs/train/xlerobot_act/checkpoint/last/pretrained_model \
    --eval.n_episodes=10
```

---

*文档结束*
