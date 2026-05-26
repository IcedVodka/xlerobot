# XLRobot Orin + PC 键盘遥操作配置与测试指南

> 适用版本：3轮全向底盘版 XLRobot  
> 通信方式：WiFi 无线 + ZMQ  
> 控制端：PC 键盘  
> 机器人端：NVIDIA Jetson Orin

---

## 目录

- [1. 概述](#1-概述)
- [2. 硬件准备清单](#2-硬件准备清单)
- [3. Orin 端配置](#3-orin-端配置)
- [4. PC 端配置](#4-pc-端配置)
- [5. 网络配置](#5-网络配置)
- [6. 硬件校准](#6-硬件校准)
- [7. 运行键盘遥操作](#7-运行键盘遥操作)
- [8. 键盘控制映射表](#8-键盘控制映射表)
- [9. 摄像头配置（可选）](#9-摄像头配置可选)
- [10. 数据集录制（可选）](#10-数据集录制可选)
- [11. 常见问题排查](#11-常见问题排查)

---

## 1. 概述

本指南介绍如何使用 **NVIDIA Jetson Orin** 作为 XLRobot 的控制主机，通过 **WiFi** 与 PC 建立 ZMQ 通信，并在 PC 端使用 **键盘** 对机器人进行远程遥操作。

### 系统架构

```
┌─────────────────┐         WiFi (ZMQ)          ┌─────────────────────────┐
│   PC 控制端     │  ◄─────5556 观测数据─────►  │   NVIDIA Jetson Orin    │
│  (键盘遥操作)   │  ─────►5555 控制指令─────►  │  (ZMQ Host + 硬件驱动)   │
└─────────────────┘                               └─────────────────────────┘
                                                          │
                                    ┌─────────────────────┼─────────────────────┐
                                    │                     │                     │
                                  Bus1                  Bus2                  摄像头
                                    │                     │
                              ┌──────────┐          ┌──────────┐
                              │ 左臂(6)   │          │ 右臂(6)   │
                              │ 头部(2)   │          │ 底盘(3)   │
                              └──────────┘          └──────────┘
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| ZMQ 命令端口 | `5555` | PC 发送控制指令到 Orin |
| ZMQ 观测端口 | `5556` | Orin 发送观测数据到 PC |
| 看门狗超时 | `500ms` | 超过此时间无指令，底盘自动停止 |
| 控制频率 | `30Hz` | Host 端主循环频率 |
| 连接时长 | `3600s` | 默认自动断开时间 |

---

## 2. 硬件准备清单

### 必备硬件

| 序号 | 硬件 | 数量 | 说明 |
|------|------|------|------|
| 1 | NVIDIA Jetson Orin | 1 | 机器人控制主机，运行 Ubuntu |
| 2 | PC/笔记本 | 1 | 运行键盘遥操作代码 |
| 3 | XLRobot 机器人本体 | 1 | 含双臂(SO101)、头部、3轮全向底盘 |
| 4 | Feetech STS3215 电机 | 17 | 左臂6 + 右臂6 + 头部2 + 底盘3 |
| 5 | USB 数据线 | 2 | 连接 Orin 与两个电机总线控制器 |
| 6 | 路由器/WiFi 热点 | 1 | Orin 和 PC 处于同一局域网 |
| 7 | 12V 电源 | 1 | 为电机总线供电 |

### 硬件连接图

```
Orin (USB-A) ───────► 总线控制器 1 (ttyACM0) ───► 左臂电机(1-6) + 头部电机(7-8)
Orin (USB-A) ───────► 总线控制器 2 (ttyACM1) ───► 右臂电机(1-6) + 底盘电机(7-9)
```

> **注意**：总线控制器 1 和 2 各自独立，电机 ID 在每个总线内部从 1 开始编号。

---

## 3. Orin 端配置

### 3.1 系统环境检查

确保 Orin 上已安装：

```bash
# 检查 Python 版本（需 >= 3.10）
python3 --version

# 检查 pip
pip3 --version

# 检查 USB 串口设备（连接好硬件后）
ls /dev/ttyACM*
# 预期输出：/dev/ttyACM0  /dev/ttyACM1
```

### 3.2 串口权限设置

将当前用户加入 `dialout` 组，避免每次使用 `sudo`：

```bash
sudo usermod -a -G dialout $USER
# 重新登录使权限生效
```

创建 udev 规则确保串口设备名稳定：

```bash
# 查看设备 ID（连接硬件后执行）
udevadm info -a -n /dev/ttyACM0 | grep -E "idVendor|idProduct|serial"
udevadm info -a -n /dev/ttyACM1 | grep -E "idVendor|idProduct|serial"
```

### 3.3 安装 LeRobot

```bash
# 克隆代码仓库（如果还没有）
git clone <your-repo-url> ~/lerobot
cd ~/lerobot

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate

# 安装 LeRobot 及所需依赖
pip install -e ".[feetech]"

# 安装 ZMQ 通信库
pip install pyzmq>=26.2.1

# 安装 OpenCV（用于图像编码传输）
pip install opencv-python-headless
```

### 3.4 验证硬件连接

```bash
cd ~/lerobot
source venv/bin/activate

# 查找串口
python -m lerobot.scripts.lerobot_find_port

# 如果找到端口，测试连接
python -c "
from lerobot.robots.xlerobot import XLerobotConfig, XLerobot
config = XLerobotConfig(id='test_robot')
robot = XLerobot(config)
print('Robot created successfully')
print('Bus1 motors:', list(robot.bus1.motors.keys()))
print('Bus2 motors:', list(robot.bus2.motors.keys()))
"
```

---

## 4. PC 端配置

### 4.1 环境准备

PC 端不需要直接连接硬件，只需要运行遥操作客户端代码。

```bash
# 克隆同样的代码仓库
git clone <your-repo-url> ~/lerobot
cd ~/lerobot

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装 LeRobot 基础依赖（不需要 feetech 电机驱动）
pip install -e ".[feetech]"

# 安装 ZMQ
pip install pyzmq>=26.2.1

# 安装键盘监听库（pynput 已在基础依赖中包含）
# pip install pynput

# 安装 rerun-sdk 用于可视化（可选）
pip install rerun-sdk>=0.24.0
```

### 4.2 验证安装

```bash
cd ~/lerobot
source venv/bin/activate

# 测试键盘模块能否正常导入
python -c "
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient, XLerobotClientConfig
print('All imports successful!')
"
```

---

## 5. 网络配置

### 5.1 确认 Orin IP 地址

在 Orin 终端执行：

```bash
# 查看 WiFi 接口 IP
ip addr show wlan0
# 或查看所有接口
hostname -I
```

假设 Orin 的 IP 地址为：`192.168.1.100`（请替换为你的实际 IP）。

### 5.2 确认 PC 与 Orin 连通

在 PC 端执行：

```bash
ping 192.168.1.100
```

确保能正常收到回复。

### 5.3 防火墙端口开放

如果 Orin 有防火墙限制，需要开放 ZMQ 端口：

```bash
# 在 Orin 上执行
sudo ufw allow 5555/tcp
sudo ufw allow 5556/tcp
sudo ufw reload
```

> **提示**：如果 Orin 和 PC 连接在同一个路由器下但 ZMQ 仍无法连接，检查路由器是否开启了 AP 隔离。

---

## 6. 硬件校准

### 6.1 首次校准（必须在 Orin 上执行）

第一次使用或更换电机后，需要进行校准。校准文件会自动保存，下次启动时会提示是否恢复。

```bash
cd ~/lerobot
source venv/bin/activate
export PYTHONPATH=src

# 运行校准（直接连接方式）
python -c "
from lerobot.robots.xlerobot import XLerobotConfig, XLerobot
config = XLerobotConfig(id='my_xlerobot')
robot = XLerobot(config)
robot.connect()  # 会自动触发校准流程
"
```

### 6.2 校准流程说明

连接后会出现交互式提示：

1. **恢复校准**：如果检测到已有校准文件，会提示 `Press ENTER to restore calibration from file, or type 'c' and press ENTER to run manual calibration:`
   - 直接按 **Enter** 恢复已有校准
   - 输入 **c** 重新手动校准

2. **手动校准步骤**：
   - 按提示将 **左臂和头部** 电机移动到行程中间位置，按 Enter
   - 依次将左臂和头部各关节移动到最大/最小位置，按 Enter 停止记录
   - 按提示将 **右臂** 电机移动到行程中间位置，按 Enter
   - 依次将右臂各关节移动到最大/最小位置，按 Enter 停止记录
   - 校准数据自动保存到配置文件目录

3. **校准结果保存位置**：
   ```
   ~/.cache/lerobot/my_xlerobot/calibration.json
   ```

### 6.3 验证校准

校准完成后，电机应能正常响应位置指令，且关节角度值在合理范围内。

---

## 7. 运行键盘遥操作

### 7.1 步骤 1：Orin 端启动 Host

在 **Orin** 上打开终端，执行：

```bash
cd ~/lerobot
source venv/bin/activate
export PYTHONPATH=src

# 启动 ZMQ Host
python -m lerobot.robots.xlerobot.xlerobot_host
```

看到以下输出说明 Host 启动成功：

```
Configuring Xlerobot
Connecting Xlerobot
Starting HostAgent
Waiting for commands...
```

> **Host 默认配置**：
> - 命令端口：`5555`
> - 观测端口：`5556`
> - 运行时长：`3600` 秒（1小时）
> - 看门狗超时：`500ms`

Orin 端启动后，会等待 PC 端 Client 连接。

### 7.2 步骤 2：PC 端启动键盘遥操作

在 **PC** 上打开终端，执行：

```bash
cd ~/lerobot
source venv/bin/activate
export PYTHONPATH=src

# 方式 1：直接运行示例脚本（需要修改脚本中的 IP 地址）
# 先编辑 examples/xlerobot/4_xlerobot_teleop_keyboard.py
# 将 ip = "localhost" 改为 Orin 的实际 IP

# 然后运行
python examples/xlerobot/4_xlerobot_teleop_keyboard.py --robot_id=my_xlerobot
```

#### 修改示例脚本中的 IP

编辑 `examples/xlerobot/4_xlerobot_teleop_keyboard.py`，找到以下代码：

```python
# 第 388 行附近
ip = "localhost"  # 改为 Orin 的 IP
# 改为：
ip = "192.168.1.100"  # Orin 的实际 IP

# 同时取消 ZMQ 连接的注释，注释掉本地连接
# For zmq connection
robot_config = XLerobotClientConfig(remote_ip=ip, id=robot_name)
robot = XLerobotClient(robot_config)

# For local/wired connection
# robot_config = XLerobotConfig(id=robot_id)
# robot = XLerobot(robot_config)
```

### 7.3 连接验证

PC 端成功连接后，会看到：

```
[MAIN] Successfully connected to robot
```

此时 PC 端会显示键盘控制映射信息，表示可以开始控制。

### 7.4 退出控制

- 按键盘上的 `b` 键退出遥操作
- 或在终端按 `Ctrl+C` 强制退出

---

## 8. 键盘控制映射表

### 8.1 底盘移动控制（3轮全向版）

| 按键 | 功能 | 说明 |
|------|------|------|
| `i` | 前进 | 向前移动（+X 方向） |
| `k` | 后退 | 向后移动（-X 方向） |
| `j` | 左移 | 向左平移（+Y 方向，全向轮特有） |
| `l` | 右移 | 向右平移（-Y 方向，全向轮特有） |
| `u` | 左转 | 原地逆时针旋转 |
| `o` | 右转 | 原地顺时针旋转 |
| `n` | 加速 | 提升一档速度 |
| `m` | 减速 | 降低一档速度 |
| `b` | 退出 | 停止并退出遥操作 |

### 8.2 速度档位

| 档位 | 线速度 (m/s) | 角速度 (deg/s) |
|------|-------------|---------------|
| 1档（慢） | 0.1 | 30 |
| 2档（中） | 0.2 | 60 |
| 3档（快） | 0.3 | 90 |

> 默认启动为 1 档，按 `n` 升档，`m` 降档。

### 8.3 左臂控制

| 按键 | 功能 | 关节 |
|------|------|------|
| `q` / `e` | 肩平移 +/- | shoulder_pan |
| `w` / `s` | X 轴 +/- | 末端前后移动（逆运动学） |
| `a` / `d` | Y 轴 +/- | 末端左右移动（逆运动学） |
| `z` / `x` | 俯仰 +/- | pitch |
| `r` / `f` | 腕滚转 +/- | wrist_roll |
| `t` / `g` | 夹爪 开/合 | gripper |
| `c` | 复位 | 回到零位 |
| `y` | 矩形轨迹 | 执行预设矩形运动 |

### 8.4 右臂控制

| 按键 | 功能 | 关节 |
|------|------|------|
| `7` / `9` | 肩平移 +/- | shoulder_pan |
| `8` / `2` | X 轴 +/- | 末端前后移动 |
| `4` / `6` | Y 轴 +/- | 末端左右移动 |
| `1` / `3` | 俯仰 +/- | pitch |
| `/` / `*` | 腕滚转 +/- | wrist_roll |
| `+` / `-` | 夹爪 开/合 | gripper |
| `0` | 复位 | 回到零位 |
| `Y` | 矩形轨迹 | 执行预设矩形运动 |

### 8.5 头部控制

| 按键 | 功能 |
|------|------|
| `<` / `>` | 头部电机 1 +/- |
| `,` / `.` | 头部电机 2 +/- |
| `?` | 头部复位到零位 |

---

## 9. 摄像头配置（可选）

### 9.1 Orin 端摄像头配置

编辑 `src/lerobot/robots/xlerobot/config_xlerobot.py`，取消相应摄像头的注释：

#### 使用 USB 摄像头

```python
def xlerobot_cameras_config() -> dict[str, CameraConfig]:
    return {
        "left_wrist": OpenCVCameraConfig(
            index_or_path="/dev/video0", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        ),
        "right_wrist": OpenCVCameraConfig(
            index_or_path="/dev/video2", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        ),
    }
```

#### 使用 Intel RealSense 摄像头

```python
def xlerobot_cameras_config() -> dict[str, CameraConfig]:
    return {
        "head": RealSenseCameraConfig(
            serial_number_or_name="125322060037",  # 替换为实际序列号
            fps=30,
            width=1280,
            height=720,
            color_mode=ColorMode.BGR,
            rotation=Cv2Rotation.NO_ROTATION,
            use_depth=True
        ),
    }
```

### 9.2 查找摄像头设备

在 Orin 上执行：

```bash
# 列出所有视频设备
ls /dev/video*

# 查看 RealSense 序列号
rs-enumerate-devices | grep "Serial Number"
```

### 9.3 摄像头图像查看

PC 端连接后，观测数据中会自动包含摄像头图像（通过 ZMQ 以 JPEG base64 编码传输）。

如果安装了 rerun-sdk，会在可视化界面中看到摄像头画面。

---

## 10. 数据集录制（可选）

### 10.1 仅键盘控制录制

使用纯键盘控制录制数据集（不需要主控臂）：

```bash
cd ~/lerobot
source venv/bin/activate
export PYTHONPATH=src

# 在 PC 端运行录制脚本
python examples/xlerobot/record_keyboard_teleop.py \
    --robot_id=my_xlerobot \
    --remote_ip=192.168.1.100 \
    --repo_id=<your_hf_username>/<dataset_name> \
    --fps=30 \
    --num_episodes=50
```

### 10.2 使用 BiSO101Leader 主控臂 + 键盘录制

如果有 BiSO101Leader 双臂主控，可以配合键盘底盘控制进行录制：

```bash
# 在 PC 端运行（需要连接 BiSO101Leader 硬件）
python examples/xlerobot/record_remote_bi_so101_leader_keyboard.py \
    --robot_id=joyandai_xlerobot \
    --remote_ip=192.168.1.100 \
    --leader_id=my_bi_so101_leader \
    --left_leader_port=/dev/ttyUSB0 \
    --right_leader_port=/dev/ttyUSB1 \
    --repo_id=<your_hf_username>/<dataset_name> \
    --fps=30 \
    --num_episodes=50
```

> **注意**：`left_leader_port` 和 `right_leader_port` 需要根据实际串口修改。Windows 下为 `COM8`、`COM9` 等。

### 10.3 录制控制按键

| 按键 | 功能 |
|------|------|
| `→` (右箭头) | 结束当前 episode |
| `←` (左箭头) | 重录当前 episode |
| `Esc` | 停止录制 |
| `b` | 舍弃当前 episode 并退出 |

### 10.4 录制参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--fps` | 30 | 录制帧率 |
| `--episode_time_s` | 300 | 每轮录制时长（秒） |
| `--reset_time_s` | 10 | 每轮间隔重置时间（秒） |
| `--num_episodes` | 50 | 总录制轮数 |
| `--task_description` | "My task description" | 任务描述 |
| `--resume` | False | 继续已有数据集录制 |
| `--display_data` | False | 显示可视化数据 |

---

## 11. 常见问题排查

### 11.1 串口设备找不到

**现象**：`/dev/ttyACM0` 或 `/dev/ttyACM1` 不存在

**排查**：
```bash
# 查看 USB 设备连接
lsusb

# 查看内核日志
dmesg | tail -50

# 检查设备权限
ls -la /dev/ttyACM*
```

**解决**：
- 确认 USB 线已插好
- 确认电机总线控制器已供电
- 将用户加入 `dialout` 组：`sudo usermod -a -G dialout $USER`
- 重新插拔 USB，或重启 Orin

### 11.2 ZMQ 连接失败

**现象**：PC 端提示 `Timeout waiting for LeKiwi Host to connect expired.`

**排查**：
```bash
# 在 PC 端测试网络连通性
ping <orin_ip>

# 在 Orin 端检查端口监听
netstat -tlnp | grep 5555
netstat -tlnp | grep 5556

# 在 Orin 端检查防火墙
sudo ufw status
```

**解决**：
- 确认 Orin 和 PC 在同一局域网
- 确认 Orin 的 Host 已先启动
- 开放防火墙端口：`sudo ufw allow 5555/tcp && sudo ufw allow 5556/tcp`
- 检查路由器 AP 隔离设置

### 11.3 底盘突然停止

**现象**：底盘在遥控过程中突然停止

**原因**：ZMQ 看门狗机制。如果超过 500ms 未收到控制指令，底盘会自动停止。

**解决**：
- 确保 PC 端控制循环正常运行
- 检查 WiFi 信号是否稳定
- 如果网络延迟大，可在 `config_xlerobot.py` 中增大 `watchdog_timeout_ms`

### 11.4 电机抖动

**现象**：手臂或底盘电机运行时抖动

**解决**：
- 降低 Host 端控制频率：修改 `XLerobotHostConfig` 中的 `max_loop_freq_hz`（如从 30 降到 15-20）
- 在 Orin 上运行 `top` 检查 CPU 负载
- 检查 PID 参数：P_Coefficient 默认为 16，可根据需要调整

### 11.5 校准丢失

**现象**：重启后电机位置不对

**解决**：
- 校准文件保存在 `~/.cache/lerobot/<robot_id>/calibration.json`
- 确保该文件存在且有读写权限
- 如校准异常，删除文件后重新校准

### 11.6 图像传输卡顿

**现象**：PC 端摄像头画面延迟或卡顿

**解决**：
- 降低摄像头分辨率或帧率
- 检查 WiFi 带宽
- 在配置中减少同时开启的摄像头数量

---

## 附录 A：快速启动命令参考

### Orin 端（每次启动）

```bash
cd ~/lerobot
source venv/bin/activate
export PYTHONPATH=src
python -m lerobot.robots.xlerobot.xlerobot_host
```

### PC 端（每次启动）

```bash
cd ~/lerobot
source venv/bin/activate
export PYTHONPATH=src
python examples/xlerobot/4_xlerobot_teleop_keyboard.py --robot_id=my_xlerobot
```

## 附录 B：配置文件速查

| 配置项 | 文件路径 | 说明 |
|--------|----------|------|
| 串口端口 | `src/lerobot/robots/xlerobot/config_xlerobot.py` | `port1`, `port2` |
| ZMQ 端口 | `src/lerobot/robots/xlerobot/config_xlerobot.py` | `port_zmq_cmd=5555`, `port_zmq_observations=5556` |
| 键盘映射 | `src/lerobot/robots/xlerobot/config_xlerobot.py` | `teleop_keys` |
| 摄像头 | `src/lerobot/robots/xlerobot/config_xlerobot.py` | `cameras` |
| 速度档位 | `src/lerobot/robots/xlerobot/xlerobot.py` | `speed_levels` |

---

> 如有其他问题，请查看 [LeRobot 官方文档](https://huggingface.co/docs/lerobot/index) 或在项目中提交 Issue。
