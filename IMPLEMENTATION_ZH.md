# joycon-robotics 实现原理详解

> 本文基于代码仓库当前版本（commit `ea3ca0b` 附近）逐层拆解 `joycon-robotics` 的设计思路、核心算法与机器人遥操作实现。

---

## 1. 与 joycon-python 的关系：它是 fork/派生项目吗？

**结论：是的，joycon-robotics 是在 `joycon-python` 基础上深度派生、面向机器人遥操作二次开发的。**

最直接的几条证据：

1. **文件结构高度重合**
   `joyconrobotics/` 目录下的 `joycon.py`、`wrappers.py`、`gyro.py`、`event.py`、`device.py`、`constants.py` 与 `joycon-python` 的 `pyjoycon/` 同名文件在类名、方法名、注释、TODO 标记上几乎完全一致。

2. **setup.py 的框架同源**
   - `joycon-python` 版本为 `0.2.4`，`joycon-robotics` 的 `__init__.py` 里也保留了 `__version__ = "0.2.4"`。
   - 两者 setup.py 的作者/URL 字段格式、classifiers 几乎一致。

3. **核心 HID 解析逻辑完全一致**
   例如 49 字节输入报告解析、SPI Flash 读取 IMU 标定参数、`0x30` 报告格式下加速度/陀螺仪字节偏移量、`set_accel_calibration` / `set_gyro_calibration` 的 `0x4000 / cx`、`0x343b / cx` 标定公式，均与 joycon-python 相同。

4. **新增的机器人层是主要增量**
   joycon-robotics 在 joycon-python 提供的“读取 Joy-Con 传感器与按键”能力之上，新增了 `joyconrobotics.py`，其中实现 `AttitudeEstimator` 和 `JoyconRobotics`，这是本项目区别于原项目的核心。

---

## 2. 与 joycon-python 的主要区别

| 维度 | joycon-python | joycon-robotics |
|---|---|---|
| **定位** | 通用 Nintendo Switch Joy-Con Python 驱动 | 低成本机器人遥操作手柄框架 |
| **顶层接口** | `JoyCon` / `PythonicJoyCon` / `GyroTrackingJoyCon` / `ButtonEventJoyCon` | `JoyconRobotics`（内部仍复用上述类） |
| **姿态解算** | 仅提供 `GyroTrackingJoyCon` 纯陀螺仪积分 | 新增互补滤波 `AttitudeEstimator`，融合加速度 + 陀螺仪 |
| **机器人控制** | 无 | 位置、姿态、夹爪、按键事件一体化输出 |
| **硬件兼容性** | 偏向标准 Nintendo Joy-Con | 支持自家盒桥智能手柄，也 patch 支持普通 Joy-Con |
| **系统驱动** | 纯 Python HID | 额外包含 `system_lib/joycond`、`udev`、`hidapi_for_windows` |
| **坐标映射** | 无 | 提供 `offset_position_m`、`euler_reverse`、`direction_reverse` 等参数 |

具体 patch 点（相对 joycon-python 的改动）：

- `joycon.py`：
  - 增加 `self.enable` 标志，`_close()` 中置 `False`，守护线程 `_update_input_report` 据此退出。
  - 增加 `calibrate_value` 条件：当 `serial[:9]` 不是自家序列号时，才启用系数缩放（兼容普通 Joy-Con）。
  - 移除 `serial[:9] not in JOYCON_SERIAL_HEAD` 的抛错，改为仅设置 `calibrate_value`。
  - `_spi_flash_read` 中不再严格校验 `report[:2] == b'\x90\x10'`。
- `device.py`：
  - 注释掉 `serial[0:6] != '9c:54:'` 过滤，支持普通 Nintendo Joy-Con。
- `constants.py`：新增 `JOYCON_SERIAL_SUPPORT`、`JOYCON_SERIAL_HEAD_L/R` 等自家手柄标识常量。

---

## 3. 总体架构

```text
┌─────────────────────────────────────────────────────────────┐
│                       JoyconRobotics                         │
│  机器人遥操作封装：姿态 + 位置 + 夹爪 + 按键事件              │
├─────────────────────────────────────────────────────────────┤
│  AttitudeEstimator（互补滤波姿态估计）                        │
│  GyroTrackingJoyCon（纯陀螺仪积分，提供姿态参考）              │
│  ButtonEventJoyCon（按键边沿事件）                            │
├─────────────────────────────────────────────────────────────┤
│  PythonicJoyCon（传感器数值封装：accel_in_g / gyro_in_rad）   │
├─────────────────────────────────────────────────────────────┤
│  JoyCon（HID 通信 + 49B 输入报告解析 + SPI 标定参数读取）      │
├─────────────────────────────────────────────────────────────┤
│  hid / hidapi（系统 HID 层） + joycond（内核驱动）            │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 底层通信：JoyCon 类

### 4.1 HID 输入报告

- 报告长度：`49` 字节。
- 报告类型：`0x30`（标准输入报告，60 Hz）。
- 每份报告包含 **3 组 IMU 样本**，每组 12 字节（加速度 6B + 陀螺仪 6B），等效 IMU 采样率约 180 Hz。

输入报告布局（关键字段）：

| 字节偏移 | 内容 |
|---|---|
| 0 | 报告类型 `0x30` |
| 2 | 电池、充电状态 |
| 3-5 | 按键位图 |
| 6-11 | 左摇杆 |
| 9-11（重叠） | 右摇杆 |
| 13 + i*12 | 第 i 组 accel x |
| 15 + i*12 | 第 i 组 accel y |
| 17 + i*12 | 第 i 组 accel z |
| 19 + i*12 | 第 i 组 gyro x |
| 21 + i*12 | 第 i 组 gyro y |
| 23 + i*12 | 第 i 组 gyro z |

### 4.2 IMU 标定

初始化时从 SPI Flash 读取 24 字节 IMU 标定数据：

- 若 `0x8026` 处为 `b"\xB2\xA1"`，读取用户标定 `0x8028`；
- 否则读取工厂标定 `0x6020`。

24 字节含义：

| 字节 | 含义 |
|---|---|
| 0-5 | 加速度计零偏 offset xyz |
| 6-11 | 加速度计缩放系数 coeff xyz |
| 12-17 | 陀螺仪零偏 offset xyz |
| 18-23 | 陀螺仪缩放系数 coeff xyz |

标定公式：

```python
accel_raw = (int16 - ACCEL_OFFSET_X) * ACCEL_COEFF_X
```

其中：

```python
ACCEL_COEFF_X = 0x4000 / cx if cx != 0x4000 and cx != 0 and calibrate_value else 1
GYRO_COEFF_X  = 0x343b / cx if cx != 0x343b and cx != 0 and calibrate_value else 1
```

`calibrate_value` 是 joycon-robotics 新增的判断：普通 Nintendo Joy-Con 不启用系数缩放（避免自家手柄系数套用错）。

---

## 5. 数值封装：PythonicJoyCon 类

在 `JoyCon` 原始整数读取之上，`PythonicJoyCon` 提供更友好的属性接口。

### 5.1 加速度

```python
@property
def accel_in_g(self):
    c = 4.0 / 0x4000          # ≈ 0.000061035
    c2 = c * self._ime_yz_coeff
    return [
        (self.get_accel_x(i) * c,
         self.get_accel_y(i) * c2,
         self.get_accel_z(i) * c2)
        for i in range(3)
    ]
```

结果单位：**g**（重力加速度）。

### 5.2 陀螺仪

```python
@property
def gyro_in_rad(self):
    c = 0.0001694 * 3.1415926536
    c2 = c * self._ime_yz_coeff
    return [...]
```

结果单位：**rad/s**。

### 5.3 左右手柄坐标统一

左手柄默认对 y/z 轴取反，使左右手柄方向系统一致：

```python
self._ime_yz_coeff = -1 if invert_left_ime_yz and self.is_left() else 1
```

---

## 6. 纯陀螺仪姿态跟踪：GyroTrackingJoyCon 类

该类维护一个随时间更新的方向四元数。

### 6.1 状态

```python
self.direction_X = vec3(1, 0, 0)   # 当前局部 X 轴在世界系的方向
self.direction_Y = vec3(0, 1, 0)
self.direction_Z = vec3(0, 0, 1)
self.direction_Q = quat()          # 累积旋转四元数
```

### 6.2 更新方式

每次新数据到达，`_gyro_update_hook` 被调用：

```python
for gx, gy, gz in self.gyro_in_rad:
    rotation = (
        angleAxis(gx * (-1/86), self.direction_X) *
        angleAxis(gy * (-1/86), self.direction_Y) *
        angleAxis(gz * (-1/86), self.direction_Z)
    )
    self.direction_X *= rotation
    self.direction_Y *= rotation
    self.direction_Z *= rotation
    self.direction_Q *= rotation
```

这里用的是 **axis-angle 四元数积分**（指数映射）。`-1/86` 是经验系数，作者注释也说明未完全明确其物理来源。

### 6.3 零偏校准

`calibrate(seconds=2)` 会在指定时间内累积陀螺仪读数，平均后作为新的零偏。

> 注意：纯陀螺积分会随时间漂移，因此该类主要作为姿态参考，不是最终机器人输出。

---

## 7. 真正的机器人姿态估计：AttitudeEstimator 类

这是 joycon-robotics 的核心新增类，输出最终用于机器人的 roll / pitch / yaw。

### 7.1 设计思路

- **roll、pitch**：加速度计提供静态倾角观测，陀螺仪提供动态更新，二者通过 **互补滤波** 融合。
- **yaw**：加速度计无法观测 yaw，因此只通过陀螺仪 z 轴积分得到。yaw 会漂移，由外部 `yaw_diff` 或 Home/Capture 复位修正。
- **平滑**：对最终 roll/pitch 做一阶低通滤波。

### 7.2 互补滤波

```python
# 1) 加速度计观测
roll_acc  = math.atan2(ay, -az)
pitch_acc = math.atan2(ax, math.sqrt(ay**2 + az**2))

# 2) 陀螺仪积分
gx, gy, gz = gyro_in_rad
self.pitch += gy * self.dt
self.roll  -= gx * self.dt

# 3) 互补滤波融合（alpha = 0.55）
self.pitch = self.alpha * self.pitch + (1 - self.alpha) * pitch_acc
self.roll  = self.alpha * self.roll  + (1 - self.alpha) * roll_acc

# 4) 低通滤波
self.pitch = self.lpf_pitch.update(self.pitch)
self.roll  = self.lpf_roll.update(self.roll)
```

- `alpha = 0.55` 表示更信任陀螺仪积分，但加速度计提供长期零漂修正。
- `dt = 0.01`，对应 100 Hz 主循环。

### 7.3 yaw 积分

```python
rotation = (
    angleAxis(gx * (-1/86), self.direction_X) *
    angleAxis(gy * (-1/86), self.direction_Y) *
    angleAxis(gz * (-1/86), self.direction_Z)
)
self.direction_X *= rotation
self.direction_Y *= rotation
self.direction_Z *= rotation
self.direction_Q *= rotation

self.yaw = self.direction_X[1]   # 取旋转后 X 轴在世界系的 y 分量
```

### 7.4 角度映射与限幅

```python
if self.common_rad:
    self.roll  = self.roll  * math.pi / 1.5
    self.pitch = self.pitch * math.pi / 1.5
    self.yaw   = -self.yaw  * math.pi / 1.5
```

`math.pi / 1.5` 是经验映射系数，将传感器角度压缩到机器人关节空间常用范围（约 ±1.05 rad）。

随后减去 `yaw_diff`（可用摇杆手动修正），并做阈值限幅：

```python
self.yaw = self.yaw - self.yaw_diff

if self.pitch_rad_T != -1:
    self.pitch = clamp(self.pitch, -T, T)
```

`pitch_down_double` 参数会让俯仰负方向（向下）灵敏度加倍：

```python
if self.pitch_down_double:
    self.pitch = self.pitch * 3.0 if self.pitch < 0 else self.pitch
```

---

## 8. 机器人遥操作封装：JoyconRobotics 类

这是用户直接交互的类，把 Joy-Con 输入映射为机器人位姿 `[x, y, z, roll, pitch, yaw]`、夹爪状态、按键控制码。

### 8.1 初始化流程

```python
self.joycon      = JoyCon(*self.joycon_id)
self.gyro        = GyroTrackingJoyCon(*self.joycon_id)
self.orientation_sensor = AttitudeEstimator(...)
self.button      = ButtonEventJoyCon(*self.joycon_id, track_sticks=True)
self.reset_joycon()
self.thread      = threading.Thread(target=self.solve_loop, daemon=True)
self.thread.start()
```

### 8.2 主循环

```python
def solve_loop(self):
    while self.running:
        try:
            self.update()
            time.sleep(0.01)
        except Exception as e:
            logging.error(f"Error solve_loop from device: {e}")
            time.sleep(1)
```

每 10 ms 执行一次，等效 100 Hz 控制频率。

### 8.3 update() 流程

```python
def update(self):
    roll, pitch, yaw = self.get_orientation()
    self.position, gripper, button_control = self.common_update()
    if self.if_limit_dof:
        self.check_limits_position()
    self.posture = [x, y, z, roll, pitch, yaw]
    return self.posture, gripper, button_control
```

### 8.4 姿态 → 方向向量

`get_orientation()` 把欧拉角转成三个方向向量，用于摇杆平移映射：

```python
# 前向向量（由 pitch、yaw 决定）
self.direction_vector = vec3(
    math.cos(pitch) * math.cos(yaw),
    math.cos(pitch) * math.sin(yaw),
    math.sin(pitch)
)

# 右向向量
self.direction_vector_right = vec3(
    math.cos(roll) * math.sin(-yaw),
    math.cos(roll) * math.cos(-yaw),
    math.sin(-roll)
)

# 上向向量
self.direction_vector_up = vec3(
    math.sin(-roll) * math.sin(-pitch),
    math.sin(-roll) * math.cos(-pitch),
    math.cos(-roll)
)
```

### 8.5 摇杆 → 位置控制

```python
# 前后推摇杆：沿 direction_vector 前后移动
if joycon_stick_v > 4000:
    self.position[0] += 0.001 * self.direction_vector[0] * dof_speed[0] * direction_reverse[0]
    self.position[2] += 0.001 * self.direction_vector[2] * dof_speed[2] * direction_reverse[2]
    if not self.if_close_y:
        self.position[1] += 0.001 * self.direction_vector[1] * dof_speed[1] * direction_reverse[1]
```

- 阈值 `4000` / `1000` 来自 Joy-Con 摇杆原始值范围（约 0-4095）。
- 每次步长 `0.001` 米（1 mm），乘以速度系数。

### 8.6 上下移动

- 默认 `pure_z=True`：R/L 按键直接控制 z 轴上下。
- `pure_z=False`：沿 `direction_vector_up` 上下移动。

### 8.7 水平摇杆模式

- `horizontal_stick_mode="y"`（默认）：左右推摇杆沿 `direction_vector_right` 平移。
- `horizontal_stick_mode="yaw_diff"`：左右推摇杆调整 yaw 偏移，用于 SO100 等需要单独旋转 yaw 的场景。

### 8.8 Home / Capture 复位

长按 Home（右手）/ Capture（左手）会把当前位姿逐步拉回初始偏移：

```python
if joycon_button_home == 1:
    # position 三轴逐次逼近 offset_position_m
    # yaw 逼近 offset_euler_rad[2]
    # 当接近零位时，重置 yaw 积分
    if abs(self.orientation_rad[2]) < 0.02 * self.dof_speed[5]:
        self.orientation_sensor.reset_yaw()
        self.yaw_diff = 0.0
        self.orientation_sensor.set_yaw_diff(self.yaw_diff)
```

### 8.9 夹爪控制

- 默认：ZR（右手）/ ZL（左手）作为夹爪切换按钮。
- `change_down_to_gripper=True`：把 ZR/ZL 改为向下移动，摇杆按下作为夹爪切换。

夹爪状态在两种 open/close 值之间切换：

```python
if self.gripper_toggle_button == 1:
    if self.gripper_state == self.gripper_open:
        self.gripper_state = self.gripper_close
    else:
        self.gripper_state = self.gripper_open
```

### 8.10 按键事件输出

`ButtonEventJoyCon` 提供边沿检测（按下/释放事件）。右手柄的 A/Y 被复用为数据录制控制：

| 按键 | 右手柄功能 | `button_control` |
|---|---|---|
| A | 保存并开始新录制 | 1 |
| Y | 删除并重录 | -1 |
| B | all_button_return 模式 | 2 |
| X | all_button_return 模式 | 3 |
| SR / SL | all_button_return 模式 | 4 / 5 |
| ZR | all_button_return 模式 | 6 |
| + | 复位姿态 | 8 |

---

## 9. 关键参数说明

| 参数 | 类型 | 说明 |
|---|---|---|
| `device` | str/tuple | `"right"` / `"left"` 或 `(vendor_id, product_id, serial)` |
| `gripper_open` / `gripper_close` | float | 夹爪开合目标值 |
| `offset_position_m` | list[3] | 初始/复位位置 [x, y, z]，单位 m |
| `offset_euler_rad` | list[3] | 姿态偏移 [roll, pitch, yaw]，单位 rad |
| `euler_reverse` | list[3] | 各轴方向取反（1 或 -1） |
| `direction_reverse` | list[3] | 平移方向取反 |
| `glimit` | list[list] | 位姿限幅 `[[min], [max]]` |
| `dof_speed` | list[6] | 各自由度速度系数 |
| `horizontal_stick_mode` | str | `"y"` 平移或 `"yaw_diff"` 旋转 |
| `pitch_down_double` | bool | 俯仰向下灵敏度加倍 |
| `pure_z` | bool | 上下按键是否只控制 z 轴 |
| `pure_dx` | bool | X/B 按键是否只控制 x 轴 |
| `change_down_to_gripper` | bool | 把夹爪切换映射到摇杆按下 |
| `rotation_filter_alpha_rate` | float | 姿态低通滤波系数倍率 |
| `common_rad` | bool | 使用 `π/1.5` 角度映射 |
| `lerobot` | bool | 针对 LeRobot SO100 的优化参数 |
| `all_button_return` | bool | 输出所有按钮的编码状态 |
| `without_rest_init` | bool | 跳过开机校准/复位 |

---

## 10. 典型使用示例

```python
from joyconrobotics import JoyconRobotics
import time

# 右手柄单臂遥操作
controller = JoyconRobotics("right")

for _ in range(1000):
    posture, gripper, button = controller.get_control(out_format="euler_rad")
    # posture = [x, y, z, roll, pitch, yaw]
    print(posture, gripper, button)
    time.sleep(0.01)

controller.disconnnect()
```

双臂场景可分别创建 `JoyconRobotics("right")` 和 `JoyconRobotics("left")`。

---

## 11. 实现特点与注意事项

1. **姿态融合是“工程化”而非“严格物理化”**
   互补滤波系数、`-1/86` 积分因子、`π/1.5` 映射、低通 alpha 都是经验参数，针对机器人遥操作手感调校，不追求最优估计理论。

2. **yaw 漂移是已知问题**
   yaw 无外部观测，依赖陀螺仪积分，长时间使用会漂移。JoyconRobotics 通过 `yaw_diff`、水平摇杆 yaw 调整模式、Home 键复位三种方式缓解。

3. **位置是“增量式”而非“绝对式”**
   位置通过摇杆积分累加得到，没有外部定位参考，因此必须配合限幅 `glimit` 和复位机制使用。

4. **多线程读取**
   HID 守护线程 `_update_input_report` 持续读最新报告；主线程 `solve_loop` 以 100 Hz 做控制解算。两者共享 `_input_report`，但没有显式锁保护 JoyCon 底层状态，实际运行依赖 GIL 和 49 字节原子赋值保证一致性。

5. **左手柄坐标镜像**
   通过 `_ime_yz_coeff = -1` 对左手柄 y/z 取反，保证左右手柄在机器人控制中的直观对应关系。

---

## 12. 小结

`joycon-robotics` 是在 `joycon-python` 的 Joy-Con 驱动能力之上，面向机器人遥操作场景做的一层完整封装：

- **保留** joycon-python 的 HID 通信、IMU 标定、传感器读取、按键事件等基础设施；
- **新增** 互补滤波姿态估计、方向向量映射、增量式位置控制、夹爪与按键状态机；
- **最终** 向用户输出可直接用于机器人控制的 `posture = [x, y, z, roll, pitch, yaw]`、`gripper_state`、`button_control`。

理解这套实现的关键不在于把它当作一个严格的 IMU 姿态估计算法，而在于把它看作一个**以 Joy-Con 为输入设备、以机器人末端执行器为目标控制对象的工程化遥操作系统**。
