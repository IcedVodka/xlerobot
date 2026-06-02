#!/usr/bin/env python3
"""
RealSense 逐级排查脚本
===================
从 Python 环境 -> SDK 导入 -> 设备枚举 -> Pipeline 测试 -> 系统层信息，
逐步打印每一层的状态，帮助定位 Orin/PC 上找不到相机的问题。

用法:
    python check_realsense.py

输出颜色:
    [OK]   绿色 - 该步骤通过
    [WARN] 黄色 - 警告，可能不是致命问题
    [FAIL] 红色 - 该步骤失败，需要关注
    [INFO] 默认 - 普通信息
"""

from __future__ import annotations

import ctypes
import getpass
import glob
import grp
import os
import platform
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 彩色输出工具
# ---------------------------------------------------------------------------

class Colors:
    OK = "\033[32m"      # 绿色
    WARN = "\033[33m"    # 黄色
    FAIL = "\033[31m"    # 红色
    INFO = "\033[36m"    # 青色
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @classmethod
    def ok(cls, msg: str) -> str:
        return f"{cls.OK}[OK]{cls.RESET}   {msg}"

    @classmethod
    def warn(cls, msg: str) -> str:
        return f"{cls.WARN}[WARN]{cls.RESET} {msg}"

    @classmethod
    def fail(cls, msg: str) -> str:
        return f"{cls.FAIL}[FAIL]{cls.RESET} {msg}"

    @classmethod
    def info(cls, msg: str) -> str:
        return f"{cls.INFO}[INFO]{cls.RESET} {msg}"

    @classmethod
    def section(cls, title: str) -> str:
        return f"\n{cls.BOLD}{'=' * 60}{cls.RESET}\n{cls.BOLD}{title}{cls.RESET}\n{cls.BOLD}{'=' * 60}{cls.RESET}"


def print_divider() -> None:
    print("-" * 60)


def print_exception(exc: Exception, label: str = "Exception") -> None:
    print(Colors.fail(f"{label}: {type(exc).__name__}: {exc}"))


# ---------------------------------------------------------------------------
# 步骤 1: Python 环境与 pyrealsense2 安装检查
# ---------------------------------------------------------------------------

def check_python_env() -> tuple[bool, object]:
    """检查 Python 版本、架构、pyrealsense2 能否导入及版本路径。"""
    print(Colors.section("步骤 1/6: Python 环境与 pyrealsense2 安装检查"))

    print(f"  Python 版本:   {platform.python_version()}")
    print(f"  Python 可执行: {sys.executable}")
    print(f"  平台:          {platform.platform()}")
    print(f"  机器架构:      {platform.machine()}")
    print(f"  当前用户:      {getpass.getuser()}")
    print(f"  UID:           {os.getuid()}")
    print(f"  Groups:        {', '.join(str(g) for g in os.getgroups())}")

    print_divider()

    # 尝试导入 pyrealsense2
    print("尝试 import pyrealsense2 ...")
    try:
        import pyrealsense2 as rs
        print(Colors.ok("pyrealsense2 导入成功"))
    except ImportError as e:
        print(Colors.fail(f"pyrealsense2 导入失败 (ImportError): {e}"))
        print(Colors.info("提示: 请确认是否已安装 pyrealsense2:"))
        print("      pip show pyrealsense2")
        print("      在 Jetson/ARM 上需要编译安装或使用 Jetson-specific 包:")
        print("      https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md")
        return False, None
    except Exception as e:
        print(Colors.fail(f"pyrealsense2 导入失败 (未知异常): {type(e).__name__}: {e}"))
        return False, None

    # 版本与路径
    version = getattr(rs, "__version__", "<unknown>")
    file_path = getattr(rs, "__file__", "<unknown>")
    print(f"  pyrealsense2 版本: {version}")
    print(f"  pyrealsense2 路径: {file_path}")

    # 检查路径是否匹配当前架构
    file_path_lower = file_path.lower()
    machine = platform.machine().lower()
    if "x86_64" in file_path_lower and machine in ("aarch64", "arm64"):
        print(Colors.warn("⚠️ 警告: pyrealsense2 路径包含 x86_64 字样，但当前是 ARM 架构!"))
        print(Colors.info("  你可能安装的是 x86_64 版本的 pyrealsense2，需要重新编译/安装 ARM 版本。"))
    elif "amd64" in file_path_lower and machine in ("aarch64", "arm64"):
        print(Colors.warn("⚠️ 警告: pyrealsense2 路径包含 amd64 字样，但当前是 ARM 架构!"))
        print(Colors.info("  你可能安装的是 x86_64 版本的 pyrealsense2，需要重新编译/安装 ARM 版本。"))
    else:
        print(Colors.ok("架构与库路径看起来匹配"))

    # 检查关键符号是否存在
    print_divider()
    print("检查关键类/函数是否存在 ...")
    required_symbols = [
        "context", "pipeline", "config", "pipeline_profile",
        "stream", "format", "camera_info",
    ]
    missing = []
    for sym in required_symbols:
        if not hasattr(rs, sym):
            missing.append(sym)
            print(Colors.fail(f"  缺失: rs.{sym}"))
        else:
            print(Colors.ok(f"  存在: rs.{sym}"))

    if missing:
        print(Colors.fail(f"缺少 {len(missing)} 个关键符号，SDK 可能不完整"))
        return False, rs

    print(Colors.ok("所有关键符号都存在"))
    return True, rs


# ---------------------------------------------------------------------------
# 步骤 2: SDK 上下文与设备枚举
# ---------------------------------------------------------------------------

def check_context_and_devices(rs_module) -> tuple[bool, list]:
    """检查 rs.context 能否创建，query_devices 能否返回设备。"""
    print(Colors.section("步骤 2/6: SDK 上下文创建与设备枚举"))

    # 2.1 创建 context
    print("尝试创建 rs.context() ...")
    try:
        ctx = rs_module.context()
        print(Colors.ok("rs.context() 创建成功"))
    except Exception as e:
        print(Colors.fail("rs.context() 创建失败"))
        print_exception(e, "rs.context()")
        return False, []

    # 2.2 查询设备数量
    print("尝试 ctx.query_devices() ...")
    try:
        devices = list(ctx.query_devices())
        count = len(devices)
        print(f"  发现设备数量: {count}")
        if count == 0:
            print(Colors.warn("⚠️ 未检测到任何 RealSense 设备"))
            print(Colors.info("  可能原因:"))
            print("    - 相机未连接 USB")
            print("    - USB 线缆/接口有问题（尝试换线/换口）")
            print("    - 相机供电不足（D435i 需要 USB3.0 才能稳定工作，USB2.1 可能只能低分辨率）")
            print("    - udev 规则未配置，设备被系统忽略")
            print("    - 内核模块 realsense 驱动未加载")
            return True, []  # context 成功但无设备，继续后面系统检查
        else:
            print(Colors.ok(f"检测到 {count} 个 RealSense 设备"))
    except Exception as e:
        print(Colors.fail("ctx.query_devices() 调用失败"))
        print_exception(e, "query_devices")
        return False, []

    return True, devices


# ---------------------------------------------------------------------------
# 步骤 3: 设备详情与传感器信息
# ---------------------------------------------------------------------------

def check_device_details(rs_module, devices) -> bool:
    """打印每个设备的详细信息，包括所有 camera_info 和传感器 profile。"""
    print(Colors.section("步骤 3/6: 设备详细信息"))

    if not devices:
        print(Colors.warn("无设备可检查，跳过此步骤"))
        return True

    # camera_info 核心字段（所有设备都应支持）
    core_info_fields = [
        ("name", rs_module.camera_info.name),
        ("serial_number", rs_module.camera_info.serial_number),
        ("firmware_version", rs_module.camera_info.firmware_version),
        ("usb_type_descriptor", rs_module.camera_info.usb_type_descriptor),
        ("physical_port", rs_module.camera_info.physical_port),
        ("product_id", rs_module.camera_info.product_id),
        ("product_line", rs_module.camera_info.product_line),
    ]
    # 可选字段（某些设备/固件版本可能不支持）
    optional_info_fields = [
        ("recommended_firmware_version", rs_module.camera_info.recommended_firmware_version),
    ]

    for i, device in enumerate(devices):
        print(f"\n  --- 设备 #{i} ---")

        # 打印核心字段
        for label, info_enum in core_info_fields:
            try:
                value = device.get_info(info_enum)
                print(f"    {label:<30}: {value}")
            except Exception as e:
                print(Colors.warn(f"    {label:<30}: [获取失败] {e}"))

        # 打印可选字段
        for label, info_enum in optional_info_fields:
            try:
                value = device.get_info(info_enum)
                print(f"    {label:<30}: {value}")
            except Exception:
                pass  # 可选字段获取失败不打印

        # 传感器信息
        print("    --- Sensors ---")
        try:
            sensors = device.query_sensors()
            for s_idx, sensor in enumerate(sensors):
                try:
                    s_name = sensor.get_info(rs_module.camera_info.name)
                except Exception:
                    s_name = f"<sensor #{s_idx}>"
                print(f"      Sensor '{s_name}':")

                try:
                    profiles = sensor.get_stream_profiles()
                    video_profiles = [
                        p for p in profiles
                        if hasattr(p, "is_video_stream_profile") and p.is_video_stream_profile()
                    ]
                    print(f"        共 {len(profiles)} 个 profile，其中 {len(video_profiles)} 个视频 profile")

                    for p in video_profiles:
                        vp = p.as_video_stream_profile()
                        stream_name = vp.stream_name()
                        fmt = vp.format().name
                        w, h = vp.width(), vp.height()
                        fps = vp.fps()
                        marker = " [DEFAULT]" if p.is_default() else ""
                        print(f"          {stream_name:<12} {fmt:<8} {w}x{h} @{fps}fps{marker}")
                except Exception as e:
                    print(Colors.warn(f"        获取 stream profiles 失败: {e}"))
        except Exception as e:
            print(Colors.warn(f"    获取 sensors 失败: {e}"))

    return True


# ---------------------------------------------------------------------------
# 步骤 4: Pipeline 连通性测试
# ---------------------------------------------------------------------------

def check_pipeline(rs_module, devices) -> bool:
    """对每个检测到的设备，尝试创建 pipeline 并读取一帧。"""
    print(Colors.section("步骤 4/6: Pipeline 连通性测试"))

    if not devices:
        print(Colors.warn("无设备可测试，跳过此步骤"))
        return True

    for i, device in enumerate(devices):
        try:
            sn = device.get_info(rs_module.camera_info.serial_number)
            name = device.get_info(rs_module.camera_info.name)
        except Exception as e:
            print(Colors.warn(f"设备 #{i}: 无法获取序列号，跳过: {e}"))
            continue

        print(f"\n  --- 测试设备: {name} (SN: {sn}) ---")

        # 4.1 创建 pipeline
        print("    创建 rs.pipeline() ...")
        try:
            pipe = rs_module.pipeline()
        except Exception as e:
            print(Colors.fail(f"    创建 pipeline 失败: {e}"))
            continue

        # 4.2 创建 config 并 enable_device
        print("    创建 rs.config() 并 enable_device ...")
        cfg = rs_module.config()
        try:
            rs_module.config.enable_device(cfg, sn)
            print(Colors.ok("    rs.config.enable_device() 成功"))
        except Exception as e:
            print(Colors.fail(f"    rs.config.enable_device() 失败: {e}"))
            continue

        # 4.3 配置 color stream（使用默认分辨率）
        print("    配置 color stream ...")
        try:
            cfg.enable_stream(rs_module.stream.color)
        except Exception as e:
            print(Colors.fail(f"    cfg.enable_stream(color) 失败: {e}"))
            continue

        # 4.4 启动 pipeline
        print("    启动 pipeline ...")
        try:
            profile = pipe.start(cfg)
            print(Colors.ok("    pipeline.start() 成功"))
        except Exception as e:
            print(Colors.fail(f"    pipeline.start() 失败: {e}"))
            print(Colors.info("    可能原因: 设备正被其他进程占用 / 固件不兼容 / USB 带宽不足"))
            continue

        # 4.5 尝试读取一帧
        print("    等待帧 (timeout=5000ms) ...")
        try:
            # 某些 Jetson/apt 安装的旧版 pyrealsense2 不支持 timeout_ms 关键字参数
            # 先尝试位置参数，失败再尝试无参数（使用默认超时）
            try:
                frames = pipe.wait_for_frames(5000)
            except TypeError:
                frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                data = color_frame.get_data()
                arr = ctypes.cast(data, ctypes.POINTER(ctypes.c_uint8 * (color_frame.get_width() * color_frame.get_height() * 3))).contents
                print(Colors.ok(f"    成功读取一帧: {color_frame.get_width()}x{color_frame.get_height()}"))
            else:
                print(Colors.warn("    收到 frames 但 get_color_frame() 返回 None"))
        except Exception as e:
            print(Colors.fail(f"    读取帧失败: {e}"))
        finally:
            try:
                pipe.stop()
                print(Colors.ok("    pipeline 已关闭"))
            except Exception as e:
                print(Colors.warn(f"    pipeline.stop() 异常: {e}"))

    return True


# ---------------------------------------------------------------------------
# 步骤 5: 系统层信息（Linux）
# ---------------------------------------------------------------------------

def run_shell(cmd: list[str]) -> tuple[int, str, str]:
    """运行 shell 命令，返回 (returncode, stdout, stderr)。"""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)


def check_system_level() -> None:
    """检查 lsusb、video 组、udev 规则、/dev/video* 等系统信息。"""
    print(Colors.section("步骤 5/6: 系统层信息 (Linux)"))

    # 5.1 lsusb
    print("  [lsusb 中的 RealSense 设备]")
    rc, out, err = run_shell(["lsusb"])
    if rc == 0:
        lines = out.strip().split("\n")
        realsense_lines = [l for l in lines if "intel" in l.lower() or "8086" in l]
        if realsense_lines:
            for l in realsense_lines:
                print(f"    {l}")
        else:
            print(Colors.warn("    lsusb 中未找到 Intel/8086 设备"))
            print(Colors.info("    提示: 8086 是 Intel 的 USB Vendor ID，RealSense 相机应显示为 8086:0b3a (D435i)"))
    else:
        print(Colors.fail(f"    lsusb 执行失败: {err}"))

    print_divider()

    # 5.2 /dev/video* 节点
    print("  [/dev/video* 节点]")
    video_devices = sorted(glob.glob("/dev/video*"))
    if video_devices:
        for vd in video_devices:
            try:
                # 尝试读取 symlink
                real = os.path.realpath(vd)
                if real != vd:
                    print(f"    {vd} -> {real}")
                else:
                    print(f"    {vd}")
            except Exception:
                print(f"    {vd}")
    else:
        print(Colors.warn("    未找到 /dev/video* 设备节点"))

    print_divider()

    # 5.3 当前用户是否在 video 组
    print("  [用户权限检查]")
    user = getpass.getuser()
    try:
        groups = [g.gr_name for g in grp.getgrall() if user in g.gr_mem or g.gr_gid == os.getgid()]
        print(f"    用户 '{user}' 所在组: {', '.join(groups)}")
        if "video" in groups:
            print(Colors.ok("    用户在 video 组中"))
        else:
            print(Colors.warn(f"    ⚠️ 用户 '{user}' 不在 video 组中!"))
            print(Colors.info("      解决: sudo usermod -aG video $USER 然后重新登录"))
    except Exception as e:
        print(Colors.fail(f"    获取用户组失败: {e}"))

    print_divider()

    # 5.4 udev 规则
    print("  [udev 规则检查]")
    udev_paths = [
        "/etc/udev/rules.d/99-realsense-libusb.rules",
        "/lib/udev/rules.d/99-realsense-libusb.rules",
        "/etc/udev/rules.d/",
        "/lib/udev/rules.d/",
    ]
    found_any = False
    for path in udev_paths:
        if os.path.isfile(path):
            print(Colors.ok(f"    找到 udev 规则文件: {path}"))
            found_any = True
        elif os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*realsense*")))
            for f in files:
                print(Colors.ok(f"    找到 udev 规则文件: {f}"))
                found_any = True
    if not found_any:
        print(Colors.warn("    未找到 RealSense udev 规则文件"))
        print(Colors.info("      解决: 在 librealsense 源码目录执行:"))
        print("              sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/")
        print("              sudo udevadm control --reload-rules && sudo udevadm trigger")

    print_divider()

    # 5.5 检查 /sys/bus/usb/devices 中是否有 8086 设备
    print("  [USB 设备树中的 Intel 设备]")
    try:
        usb_ids = Path("/sys/bus/usb/devices").glob("*/idVendor")
        found_intel = False
        for vid_file in usb_ids:
            vid = vid_file.read_text().strip().lower()
            if vid == "8086":
                dev_dir = vid_file.parent
                pid = (dev_dir / "idProduct").read_text().strip()
                busnum = (dev_dir / "busnum").read_text().strip()
                devnum = (dev_dir / "devnum").read_text().strip()
                speed = (dev_dir / "speed").read_text().strip() if (dev_dir / "speed").exists() else "?"
                print(f"    bus={busnum} dev={devnum}  pid={pid}  speed={speed}Mbps")
                found_intel = True
        if not found_intel:
            print(Colors.warn("    /sys/bus/usb/devices 中未找到 Intel(8086) USB 设备"))
    except Exception as e:
        print(Colors.fail(f"    读取 USB 设备树失败: {e}"))


# ---------------------------------------------------------------------------
# 步骤 6: USB 拓扑与带宽
# ---------------------------------------------------------------------------

def check_usb_topology() -> None:
    """打印 USB 拓扑，特别关注 RealSense 设备接在哪里。"""
    print(Colors.section("步骤 6/6: USB 拓扑与带宽"))

    # 尝试 lsusb -t
    print("  [USB 拓扑树 (lsusb -t)]")
    rc, out, err = run_shell(["lsusb", "-t"])
    if rc == 0:
        lines = out.strip().split("\n")
        # 打印完整拓扑，但高亮包含 Intel/RealSense 的行
        for line in lines:
            if any(k in line.lower() for k in ("video", "camera", "intel")):
                print(f"    {Colors.BOLD}{line}{Colors.RESET}")
            else:
                print(f"    {line}")
    else:
        print(Colors.warn(f"    lsusb -t 执行失败: {err}"))

    print_divider()

    # 检查每个 Intel USB 设备的 speed
    print("  [RealSense 设备 USB 速度]")
    rc, out, _ = run_shell(["lsusb"])
    if rc != 0:
        print(Colors.warn("    无法执行 lsusb"))
        return

    for line in out.strip().split("\n"):
        if "8086" in line:
            parts = line.split()
            # lsusb 格式: "Bus 002 Device 003: ID 8086:0b3a ..."
            bus = None
            dev = None
            if len(parts) >= 4 and parts[0] == "Bus" and parts[2] == "Device":
                bus = parts[1]
                dev = parts[3].rstrip(":")
            elif len(parts) >= 2 and ":" in parts[1]:
                # 备用格式: "002:003" 等
                bus_dev = parts[1].rstrip(":")
                bus = bus_dev.split(":")[0]
                dev = bus_dev.split(":")[1]

            if bus is not None and dev is not None:
                speed_file = Path(f"/sys/bus/usb/devices/{bus}-{dev}/speed")
                if speed_file.exists():
                    speed = speed_file.read_text().strip()
                    print(f"    {line}")
                    print(f"      -> USB speed: {speed} Mbps")
                    if speed in ("480", "12", "1.5"):
                        print(Colors.warn(f"      -> ⚠️ 当前是 USB2.x 速度! D435i 建议用 USB3.0 (5000Mbps)"))
                    elif speed == "5000":
                        print(Colors.ok(f"      -> ✅ USB3.0 (5Gbps)"))
                    elif speed == "10000":
                        print(Colors.ok(f"      -> ✅ USB3.1 (10Gbps)"))
                else:
                    print(f"    {line}")
            else:
                print(f"    {line}")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> int:
    print(Colors.BOLD + "=" * 60)
    print("RealSense 逐级排查脚本")
    print("=" * 60 + Colors.RESET)
    print("\n此脚本将逐步检查 pyrealsense2 SDK 到系统层的每个环节，")
    print("帮助定位 Orin/PC 上检测不到 RealSense 相机的问题。\n")

    # 步骤 1
    ok, rs = check_python_env()
    if not ok:
        print(Colors.fail("\n步骤 1 失败，后续 SDK 检查无法进行。"))
        print(Colors.info("请先解决 pyrealsense2 的安装问题，然后重新运行本脚本。"))
        # 即使 SDK 失败，也继续检查系统层
        print_divider()
        check_system_level()
        check_usb_topology()
        return 1

    # 步骤 2
    ok, devices = check_context_and_devices(rs)
    if not ok:
        print(Colors.fail("\n步骤 2 失败，无法创建 RealSense 上下文。"))
        print(Colors.info("请检查 pyrealsense2 版本是否与系统兼容。"))
        check_system_level()
        check_usb_topology()
        return 1

    # 步骤 3
    check_device_details(rs, devices)

    # 步骤 4
    check_pipeline(rs, devices)

    # 步骤 5
    check_system_level()

    # 步骤 6
    check_usb_topology()

    # 总结
    print(Colors.section("排查总结"))
    if not devices:
        print(Colors.fail("未检测到 RealSense 设备，请检查:"))
        print("  1. 相机是否正确连接 USB")
        print("  2. USB 线缆是否支持数据传输（不是充电线）")
        print("  3. 换 USB 端口尝试（尤其是 USB3.0 端口）")
        print("  4. 检查 udev 规则是否配置")
        print("  5. 检查当前用户是否在 video 组")
        print("  6. dmesg | tail 查看内核日志中是否有 USB 连接/断开信息")
        return 1
    else:
        print(Colors.ok(f"检测到 {len(devices)} 个 RealSense 设备"))
        print(Colors.info("如果 pipeline 测试通过但 lerobot 仍然找不到，请检查 lerobot 的日志级别"))
        print("      或确认 pyrealsense2 版本与 lerobot 兼容。")
        return 0


if __name__ == "__main__":
    sys.exit(main())
