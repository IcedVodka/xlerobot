#!/usr/bin/env python
"""
检测系统中所有摄像头设备 (/dev/video*) 的端口分配。

运行:
    python script/detect_cameras.py

输出示例:
    /dev/video0  USB Camera (046d:0825)  1280x720@30  OK
    /dev/video2  Intel RealSense D435    640x480@30   OK
    /dev/video4  (no device name)         --           OPEN FAILED
"""

import glob
import os
import sys


def get_sysfs_name(video_path: str) -> str:
    """从 sysfs 读取设备名称。"""
    dev_name = os.path.basename(video_path)
    sysfs_path = f"/sys/class/video4linux/{dev_name}/name"
    try:
        with open(sysfs_path, "r") as f:
            return f.read().strip()
    except Exception:
        return ""


def get_sysfs_info(video_path: str) -> dict:
    """从 sysfs 读取尽可能多的设备信息。"""
    dev_name = os.path.basename(video_path)
    base = f"/sys/class/video4linux/{dev_name}"
    info = {}

    # 设备名
    info["name"] = get_sysfs_name(video_path)

    # 尝试读 index
    try:
        with open(os.path.join(base, "index"), "r") as f:
            info["index"] = f.read().strip()
    except Exception:
        info["index"] = "?"

    # 尝试找到对应的 USB/PCI 设备路径，获取厂商/产品信息
    try:
        real_path = os.path.realpath(base)
        # 向上回溯找 device 链接
        device_link = os.path.join(real_path, "device")
        if os.path.islink(device_link):
            device_path = os.path.realpath(device_link)
            # 尝试读 manufacturer / product
            for key in ("manufacturer", "product", "idVendor", "idProduct"):
                try:
                    with open(os.path.join(device_path, key), "r") as f:
                        info[key] = f.read().strip()
                except Exception:
                    pass
    except Exception:
        pass

    return info


def try_opencv(video_path: str) -> tuple[bool, str]:
    """尝试用 OpenCV 打开摄像头并获取基本信息。"""
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "OPEN FAILED"

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        fps_str = f"{fps:.0f}" if fps > 0 else "?"
        return True, f"{width}x{height}@{fps_str}fps"
    except ImportError:
        return False, "opencv not installed"
    except Exception as e:
        return False, f"error: {e}"


def main():
    video_devices = sorted(glob.glob("/dev/video*"))

    if not video_devices:
        print("未找到任何 /dev/video* 设备")
        sys.exit(1)

    print(f"\n发现 {len(video_devices)} 个 video 设备:\n")
    print(f"{'Device':<14} {'Name':<30} {'Resolution':<16} {'Status':<20}")
    print("-" * 80)

    for dev in video_devices:
        info = get_sysfs_info(dev)
        name = info.get("name") or "(no name)"

        # 补充 USB 信息
        usb_info = ""
        if "idVendor" in info and "idProduct" in info:
            usb_info = f" [{info['idVendor']}:{info['idProduct']}]"
        if "manufacturer" in info:
            usb_info += f" {info['manufacturer']}"
        if "product" in info:
            usb_info += f" {info['product']}"

        ok, status = try_opencv(dev)
        status_icon = "✓" if ok else "✗"

        print(f"{dev:<14} {name:<30} {status:<16} {status_icon} {status}")
        if usb_info:
            print(f"  └─ sysfs:{usb_info}")

    print()

    # 额外：列出 /dev/v4l/by-id/ 和 /dev/v4l/by-path/ 的软链接
    by_id = glob.glob("/dev/v4l/by-id/*")
    by_path = glob.glob("/dev/v4l/by-path/*")

    if by_id:
        print("\n/dev/v4l/by-id/ (稳定标识，推荐用于固定配置):")
        for p in sorted(by_id):
            target = os.path.realpath(p)
            print(f"  {os.path.basename(p):<50} -> {target}")

    if by_path:
        print("\n/dev/v4l/by-path/ (按 USB 端口路径):")
        for p in sorted(by_path):
            target = os.path.realpath(p)
            print(f"  {os.path.basename(p):<50} -> {target}")

    print()


if __name__ == "__main__":
    main()
