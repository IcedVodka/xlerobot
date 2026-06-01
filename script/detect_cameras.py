#!/usr/bin/env python
"""
逐个检测摄像头：显示实时视频流，按任意键切换下一个，按 q 退出。

运行:
    python script/detect_cameras.py

按键:
    任意键(除q)  切换到下一个摄像头
    q / ESC      退出
"""

import glob
import os
import sys

try:
    import cv2
except ImportError:
    print("错误：需要安装 opencv-python")
    print("  pip install opencv-python")
    sys.exit(1)


def get_sysfs_name(video_path: str) -> str:
    """从 sysfs 读取设备名称。"""
    dev_name = os.path.basename(video_path)
    sysfs_path = f"/sys/class/video4linux/{dev_name}/name"
    try:
        with open(sysfs_path, "r") as f:
            return f.read().strip()
    except Exception:
        return ""


def get_usb_info(video_path: str) -> str:
    """获取 USB 厂商/产品信息。"""
    dev_name = os.path.basename(video_path)
    try:
        real_path = os.path.realpath(f"/sys/class/video4linux/{dev_name}")
        device_link = os.path.join(real_path, "device")
        if not os.path.islink(device_link):
            return ""
        device_path = os.path.realpath(device_link)

        parts = []
        for key in ("manufacturer", "product"):
            try:
                with open(os.path.join(device_path, key), "r") as f:
                    v = f.read().strip()
                    if v:
                        parts.append(v)
            except Exception:
                pass
        for key in ("idVendor", "idProduct"):
            try:
                with open(os.path.join(device_path, key), "r") as f:
                    v = f.read().strip()
                    if v:
                        parts.append(f"{key}={v}")
            except Exception:
                pass
        return " | ".join(parts) if parts else ""
    except Exception:
        return ""


def get_stable_links(video_path: str) -> list[str]:
    """获取 /dev/v4l/by-id/ 和 by-path/ 中指向该设备的软链接。"""
    results = []
    for base in ("/dev/v4l/by-id", "/dev/v4l/by-path"):
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            full = os.path.join(base, name)
            if os.path.islink(full) and os.path.realpath(full) == video_path:
                results.append(full)
    return results


def show_camera(dev_path: str, index: int, total: int) -> bool:
    """
    打开一个摄像头并显示实时视频流。
    返回 True 表示用户要求下一个，False 表示退出。
    """
    name = get_sysfs_name(dev_path)
    usb_info = get_usb_info(dev_path)
    stable_links = get_stable_links(dev_path)

    cap = cv2.VideoCapture(dev_path)
    if not cap.isOpened():
        # 创建黑屏显示错误信息
        frame = None
        status = "OPEN FAILED"
    else:
        status = "OK"

    info_lines = [
        f"Device: {dev_path}",
        f"Name: {name or '(unknown)'}",
        f"USB: {usb_info or '(no usb info)'}",
    ]
    if stable_links:
        info_lines.append("Stable links:")
        for link in stable_links:
            info_lines.append(f"  {link}")
    info_lines.append(f"")
    info_lines.append(f"[{index+1}/{total}] 按任意键下一个 | 按 q 退出")

    while True:
        if cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = None
                status = "FRAME ERROR"

        if frame is None:
            # 创建黑屏
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, status, (200, 240), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 255), 3)

        # 在画面上叠加信息
        overlay = frame.copy()
        h, w = frame.shape[:2]
        # 半透明黑色背景
        cv2.rectangle(overlay, (10, 10), (min(600, w - 10), 30 + len(info_lines) * 25),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y = 35
        for line in info_lines:
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 1)
            y += 25

        # 显示分辨率
        if cap.isOpened():
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            res_text = f"{fw}x{fh} @ {fps:.1f}fps"
            cv2.putText(frame, res_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

        cv2.imshow("Camera Test", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:  # q 或 ESC
            if cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            return False
        elif key != 255:  # 任意其他键 -> 下一个
            if cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            return True


def main():
    video_devices = sorted(glob.glob("/dev/video*"))

    if not video_devices:
        print("未找到任何 /dev/video* 设备")
        sys.exit(1)

    print(f"\n发现 {len(video_devices)} 个 video 设备，准备逐个预览...")
    print("按键说明: 任意键 = 下一个摄像头,  q/ESC = 退出\n")

    try:
        import numpy as np
    except ImportError:
        print("需要安装 numpy: pip install numpy")
        sys.exit(1)

    for i, dev in enumerate(video_devices):
        if not show_camera(dev, i, len(video_devices)):
            print("用户退出")
            break
    else:
        print("所有摄像头已浏览完毕")

    cv2.destroyAllWindows()
    print("结束")


if __name__ == "__main__":
    main()
