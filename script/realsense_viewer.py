#!/usr/bin/env python3
"""
RealSense 深度 + 彩色流实时读取示例
=====================================
- 设备: 通过序列号指定 (SN: 327122072195)
- 分辨率: 640x480
- 深度流: rs.format.z16
- 彩色流: rs.format.bgr8 (OpenCV 默认格式)
- 对齐: 深度对齐到彩色流

用法:
    python realsense_viewer.py

按 'q' 退出窗口。
"""

from __future__ import annotations

import cv2
import numpy as np
import pyrealsense2 as rs

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
SERIAL_NUMBER = "327122072195"
WIDTH = 640
HEIGHT = 480
FPS = 30


def main() -> None:
    # 创建 pipeline 和 config
    pipe = rs.pipeline()
    cfg = rs.config()

    # 绑定指定设备（防止多相机时连错）
    cfg.enable_device(SERIAL_NUMBER)

    # 启用深度流
    cfg.enable_stream(
        rs.stream.depth,
        WIDTH,
        HEIGHT,
        rs.format.z16,
        FPS,
    )

    # 启用彩色流（OpenCV 默认 BGR）
    cfg.enable_stream(
        rs.stream.color,
        WIDTH,
        HEIGHT,
        rs.format.bgr8,
        FPS,
    )

    print(f"启动 RealSense (SN: {SERIAL_NUMBER})")
    print(f"  深度: {WIDTH}x{HEIGHT} @ {FPS}fps  (z16)")
    print(f"  彩色: {WIDTH}x{HEIGHT} @ {FPS}fps  (bgr8)")
    print("按 'q' 退出 ...\n")

    profile = pipe.start(cfg)

    # 深度对齐到彩色流（可选，需要时取消注释）
    align = rs.align(rs.stream.color)

    try:
        while True:
            # 等待帧（兼容旧版 SDK 的位置参数）
            frames = pipe.wait_for_frames(5000)

            # 对齐深度到彩色
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # 转为 numpy 数组
            depth_image = np.asanyarray(depth_frame.get_data())   # uint16, mm
            color_image = np.asanyarray(color_frame.get_data())   # uint8, BGR

            # 深度图转为可视化灰度 / 伪彩色
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET,
            )

            # 拼接显示（左彩色，右深度伪彩）
            combined = np.hstack((color_image, depth_colormap))
            cv2.imshow("RealSense  Color | Depth", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        print("\n已关闭 pipeline")


if __name__ == "__main__":
    main()
