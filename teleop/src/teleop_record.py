#!/usr/bin/env python
"""
XLerobot 遥操数据录制模块 — 通用封装

复用 LeRobot v3 数据格式，支持：
- 固定底盘 / 移动底盘 两套数据集特征
- Episode 控制（开始/重录/停止）
- 键盘/Joy-Con 事件绑定
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = logging.getLogger(__name__)

# ---- 头部控制 ----
# NOTE(cwl): 实际电机接线与代码命名假设相反：
#   head_motor_1 (ID=7) 实际控制 yaw（左右转），不是 pitch
#   head_motor_2 (ID=8) 实际控制 pitch（抬头/低头），不是 yaw
# 因此将按键映射互换，使 T/G = pitch，F/H = yaw
HEAD_KEYMAP = {
    "head_motor_1+": "f",   # F → yaw+（左转）
    "head_motor_1-": "h",   # H → yaw-（右转）
    "head_motor_2+": "t",   # T → pitch+（抬头）
    "head_motor_2-": "g",   # G → pitch-（低头）
}


def make_head_action(pressed_keys: set[str], current_head_pos: dict[str, float], step_deg: float = 2.0) -> dict[str, float]:
    """根据按键状态更新头部目标位置（位置控制）。"""
    if HEAD_KEYMAP["head_motor_1+"] in pressed_keys:
        current_head_pos["head_motor_1.pos"] += step_deg
    elif HEAD_KEYMAP["head_motor_1-"] in pressed_keys:
        current_head_pos["head_motor_1.pos"] -= step_deg
    if HEAD_KEYMAP["head_motor_2+"] in pressed_keys:
        current_head_pos["head_motor_2.pos"] += step_deg
    elif HEAD_KEYMAP["head_motor_2-"] in pressed_keys:
        current_head_pos["head_motor_2.pos"] -= step_deg
    return {
        "head_motor_1.pos": current_head_pos["head_motor_1.pos"],
        "head_motor_2.pos": current_head_pos["head_motor_2.pos"],
    }


# ---- 数据集特征 ----
def make_dataset_features(robot, fixed_base: bool = False, use_videos: bool = True) -> dict[str, dict]:
    """
    构建 LeRobot v3 数据集特征定义。

    Args:
        robot: XLerobotClient 实例
        fixed_base: 是否固定底盘（True=不含底盘速度特征）
        use_videos: 是否使用视频编码存储图像

    Returns:
        features: LeRobot 风格的特征字典
    """
    action_features = dict(robot.action_features)
    obs_features = dict(robot.observation_features)

    if fixed_base:
        # 移除底盘速度特征
        for key in ["x.vel", "y.vel", "theta.vel"]:
            action_features.pop(key, None)
            obs_features.pop(key, None)

    action_ds_features = hw_to_dataset_features(action_features, "action")
    obs_ds_features = hw_to_dataset_features(obs_features, "observation")

    # 合并
    features = {**action_ds_features, **obs_ds_features}
    return features


class TeleopRecorder:
    """
    遥操数据录制器。

    封装 LeRobotDataset 的创建、帧写入、episode 管理。
    支持键盘/Joy-Con 事件触发 episode 控制。
    """

    def __init__(
        self,
        repo_id: str,
        robot,
        fps: int = 30,
        fixed_base: bool = False,
        use_videos: bool = True,
        single_task: str = "xlerobot teleop task",
        root: str | Path | None = None,
        image_writer_processes: int = 0,
        image_writer_threads_per_camera: int = 4,
    ):
        self.repo_id = repo_id
        self.robot = robot
        self.fps = fps
        self.fixed_base = fixed_base
        self.single_task = single_task
        self.use_videos = use_videos
        self.image_writer_processes = image_writer_processes
        self.image_writer_threads_per_camera = image_writer_threads_per_camera

        self.dataset: LeRobotDataset | None = None
        self.num_cameras = len(getattr(robot, "cameras", {}))
        self._init_dataset()

        # Episode 状态
        self.is_recording = False
        self.episode_count = 0
        self.frame_count = 0
        self.episode_start_time = 0.0

        # 处理器
        self.teleop_action_processor, self.robot_action_processor, self.robot_observation_processor = make_default_processors()

    def _init_dataset(self):
        """初始化 LeRobotDataset。"""
        features = make_dataset_features(self.robot, fixed_base=self.fixed_base, use_videos=self.use_videos)

        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            root=self.root,
            robot_type=self.robot.name,
            features=features,
            use_videos=self.use_videos,
            image_writer_processes=self.image_writer_processes,
            image_writer_threads=self.image_writer_threads_per_camera * self.num_cameras,
        )
        logger.info(f"Dataset created: {self.repo_id}, fixed_base={self.fixed_base}")

    @property
    def root(self) -> Path | None:
        return None

    def start_episode(self) -> None:
        """开始录制新 episode。"""
        self.is_recording = True
        self.frame_count = 0
        self.episode_start_time = time.perf_counter()
        self.episode_count += 1
        logger.info(f"=== Episode {self.episode_count} started ===")

    def save_episode(self) -> None:
        """保存当前 episode。"""
        if self.dataset is not None and self.frame_count > 0:
            self.dataset.save_episode()
            logger.info(f"Episode {self.episode_count} saved ({self.frame_count} frames)")
        self.is_recording = False
        self.frame_count = 0

    def rerecord_episode(self) -> None:
        """重录当前 episode（丢弃已录制的帧）。"""
        if self.dataset is not None:
            self.dataset.clear_episode_buffer()
            self.frame_count = 0
            self.episode_start_time = time.perf_counter()
            logger.info(f"Episode {self.episode_count} cleared, re-recording...")

    def stop_recording(self) -> None:
        """停止录制，保存当前 episode，finalize 数据集。"""
        self.save_episode()
        self.is_recording = False
        logger.info("Recording stopped.")

    def record_frame(self, obs: dict[str, Any], action: dict[str, Any]) -> None:
        """录制单帧数据。"""
        if not self.is_recording or self.dataset is None:
            return

        # 处理观测
        obs_processed = self.robot_observation_processor(obs)
        observation_frame = build_dataset_frame(self.dataset.features, obs_processed, prefix="observation")

        # 处理动作（根据 fixed_base 过滤底盘速度）
        action_to_record = dict(action)
        if self.fixed_base:
            for key in ["x.vel", "y.vel", "theta.vel"]:
                action_to_record.pop(key, None)

        action_frame = build_dataset_frame(self.dataset.features, action_to_record, prefix="action")

        # 构建帧并写入
        frame = {**observation_frame, **action_frame, "task": self.single_task}
        self.dataset.add_frame(frame)
        self.frame_count += 1

    def finalize(self) -> None:
        """Finalize 数据集，关闭 writer。"""
        if self.dataset is not None:
            if self.is_recording and self.frame_count > 0:
                self.save_episode()
            self.dataset.finalize()
            logger.info(f"Dataset finalized: {self.repo_id}")

    def get_status(self) -> str:
        """返回当前录制状态。"""
        if not self.is_recording:
            return "[READY] Press Space/+ to start recording"
        return f"[RECORDING] Episode {self.episode_count}, Frame {self.frame_count}"


class TeleopRecordManager:
    """
    录制管理器 — 负责两套数据集的协调管理。

    同时维护固定底盘和移动底盘两个 TeleopRecorder，
    根据用户选择决定录制到哪套数据集。
    """

    def __init__(
        self,
        repo_id_fixed: str | None,
        repo_id_mobile: str | None,
        robot,
        fps: int = 30,
        use_videos: bool = True,
        single_task: str = "xlerobot teleop task",
    ):
        self.recorder_fixed: TeleopRecorder | None = None
        self.recorder_mobile: TeleopRecorder | None = None
        self.active_recorder: TeleopRecorder | None = None

        if repo_id_fixed:
            self.recorder_fixed = TeleopRecorder(
                repo_id=repo_id_fixed,
                robot=robot,
                fps=fps,
                fixed_base=True,
                use_videos=use_videos,
                single_task=single_task,
            )
            logger.info(f"Fixed-base dataset: {repo_id_fixed}")

        if repo_id_mobile:
            self.recorder_mobile = TeleopRecorder(
                repo_id=repo_id_mobile,
                robot=robot,
                fps=fps,
                fixed_base=False,
                use_videos=use_videos,
                single_task=single_task,
            )
            logger.info(f"Mobile-base dataset: {repo_id_mobile}")

        # 默认激活移动底盘数据集（如果存在）
        self.active_recorder = self.recorder_mobile or self.recorder_fixed

    def switch_dataset(self, fixed_base: bool) -> None:
        """切换当前录制的数据集。"""
        if fixed_base and self.recorder_fixed:
            self.active_recorder = self.recorder_fixed
            logger.info("Switched to fixed-base dataset")
        elif not fixed_base and self.recorder_mobile:
            self.active_recorder = self.recorder_mobile
            logger.info("Switched to mobile dataset")

    def start_episode(self) -> None:
        if self.active_recorder:
            self.active_recorder.start_episode()

    def save_episode(self) -> None:
        if self.active_recorder:
            self.active_recorder.save_episode()

    def rerecord_episode(self) -> None:
        if self.active_recorder:
            self.active_recorder.rerecord_episode()

    def stop_recording(self) -> None:
        if self.active_recorder:
            self.active_recorder.stop_recording()

    def record_frame(self, obs: dict[str, Any], action: dict[str, Any]) -> None:
        if self.active_recorder:
            self.active_recorder.record_frame(obs, action)

    def finalize(self) -> None:
        if self.recorder_fixed:
            self.recorder_fixed.finalize()
        if self.recorder_mobile:
            self.recorder_mobile.finalize()

    def get_status(self) -> str:
        if self.active_recorder:
            mode = "FIXED" if self.active_recorder.fixed_base else "MOBILE"
            return f"[{mode}] {self.active_recorder.get_status()}"
        return "[NO DATASET]"
