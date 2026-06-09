#!/usr/bin/env python
"""
XLerobot 遥操数据采集共用工具模块

提供：
- 双臂-only features 过滤
- 动作合并与补齐
- 通用录制循环
- Episode 提示与等待工具
"""

from __future__ import annotations

import logging
import platform
import time
from typing import Callable

import numpy as np

from lerobot.datasets.utils import build_dataset_frame
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import log_rerun_data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Features 过滤
# ---------------------------------------------------------------------------


def filter_arm_head_features(features: dict) -> dict:
    """从完整 dataset features 中保留双臂 + 头部字段。

    对 ``action`` 和 ``observation.state`` 条目，只保留 names 列表中以
    ``left_arm_``、``right_arm_`` 或 ``head_motor_`` 开头的字段；
    图像字段全部保留；DEFAULT_FEATURES（timestamp、frame_index 等）也保留。
    """
    from lerobot.datasets.utils import DEFAULT_FEATURES

    filtered = {}
    for key, ft in features.items():
        # 保留 LeRobot 默认字段（timestamp、frame_index 等）
        if key in DEFAULT_FEATURES:
            filtered[key] = ft
            continue

        # 图像 / 视频字段全部保留
        if ft.get("dtype") in ("image", "video"):
            filtered[key] = ft
            continue

        # 对 action 和 observation.state 只保留双臂 + 头部字段
        if key in ("action", "observation.state"):
            arm_head_names = [
                name for name in ft.get("names", [])
                if name.startswith(("left_arm_", "right_arm_", "head_motor_"))
            ]
            if arm_head_names:
                filtered[key] = {
                    "dtype": ft["dtype"],
                    "shape": (len(arm_head_names),),
                    "names": arm_head_names,
                }
            continue

        # 其他字段（如 observation.images.* 已经处理，这里兜底保留）
        filtered[key] = ft

    return filtered


# ---------------------------------------------------------------------------
# 2. 动作合并
# ---------------------------------------------------------------------------


def merge_actions(
    leader_action: dict[str, float],
    head_action: dict[str, float],
    base_action: dict[str, float],
    observation: dict,
    action_features: dict[str, type],
) -> dict[str, float]:
    """合并 leader + head + base 动作，并对缺失字段做补齐。

    优先级：leader_action > head_action > base_action > observation 当前值 > 0.0。
    XleBiSO101Leader 已经自动填充 head/base 占位符为 0.0，因此当
    head_action / base_action 为空时，这些字段会保持为 0。
    """
    action = {**leader_action}

    # head/base 覆盖 leader 中的占位符（如果提供了）
    action.update(head_action)
    action.update(base_action)

    # 补齐缺失字段：.vel 用 0.0，.pos 用 observation 当前值
    for key in action_features:
        if key in action:
            continue
        if key.endswith(".vel"):
            action[key] = 0.0
        elif key.endswith(".pos"):
            action[key] = float(observation.get(key, 0.0))

    return action


# ---------------------------------------------------------------------------
# 3. 通用录制循环
# ---------------------------------------------------------------------------


def record_loop(
    robot,
    leader,
    events: dict,
    fps: int,
    control_time_s: float,
    dataset,
    single_task: str,
    display_data: bool,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    build_action: Callable,
    extra_quit_check: Callable | None = None,
    post_frame_callback: Callable | None = None,
):
    """单 episode 录制/遥操循环。

    Args:
        robot: XLerobotClient 实例。
        leader: XleBiSO101Leader 实例。
        events: 包含 ``exit_early``、``discard_current_episode``、
            ``stop_recording`` 等状态的字典。
        fps: 控制频率。
        control_time_s: 本 episode 最大时长（秒）。
        dataset: LeRobotDataset 实例，或 ``None``（重置阶段不记录）。
        single_task: 任务描述字符串，写入 dataset。
        display_data: 是否通过 rerun 可视化。
        *processors: LeRobot 默认处理管线。
        build_action: 回调函数 ``build_action(obs) -> action_dict``，
            由调用方定义如何合并 leader/head/base 动作。
        extra_quit_check: 可选回调，每帧调用；返回 True 则触发退出。
    """
    timestamp = 0.0
    start_episode_t = time.perf_counter()

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        # 全局键盘/joycon 监听器请求提前结束本 episode
        if events.get("exit_early"):
            events["exit_early"] = False
            break

        try:
            # 1. 先读取从端状态，保证日志和动作处理用同一帧观测
            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)

            observation_frame = None
            if dataset is not None:
                observation_frame = build_dataset_frame(
                    dataset.features, obs_processed, prefix="observation"
                )

            # 2. 由调用方定义如何构建完整动作
            act = build_action(obs)

            # 3. LeRobot 标准处理管线
            act_processed_teleop = teleop_action_processor((act, obs))
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
            sent_action = robot.send_action(robot_action_to_send)

            # 4. 写入数据集
            if dataset is not None and observation_frame is not None:
                action_frame = build_dataset_frame(
                    dataset.features, act_processed_teleop, prefix="action"
                )
                frame = {**observation_frame, **action_frame, "task": single_task}
                dataset.add_frame(frame)

            # 5. 可视化
            if display_data:
                log_rerun_data(observation=obs_processed, action=sent_action)

            # 6. 额外的退出检查（如键盘 quit 键）
            if extra_quit_check is not None and extra_quit_check():
                events["discard_current_episode"] = True
                events["stop_recording"] = True
                break

            # 7. 每帧后回调（如 Joy-Con episode 控制）
            if post_frame_callback is not None:
                post_frame_callback()

        except Exception as exc:
            logger.exception("Recording interrupted, discarding current episode: %s", exc)
            print(f"录制中断，舍弃本轮数据：{exc}")
            events["discard_current_episode"] = True
            events["stop_recording"] = True
            break

        # 维持目标频率
        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)
        timestamp = time.perf_counter() - start_episode_t


# ---------------------------------------------------------------------------
# 4. Episode 控制键盘监听器
# ---------------------------------------------------------------------------


class EpisodeKeyboardListener:
    """监听数字键 1/2/3/4，提供 episode 控制功能。

    按键映射：
        1 = 开始 / 跳过重置 并进入下一轮
        2 = 结束当前 episode
        3 = 重新录制当前 episode
        4 = 完全退出录制流程
    """

    def __init__(self):
        self.events = {
            "start_next": False,
            "end_current": False,
            "rerecord": False,
            "stop": False,
        }
        self._listener = None

    def start(self) -> None:
        from pynput import keyboard

        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _on_press(self, key) -> None:
        try:
            if hasattr(key, "char"):
                char = key.char
                if char == "1":
                    print("[KEYBOARD] 1: 开始/跳过重置")
                    self.events["start_next"] = True
                elif char == "2":
                    print("[KEYBOARD] 2: 结束当前 episode")
                    self.events["end_current"] = True
                elif char == "3":
                    print("[KEYBOARD] 3: 重新录制")
                    self.events["rerecord"] = True
                elif char == "4":
                    print("[KEYBOARD] 4: 完全退出")
                    self.events["stop"] = True
        except Exception:
            pass

    def consume_events(self) -> dict:
        """读取并清空当前事件状态。"""
        ev = self.events.copy()
        self.events = {k: False for k in self.events}
        return ev


# ---------------------------------------------------------------------------
# 5. 辅助函数
# ---------------------------------------------------------------------------


def sync_episode_events(src: dict, dst: dict) -> None:
    """把控制器检测到的 episode 事件同步到主 events 字典。

    映射规则：
        - start_next / end_current -> exit_early（提前结束当前阶段）
        - rerecord -> rerecord_episode + exit_early
        - stop -> stop_recording + exit_early
    """
    if src.get("start_next") or src.get("end_current"):
        dst["exit_early"] = True
    if src.get("rerecord"):
        dst["rerecord_episode"] = True
        dst["exit_early"] = True
    if src.get("stop"):
        dst["stop_recording"] = True
        dst["exit_early"] = True


def busy_wait(seconds: float) -> None:
    """在 Windows/macOS 上用忙等保持更稳定的控制频率；Linux 直接用 time.sleep。"""
    if platform.system() in {"Darwin", "Windows"}:
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            pass
    elif seconds > 0:
        time.sleep(seconds)


def make_round_prompt(
    session_episode_idx: int,
    session_total: int,
    dataset_episode_idx: int,
    *,
    rerecord: bool = False,
) -> str:
    action_text = "准备重录" if rerecord else "准备开始"
    return (
        f"{action_text}第 {session_episode_idx + 1}/{session_total} 轮，"
        f"数据集总第 {dataset_episode_idx + 1} 轮。"
    )


def clear_phase_exit_event(events: dict) -> None:
    """进入新阶段前清除 exit_early，避免上一阶段的按键误触发。"""
    events["exit_early"] = False


# ---------------------------------------------------------------------------
# 5. 主流程封装（两个脚本共用）
# ---------------------------------------------------------------------------


def run_recording_session(
    robot,
    leader,
    events: dict,
    dataset,
    args,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    build_action: Callable,
    extra_quit_check: Callable | None = None,
):
    """跑完整录制会话：episode 循环 + 重置阶段 + 重录/保存逻辑。

    与 ``record_remote_bi_so101_leader_keyboard.py`` 的主循环逻辑保持一致。
    """
    recorded_episodes = 0

    try:
        with VideoEncodingManager(dataset):
            while recorded_episodes < args.num_episodes and not events.get("stop_recording"):
                clear_phase_exit_event(events)
                prompt = make_round_prompt(
                    session_episode_idx=recorded_episodes,
                    session_total=args.num_episodes,
                    dataset_episode_idx=dataset.num_episodes,
                )
                print(prompt)
                log_say(prompt)

                # ---- 录制阶段 ----
                record_loop(
                    robot=robot,
                    leader=leader,
                    events=events,
                    fps=args.fps,
                    control_time_s=args.episode_time_s,
                    dataset=dataset,
                    single_task=args.task_description,
                    display_data=args.display_data,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    build_action=build_action,
                    extra_quit_check=extra_quit_check,
                )

                # ---- 重置阶段（如果还需要继续） ----
                if not events.get("stop_recording") and (
                    (recorded_episodes < args.num_episodes - 1)
                    or events.get("rerecord_episode")
                ):
                    clear_phase_exit_event(events)
                    print("进入重置阶段，请复位环境。")
                    log_say("Resetting environment")
                    record_loop(
                        robot=robot,
                        leader=leader,
                        events=events,
                        fps=args.fps,
                        control_time_s=args.reset_time_s,
                        dataset=None,
                        single_task=args.task_description,
                        display_data=args.display_data,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        build_action=build_action,
                        extra_quit_check=extra_quit_check,
                    )
                    clear_phase_exit_event(events)

                # ---- 处理重录 ----
                if events.get("rerecord_episode"):
                    rerecord_prompt = make_round_prompt(
                        session_episode_idx=recorded_episodes,
                        session_total=args.num_episodes,
                        dataset_episode_idx=dataset.num_episodes,
                        rerecord=True,
                    )
                    print(rerecord_prompt)
                    log_say(rerecord_prompt)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                # ---- 处理丢弃 ----
                if events.get("discard_current_episode"):
                    print("已舍弃当前轮数据，之前已保存轮次保持不变。")
                    log_say("Discard current episode")
                    events["discard_current_episode"] = False
                    dataset.clear_episode_buffer()
                    break

                # ---- 处理停止 ----
                if events.get("stop_recording"):
                    print("停止录制，舍弃当前未完成轮次，之前已保存数据保留。")
                    log_say("Discard unfinished episode")
                    dataset.clear_episode_buffer()
                    break

                # ---- 保存本 episode ----
                dataset.save_episode()
                print(
                    f"已保存第 {recorded_episodes + 1}/{args.num_episodes} 轮，"
                    f"数据集当前共有 {dataset.num_episodes} 轮。"
                )
                recorded_episodes += 1
    finally:
        log_say("Stopping recording")
