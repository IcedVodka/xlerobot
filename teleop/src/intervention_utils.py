#!/usr/bin/env python

"""
人类介入纠正采集公共工具模块

提供：
- 控制模式切换监听（autonomous / intervention）
- 根据当前模式决定发送 policy 动作还是遥操动作

control_mode 定义：
    0 = autonomous（策略执行）
    1 = intervention（人类接管）

每个 episode 从 autonomous(0) 开始，按 Space 单向切到 intervention(1)，
本 episode 内不再切回；下一 episode 通过 reset() 回到 autonomous。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from inference_utils import fill_missing_action_keys, run_policy_inference

if TYPE_CHECKING:
    import torch
    from lerobot.policies.pretrained import PreTrainedPolicy

logger = logging.getLogger(__name__)


class ModeToggleListener:
    """监听模式切换键，单向切到 intervention。

    按键映射：
        Space = 切到 intervention（单向，本 episode 内不再切回）

    模式含义：
        0 = autonomous（策略执行）
        1 = intervention（人类接管，policy 不再运行）

    每个 episode 开始时调用 reset() 回到 autonomous(0)。
    """

    MODES = [0, 1]
    MODE_NAMES = {
        0: "autonomous",
        1: "intervention",
    }

    def __init__(self, toggle_key: str = "space"):
        self._control_mode = 0
        self._toggle_key = toggle_key
        self._listener = None

    def start(self) -> None:
        from pynput import keyboard

        def on_press(key):
            if self._toggle_key == "space" and key == keyboard.Key.space:
                self._switch_to_intervention()
            elif self._toggle_key == "t" and hasattr(key, "char") and key.char == "t":
                self._switch_to_intervention()

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _switch_to_intervention(self) -> None:
        if self._control_mode == 1:
            return
        self._control_mode = 1
        print(f"[MODE] 切换到 {self.MODE_NAMES[1]} (1)")

    def reset(self) -> None:
        """回到 autonomous(0)，供每个 episode 开始时调用。"""
        self._control_mode = 0

    @property
    def control_mode(self) -> int:
        return self._control_mode


def decide_action(
    mode: int,
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    preprocessor,
    postprocessor,
    ds_features: dict,
    device: torch.device,
    teleop_action: dict[str, float],
    robot,
    task_description: str | None = None,
) -> dict[str, float]:
    """根据当前控制模式决定最终发送给机器人的动作。

    每个 episode 从 autonomous(0) 开始，按 Space 单向切到 intervention(1)，
    切入后本 episode 内不再切回，因此无需保持 policy 状态温热。

    Args:
        mode: 当前控制模式（0=autonomous, 1=intervention）。
        observation: 当前观测字典。
        policy: 已加载的策略模型。
        preprocessor: 观测预处理器。
        postprocessor: 动作后处理器。
        ds_features: 数据集 features 字典。
        device: 推理设备。
        teleop_action: 遥操作动作（人类动作）。
        robot: XLerobotClient 实例，用于补齐缺失动作字段。
        task_description: 任务描述。

    Returns:
        最终发送给机器人的完整动作字典。
    """
    if mode == 0:  # autonomous
        try:
            predicted_action = run_policy_inference(
                observation=observation,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                ds_features=ds_features,
                device=device,
                task=task_description,
                robot_type=robot.robot_type,
            )
            return fill_missing_action_keys(
                predicted_action=predicted_action,
                observation=observation,
                robot_action_features=robot.action_features,
            )
        except Exception as exc:
            logger.exception("Policy inference failed, fallback to teleop: %s", exc)
            print(f"[WARN] 推理失败，回退到遥操: {exc}")
            return teleop_action

    elif mode == 1:  # intervention
        return teleop_action

    else:
        raise ValueError(f"未知的 control_mode: {mode}")
