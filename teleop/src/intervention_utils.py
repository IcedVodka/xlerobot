#!/usr/bin/env python

"""
人类介入纠正采集公共工具模块

提供：
- 控制模式切换监听（autonomous / intervention / teleop_demo）
- 根据当前模式决定发送 policy 动作还是遥操动作

control_mode 定义：
    0 = autonomous（策略执行）
    1 = intervention（人类接管）
    2 = teleop_demo（纯遥操示教）
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
    """监听模式切换键，循环切换 control_mode。

    按键映射：
        Space = 按顺序切换 0 -> 1 -> 2 -> 0

    模式含义：
        0 = autonomous（策略执行）
        1 = intervention（人类接管，policy 可在后台继续运行保持状态）
        2 = teleop_demo（纯遥操示教，policy 不运行）
    """

    MODES = [0, 1, 2]
    MODE_NAMES = {
        0: "autonomous",
        1: "intervention",
        2: "teleop_demo",
    }

    def __init__(self, init_mode: int = 0, toggle_key: str = "space"):
        if init_mode not in self.MODES:
            raise ValueError(f"init_mode 必须是 {self.MODES} 之一，得到 {init_mode}")
        self._control_mode = init_mode
        self._toggle_key = toggle_key
        self._listener = None

    def start(self) -> None:
        from pynput import keyboard

        def on_press(key):
            if self._toggle_key == "space" and key == keyboard.Key.space:
                self._cycle_mode()
            elif self._toggle_key == "t" and hasattr(key, "char") and key.char == "t":
                self._cycle_mode()

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _cycle_mode(self) -> None:
        idx = self.MODES.index(self._control_mode)
        self._control_mode = self.MODES[(idx + 1) % len(self.MODES)]
        print(
            f"[MODE] 切换到 {self.MODE_NAMES[self._control_mode]} "
            f"({self._control_mode})"
        )

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
    keep_policy_warm: bool = True,
) -> dict[str, float]:
    """根据当前控制模式决定最终发送给机器人的动作。

    Args:
        mode: 当前控制模式（0/1/2）。
        observation: 当前观测字典。
        policy: 已加载的策略模型。
        preprocessor: 观测预处理器。
        postprocessor: 动作后处理器。
        ds_features: 数据集 features 字典。
        device: 推理设备。
        teleop_action: 遥操作动作（人类动作）。
        robot: XLerobotClient 实例，用于补齐缺失动作字段。
        task_description: 任务描述。
        keep_policy_warm: intervention 期间是否继续把观测喂给 policy，
            以保持其内部历史队列连续。建议对 ACT/Diffusion/π0/VQ-BeT 等
            带状态队列的策略开启。teleop_demo 模式下不运行 policy。

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
        # 可选：保持 policy 状态温热
        if keep_policy_warm:
            try:
                run_policy_inference(
                    observation=observation,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    ds_features=ds_features,
                    device=device,
                    task=task_description,
                    robot_type=robot.robot_type,
                )
            except Exception as exc:
                logger.warning("Policy warm-up inference failed during intervention: %s", exc)
        return teleop_action

    elif mode == 2:  # teleop_demo
        return teleop_action

    else:
        raise ValueError(f"未知的 control_mode: {mode}")
