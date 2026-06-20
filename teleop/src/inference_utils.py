#!/usr/bin/env python

"""
XLerobot 双臂策略推理公共工具模块

从 ``teleop/scripts/infer_bimanual_policy.py`` 中抽取的可复用逻辑：
- 从训练配置推断 dataset_repo_id
- 加载预训练策略、preprocessor、postprocessor
- 单步策略推理
- 动作补齐

该模块保持与现有推理脚本行为一致，供新脚本组合使用。
"""

from __future__ import annotations

import json as json_module
import logging
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device

logger = logging.getLogger(__name__)


def infer_dataset_repo_id(model_path: str) -> str | None:
    """从训练配置中自动推断数据集 repo_id。

    LeRobot 训练时会在 ``pretrained_model/train_config.json`` 中保存
    ``dataset.repo_id``。如果找到就返回，否则返回 ``None``。
    """
    train_config_path = Path(model_path) / "train_config.json"
    if not train_config_path.exists():
        return None
    try:
        with open(train_config_path) as f:
            cfg = json_module.load(f)
        repo_id = cfg.get("dataset", {}).get("repo_id")
        if repo_id:
            print(f"[INFO] 从 train_config.json 自动推断 dataset_repo_id: {repo_id}")
            return repo_id
    except Exception:
        pass
    return None


def fill_missing_action_keys(
    predicted_action: dict[str, float],
    observation: dict[str, np.ndarray],
    robot_action_features: dict[str, type],
) -> dict[str, float]:
    """将模型预测的动作补齐为完整的机器人动作字典。

    策略：
    - 模型已预测的字段 → 直接使用
    - 缺失的 ``.pos`` 字段 → 使用观测中的当前值（保持不动）
    - 缺失的 ``.vel`` 字段 → 置为 0.0（停止运动）
    """
    full_action: dict[str, float] = {}

    for key in robot_action_features:
        if key in predicted_action:
            full_action[key] = predicted_action[key]
        elif key.endswith(".vel"):
            full_action[key] = 0.0
        elif key.endswith(".pos"):
            val = observation.get(key, 0.0)
            full_action[key] = float(val) if val is not None else 0.0
        else:
            full_action[key] = 0.0

    return full_action


def load_policy_and_processors(
    model_path: str,
    dataset_repo_id: str | None = None,
    dataset_root: str | None = None,
    device_arg: str = "auto",
) -> tuple[PreTrainedPolicy, object, object, dict, torch.device]:
    """加载预训练策略及其 preprocessor / postprocessor。

    Args:
        model_path: 预训练模型路径或 HuggingFace Hub ID。
        dataset_repo_id: 数据集 repo_id，未指定时从 train_config.json 推断。
        dataset_root: 数据集本地根目录。
        device_arg: 推理设备，``auto`` 时自动检测 cuda/mps/cpu。

    Returns:
        (policy, preprocessor, postprocessor, ds_features, device)
    """
    print(f"[INFO] 加载模型: {model_path}")

    # 1. 读取配置，自动识别策略类型
    policy_cfg = PreTrainedConfig.from_pretrained(model_path)
    policy_type = policy_cfg.type
    print(f"[INFO] 策略类型: {policy_type}")

    # 2. 动态获取策略类并加载
    PolicyClass = get_policy_class(policy_type)
    policy: PreTrainedPolicy = PolicyClass.from_pretrained(model_path)

    # 3. 自动检测设备
    if device_arg == "auto":
        device = get_safe_torch_device(policy.config.device)
    else:
        device = torch.device(device_arg)
    policy.to(device)
    policy.eval()

    # 4. 打印模型信息
    action_shape = policy.config.output_features["action"].shape
    action_names = getattr(policy.config.output_features["action"], "names", None)
    print(f"[INFO] 模型设备: {device}")
    print(f"[INFO] 模型输出 action shape: {action_shape}")
    if action_names:
        print(f"[INFO] 模型输出 action 字段 ({len(action_names)}个):")
        for name in action_names:
            print(f"       - {name}")

    # 5. 加载数据集 metadata（用于 preprocessor / postprocessor）
    inferred_repo_id = infer_dataset_repo_id(model_path)
    if dataset_repo_id is None:
        dataset_repo_id = inferred_repo_id
    if dataset_repo_id is None:
        raise ValueError(
            "无法自动推断 dataset_repo_id。请显式指定:\n"
            "  --dataset_repo_id=<your_dataset_name>"
        )

    print(f"[INFO] 加载数据集 metadata: {dataset_repo_id}")
    dataset_metadata = LeRobotDatasetMetadata(
        dataset_repo_id,
        root=dataset_root,
    )
    ds_features = dataset_metadata.features
    print(f"[INFO] 数据集 fps: {dataset_metadata.fps}")

    # 6. 构建 preprocessor / postprocessor
    print("[INFO] 构建 preprocessor / postprocessor...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=model_path,
        dataset_stats=dataset_metadata.stats,
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
        },
    )

    return policy, preprocessor, postprocessor, ds_features, device


def run_policy_inference(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    preprocessor,
    postprocessor,
    ds_features: dict,
    device: torch.device,
    task: str | None = None,
    robot_type: str | None = None,
) -> dict[str, float]:
    """执行单步策略推理，返回命名动作字典（可能缺少部分字段）。

    Args:
        observation: 当前观测字典。
        policy: 已加载的策略模型。
        preprocessor: 观测预处理器。
        postprocessor: 动作后处理器。
        ds_features: 数据集 features 字典。
        device: 推理设备。
        task: 任务描述。
        robot_type: 机器人类型。

    Returns:
        模型预测的动作字典（可能缺少部分字段，需用 ``fill_missing_action_keys`` 补齐）。
    """
    from lerobot.datasets.utils import build_dataset_frame

    obs_frame = build_dataset_frame(ds_features, observation, prefix=OBS_STR)

    action_tensor = predict_action(
        observation=obs_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=policy.config.use_amp,
        task=task,
        robot_type=robot_type,
    )

    return make_robot_action(action_tensor, ds_features)
