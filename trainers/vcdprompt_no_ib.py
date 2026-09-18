"""
VcdPrompt 消融：去掉 IB（Information Bottleneck）模块，蒸馏分支上直接使用原始预训练文本特征。

训练出的权重用于与完整 VcdPrompt 对比（可视化 / 论文消融）。
"""
import json
import os.path as osp

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from dassl.engine import TRAINER_REGISTRY
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_pretrained_weights

from .vcdprompt import (
    VcdPrompt,
    CustomCLIP,
    load_clip_to_cpu,
    gpt_clip_classifier,
    dataset_name_mapping,
    weights_mapping,
    dataset_clsnum_mapping,
)


class CustomCLIPNoIB(CustomCLIP):
    """与 CustomCLIP 相同，仅将 IBMLPNetwork 换为 Identity（不经过 IB 变换）。"""

    def __init__(self, cfg, classnames, clip_model, clip_model_distill, clip_prompt_weights):
        super().__init__(cfg, classnames, clip_model, clip_model_distill, clip_prompt_weights)
        self.IB = nn.Identity()
        print("[VcdPromptNoIB] self.IB -> nn.Identity()（w/o IB 消融）")


@TRAINER_REGISTRY.register()
class VcdPromptNoIB(VcdPrompt):
    """VcdPrompt，无 IB 模块；其余训练逻辑与 VcdPrompt 一致。"""

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"[VcdPromptNoIB] Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        print("Loading original CLIP for distillation")
        design_details = {
            "trainer": "CoOp",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
        }
        clip_model_distill = load_clip_to_cpu(cfg, design_details=design_details)

        if cfg.TRAINER.CoPrompt.PREC == "fp32" or cfg.TRAINER.CoPrompt.PREC == "amp":
            clip_model.float()
            clip_model_distill.float()

        _ap = osp.dirname(osp.dirname(osp.abspath(__file__)))
        gpt_path = osp.join(
            _ap,
            "gpt_file",
            f"{dataset_name_mapping[cfg.DATASET.NAME]}_prompt.json",
        )
        with open(gpt_path) as f:
            gpt3_prompt = json.load(f)

        self.alpha = weights_mapping[cfg.DATASET.NAME]
        print("\nGetting textual features as CLIP's classifier.")
        clip_weights = gpt_clip_classifier(
            classnames, gpt3_prompt, clip_model_distill, cfg.DATASET.NAME
        )

        print("Building CustomCLIPNoIB (no IB)")
        self.model = CustomCLIPNoIB(
            cfg, classnames, clip_model, clip_model_distill, clip_weights
        )

        print("Turning off gradients in both the image and the text encoder")
        for _, param in self.model.named_parameters():
            param.requires_grad_(False)

        name_to_update = ["prompt_learner", "adapter"]
        for name, param in self.model.named_parameters():
            for n2u in name_to_update:
                if n2u in name:
                    param.requires_grad_(True)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(
            "MultiModalPromptLearner", self.model, self.optim, self.sched
        )

        self.scaler = GradScaler() if cfg.TRAINER.CoPrompt.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        dataset_select = cfg.DATASET.NAME
        self.cls_num = dataset_clsnum_mapping.get(dataset_select)
        print("self.cls_num", self.cls_num)
