"""
VcdPrompt λ敏感性实验专用 Trainer。

不修改原始 `trainers/vcdprompt.py`，仅：
- 增加 3 个可控权重（每次只改一个参数时由命令行 opts 覆盖）：
  - TRAINER.VCD_LAMBDA.ALPHA      ：scloss 权重（原 self.alpha）
  - TRAINER.VCD_LAMBDA.DISTILL_W  ：distill 权重（原 self.lambd * loss_distill）
  - TRAINER.VCD_LAMBDA.VSD_W      ：vsd 权重（原 + loss_vsd）
"""

import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_pretrained_weights

# 复用原实现的绝大部分组件
from . import vcdprompt as base


class CustomCLIPLambda(base.CustomCLIP):
    def __init__(
        self,
        cfg,
        classnames,
        clip_model,
        clip_model_distill,
        clip_prompt_weights,
        distill_w: float,
        vsd_w: float,
    ):
        super().__init__(cfg, classnames, clip_model, clip_model_distill, clip_prompt_weights)
        self.distill_w = float(distill_w)
        self.vsd_w = float(vsd_w)

    def forward(self, image1, image2=None, label=None):
        # 基本结构与原 CustomCLIP.forward 一致，仅修改 loss 组合处的权重
        import random

        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        (
            prompts,
            shared_ctx,
            deep_compound_prompts_text,
            deep_compound_prompts_vision,
        ) = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)

        # 确保 image1 具有梯度（用于 saliency）
        image1 = image1.clone().detach().requires_grad_(True)

        image_features = self.image_encoder(image1.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        logits_raw = logit_scale * image_features @ text_features.t()

        x_a = self.adapter_image(image_features)
        image_features = self.image_adapter_m * x_a + (1 - self.image_adapter_m) * image_features

        x_b = self.adapter_text(text_features)
        text_features = self.text_adapter_m * x_b + (1 - self.text_adapter_m) * text_features

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            pre_trained_text_features = self.ori_embedding[random.randint(0, self.ori_embedding.shape[0] - 1)]
            original_trained_text_features = pre_trained_text_features.to(torch.float32)
            original_trained_text_features = original_trained_text_features / original_trained_text_features.norm(
                dim=-1, keepdim=True
            )
            pre_trained_text_features = self.IB(original_trained_text_features)

            pre_trained_image_features = self.model_distill.encode_image(image2)

            pre_trained_text_features = pre_trained_text_features / pre_trained_text_features.norm(dim=-1, keepdim=True)
            pre_trained_image_features = pre_trained_image_features / pre_trained_image_features.norm(
                dim=-1, keepdim=True
            )

            loss = F.cross_entropy(logits, label)

            # inner-loop: 计算梯度并生成 saliency map（去掉原版的 grad*.pt 保存，避免 I/O）
            loss.backward(retain_graph=True)
            gradients = image1.grad
            gradient = torch.sqrt(torch.mean(gradients**2, dim=1, keepdim=True)).detach()
            saliency = self.get_SBF_map(gradient, grid_size=3)

            if self.distill_criteria == "cosine":
                loss_vsd = F.kl_div(
                    input=F.softmax(text_features, dim=-1),
                    target=F.softmax(pre_trained_text_features, dim=-1),
                    reduction="batchmean",
                )

                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
                score = cos(text_features, pre_trained_text_features)
                loss_distill_text = 1.0 - torch.mean(score)
                score = cos(image_features, pre_trained_image_features)
                loss_distill_image = 1.0 - torch.mean(score)
                loss_distill = loss_distill_text + loss_distill_image
            else:
                loss_vsd = F.kl_div(
                    input=F.softmax(text_features, dim=-1),
                    target=F.softmax(pre_trained_text_features, dim=-1),
                    reduction="batchmean",
                )
                loss_distill = F.mse_loss(text_features, pre_trained_text_features) + F.mse_loss(
                    image_features, pre_trained_image_features
                )

            # outer loop
            mixed_img = image1.detach() * saliency + image2 * (1 - saliency)
            image_features2 = self.image_encoder(mixed_img.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
            x_a2 = self.adapter_image(image_features2)
            image_features2 = self.image_adapter_m * x_a2 + (1 - self.image_adapter_m) * image_features2
            image_features2 = image_features2 / image_features2.norm(dim=-1, keepdim=True)

            perturbed_logits = logit_scale * image_features2 @ text_features.t()
            loss2 = F.cross_entropy(perturbed_logits, label)

            # 关键：三个权重可控
            return loss2 + self.distill_w * loss_distill + self.vsd_w * loss_vsd

        return logits, image_features, text_features, label


@TRAINER_REGISTRY.register()
class VcdPromptLambda(base.VcdPrompt):
    """
    仅用于 few-shot Kather 的 λ 敏感性实验：
    - alpha：scloss 权重
    - distill_w：loss_distill 权重
    - vsd_w：loss_vsd 权重
    """

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = base.load_clip_to_cpu(cfg)

        print("Loading original CLIP for distillation")
        design_details = {
            "trainer": "CoOp",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
        }
        clip_model_distill = base.load_clip_to_cpu(cfg, design_details=design_details)

        if cfg.TRAINER.CoPrompt.PREC in ("fp32", "amp"):
            clip_model.float()
            clip_model_distill.float()

        with open(f"gpt_file/{base.dataset_name_mapping[cfg.DATASET.NAME]}_prompt.json") as f:
            gpt3_prompt = json.load(f)

        # alpha：默认沿用原 weights_mapping；若命令行覆盖为 >=0，则使用覆盖值
        alpha_cfg = float(getattr(cfg.TRAINER.VCD_LAMBDA, "ALPHA", -1.0))
        self.alpha = alpha_cfg if alpha_cfg >= 0 else base.weights_mapping[cfg.DATASET.NAME]

        # distill/vsd 权重
        distill_w = float(getattr(cfg.TRAINER.VCD_LAMBDA, "DISTILL_W", 1.0))
        vsd_w = float(getattr(cfg.TRAINER.VCD_LAMBDA, "VSD_W", 1.0))

        print("\nGetting textual features as CLIP's classifier.")
        clip_weights = base.gpt_clip_classifier(classnames, gpt3_prompt, clip_model_distill, cfg.DATASET.NAME)

        print("Building custom CLIP (lambda)")
        self.model = CustomCLIPLambda(
            cfg, classnames, clip_model, clip_model_distill, clip_weights, distill_w=distill_w, vsd_w=vsd_w
        )

        print("Turning off gradients in both the image and the text encoder")
        for _, param in self.model.named_parameters():
            param.requires_grad_(False)

        name_to_update = ["prompt_learner", "adapter"]
        for name, param in self.model.named_parameters():
            for n2u in name_to_update:
                if n2u in name:
                    param.requires_grad_(True)

        enabled = {name for name, p in self.model.named_parameters() if p.requires_grad}
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.CoPrompt.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        dataset_select = cfg.DATASET.NAME
        self.cls_num = base.dataset_clsnum_mapping.get(dataset_select)
        print("self.cls_num", self.cls_num)

