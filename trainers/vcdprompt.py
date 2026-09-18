import copy
import json
import os.path as osp
import random
import os 
import torch
import torch.nn as nn
import time
from torch.nn.functional import kl_div
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
import datetime
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint, load_pretrained_weights
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch.nn.functional import kl_div
from .utils import SinkhornDistance
from tqdm import tqdm
_tokenizer = _Tokenizer()

dataset_name_mapping = {
    "Kather": "kather",
    "colorectal": "colorectal",
    'PathMNIST':"pathminst",
    'bloodmnist':"bloodmnist",
    "tissuemnist" : "tissuemnist",
    "organamnist" : "organamnist",
    "KIMIA" : "KIMIA",
    "pannuke" : "pannuke",
    "pannuke_adrenal_gland": "pannuke",
    "pannuke_bile_duct": "pannuke",
    "pannuke_bladder": "pannuke",
    "pannuke_breast": "pannuke",
    "pannuke_cervix": "pannuke",
    "pannuke_colon": "pannuke",
    "pannuke_esophagus": "pannuke",
    "pannuke_headneck": "pannuke",
    "pannuke_kidney": "pannuke",
    "pannuke_liver": "pannuke",
    "pannuke_lung": "pannuke",
    "pannuke_ovarian": "pannuke",
    "pannuke_pancreatic": "pannuke",
    "pannuke_prostate": "pannuke",
    "pannuke_skin": "pannuke",
    "pannuke_stomach": "pannuke",
    "pannuke_testis": "pannuke",
    "pannuke_thyroid": "pannuke",
    "pannuke_uterus": "pannuke"
}

CUSTOM_TEMPLATES = {
    "Kather": "a photo of a {}.",
    "colorectal": "a photo of a {}.",
    'PathMNIST': "a photo of a {}.",
    "bloodmnist": "a photo of a {}.",
    "tissuemnist": "a photo of a {}.",
    "organamnist": "a photo of a {}.",
    "KIMIA" :"a photo of a {}.",
    "pannuke" : "An H&E image of {} tissue.",
    "pannuke_adrenal_gland" : "An H&E image of {} tissue.",
    "pannuke_adrenal_gland" : "An H&E image of {} tissue.",
    "pannuke_bile_duct" : "An H&E image of {} tissue.",
    "pannuke_bladder" : "An H&E image of {} tissue.",
    "pannuke_breast" : "An H&E image of {} tissue.",
    "pannuke_cervix" : "An H&E image of {} tissue.",
    "pannuke_colon" : "An H&E image of {} tissue.",
    "pannuke_esophagus" : "An H&E image of {} tissue.",
    "pannuke_headneck" : "An H&E image of {} tissue.",
    "pannuke_kidney" : "An H&E image of {} tissue.",
    "pannuke_liver" : "An H&E image of {} tissue.",
    "pannuke_lung" : "An H&E image of {} tissue.",
    "pannuke_ovarian" : "An H&E image of {} tissue.",
    "pannuke_pancreatic" : "An H&E image of {} tissue.",
    "pannuke_prostate" : "An H&E image of {} tissue.",
    "pannuke_skin" : "An H&E image of {} tissue.",
    "pannuke_stomach" : "An H&E image of {} tissue.",
    "pannuke_testis" : "An H&E image of {} tissue.",
    "pannuke_thyroid" : "An H&E image of {} tissue.",
    "pannuke_uterus" : "An H&E image of {} tissue.",

}

dataset_clsnum_mapping = {
    "Kather": 6,
    "colorectal": 6,
    "PathMNIST": 6,
    "bloodmnist": 6,
    "tissuemnist": 6 ,
    "organamnist": 6 ,
    "KIMIA" : 9,
    "pannuke" : 2,
    "pannuke_adrenal_gland": 2
}
weights_mapping = {
    "Kather": 0.1,
    "colorectal": 0.1,
    "PathMNIST": 0.1,
    "bloodmnist": 0.1,
    "tissuemnist": 0.1 ,
    "organamnist": 0.1 ,
    "KIMIA" : 0.1,
    "pannuke" : 0,
    "pannuke_adrenal_gland": 0.1,
    "pannuke_bile_duct": 0.1,
    "pannuke_bladder": 0.1,
    "pannuke_breast": 0.1,
    "pannuke_cervix": 0.1,
    "pannuke_colon": 0.1,
    "pannuke_esophagus": 0.1,
    "pannuke_headneck": 0.1,
    "pannuke_kidney": 0.1,
    "pannuke_liver": 0.1,
    "pannuke_lung": 0.1,
    "pannuke_ovarian": 0.1,
    "pannuke_pancreatic": 0.1,
    "pannuke_prostate": 0.1,
    "pannuke_skin": 0.1,
    "pannuke_stomach": 0.1,
    "pannuke_testis": 0.1,
    "pannuke_thyroid": 0.1,
    "pannuke_uterus": 0.1,

}


def cosine_loss(student_embedding, teacher_embedding):
    return (1 - F.cosine_similarity(student_embedding, teacher_embedding)).mean()


def load_clip_to_cpu(cfg, design_details=None):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    # 预训练权重：<repo>/pretrained/plip/plip_vit_b32.pt
    _repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    model_path = os.path.join(_repo_root, "pretrained", "plip", "plip_vit_b32.pt")
    #model = torch.jit.load(model_path, map_location="cpu").eval()
    state_dict = torch.load(model_path, map_location="cpu")


    if design_details is None:
        design_details = {
            "trainer": "CoPrompt",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
            "maple_length": cfg.TRAINER.CoPrompt.N_CTX,
        }
    model = clip.build_model(state_dict, design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [
            x,
            compound_prompts_deeper_text,
            0,
        ]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, clip_model_distill=None):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.CoPrompt.N_CTX
        ctx_init = cfg.TRAINER.CoPrompt.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert (
            cfg.TRAINER.CoPrompt.PROMPT_DEPTH >= 1
        ), "For CoPrompt, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = (
            cfg.TRAINER.CoPrompt.PROMPT_DEPTH
        )  # max=12, but will create 11 such shared prompts
        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print("CoPrompt design: Multi-modal Prompt Learning")
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of CoPrompt context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        if dtype == torch.float16:
            self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow CoPrompt
        # compound prompts
        self.compound_prompts_text = nn.ParameterList(
            [
                nn.Parameter(torch.empty(n_ctx, 512))
                for _ in range(self.compound_prompts_depth - 1)
            ]
        )
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(
            single_layer, self.compound_prompts_depth - 1
        )

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        clip_model_ = clip_model_distill
        if cfg.TRAINER.CoPrompt.PREC == "fp32" or cfg.TRAINER.CoPrompt.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model_.float()
        if torch.cuda.is_available():
            clip_model_.cuda()

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts_}")
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        if torch.cuda.is_available():
            prompts_ = prompts_.cuda()

        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return (
            prompts,
            self.proj(self.ctx),
            self.compound_prompts_text,
            visual_deep_prompts,
        )  # pass here original, as for visual 768 is required


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
import numpy as np
import torch
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn.functional as F

class MemoryBank:
    def __init__(self, num_labeled_samples, cls_train):
        self.memory_img_feature = [None] * num_labeled_samples  # 存储图像特征
        self.memory_label = [None] * num_labeled_samples  # 存储标签
        self.num_cls = cls_train
        self.current_index = 0  # 初始化存储位置指针
        
        self.memory_text_feature = None  # 文本特征只存储一次，不需要为每个样本都存储
        self.text_feature_updated = False  # 标记文本特征是否已更新

    def update_labeled_feature(self, img_feature, text_feature, label):
        # 更新图像特征
        self.memory_img_feature[self.current_index] = img_feature  # 保持在计算图内
        # 更新文本特征
        if not self.text_feature_updated:
            self.memory_text_feature = text_feature  # 只更新一次文本特征
            self.text_feature_updated = True  # 标记为已更新
        # 更新标签
        self.memory_label[self.current_index] = label
        # 更新指针位置 (循环队列)
        self.current_index = (self.current_index + 1) % len(self.memory_img_feature)

    def _flatten_img_feats_and_labels(self):
        """
        forward_feature 返回图像特征 [B, D]；memory 按 batch 存储。
        展开为逐样本 [N, D] 与 [N]，供原型与 SCC 使用。
        """
        flat_feats: list = []
        flat_labels: list = []
        feature_list = self.memory_img_feature
        for i in range(len(self.memory_label)):
            if self.memory_label[i] is None:
                continue
            feat = feature_list[i]
            lab = self.memory_label[i]
            if feat is None:
                continue
            if feat.dim() == 1:
                flat_feats.append(feat)
                flat_labels.append(lab.view(()) if lab.dim() > 0 else lab)
            elif feat.dim() == 2:
                bsz = feat.shape[0]
                if lab.dim() == 0:
                    if bsz != 1:
                        continue
                    flat_feats.append(feat[0])
                    flat_labels.append(lab)
                else:
                    lab_row = lab.reshape(-1)
                    if lab_row.numel() != bsz:
                        continue
                    for j in range(bsz):
                        flat_feats.append(feat[j])
                        flat_labels.append(lab_row[j].view(()))
            else:
                continue

        if len(flat_feats) == 0:
            return None, None
        tmp_feature_list = torch.stack(flat_feats)
        tmp_label_list = torch.stack(
            [x.long().view(()) for x in flat_labels]
        ).to(device=tmp_feature_list.device)
        return tmp_feature_list, tmp_label_list

    def compute_class_prototypes(self):
        tmp_feature_list, tmp_label_list = self._flatten_img_feats_and_labels()
        if tmp_feature_list is None:
            raise ValueError("MemoryBank is empty, no image features to compute mean.")

        mean_features = []
        feat_dim = tmp_feature_list.size(-1)

        for c in range(self.num_cls):
            mask_c = tmp_label_list == c
            features_c = tmp_feature_list[mask_c]
            if features_c.size(0) > 0:
                mean_feature_c = torch.mean(features_c, dim=0)
                mean_features.append(mean_feature_c)
            else:
                mean_features.append(
                    torch.zeros(
                        feat_dim,
                        device=tmp_feature_list.device,
                        dtype=tmp_feature_list.dtype,
                    )
                )

        mean_features = torch.stack(mean_features)
        return mean_features
    
    def compute_sdc_loss(self, prototypes, distance_type='euclidean'):
        """
        计算类原型之间的距离损失。
        :param prototypes: (class, 512) tensor (从 NumPy 转换为 PyTorch 张量)
        :param distance_type: 距离类型，'mse', 'euclidean' 或 'cosine'
        :return: sdc_loss
        """
        sdc_loss = 0
        num_classes = prototypes.size(0)  # 类的数量

        # 遍历所有类原型对
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                if distance_type == 'mse':
                    # 均方误差 (MSE)
                    distance = torch.mean((prototypes[i] - prototypes[j]) ** 2)
                elif distance_type == 'euclidean':
                    # 欧氏距离
                    distance = torch.norm(prototypes[i] - prototypes[j], p=2)  # 使用 PyTorch 的 norm
                elif distance_type == 'cosine':
                    # 余弦相似度：1 - 余弦相似度（cosine similarity）
                    cosine_similarity = torch.nn.functional.cosine_similarity(prototypes[i], prototypes[j], dim=0)
                    distance = 1 - cosine_similarity  # 距离 = 1 - 相似度
                else:
                    raise ValueError("Unsupported distance type. Choose 'mse', 'euclidean', or 'cosine'.")

                #print(f"Distance between class {i} and class {j} ({distance_type}): {distance.item()}")
                sdc_loss += distance  # 累加距离
            
        #print("sdc_loss", sdc_loss.item())
        return sdc_loss
    
    def compute_scc_loss(self, prototypes, distance_type='euclidean'):
        """
        计算每个类的所有特征到该类原型的距离，生成类内紧凑性损失 (Self Class Compactness Loss, SCC Loss)。
        :param prototypes: (class, 512) tensor (从 NumPy 转换为 PyTorch 张量)
        :param distance_type: 距离类型，'mse', 'euclidean' 或 'cosine'
        :return: scc_loss
        """
        scc_loss = 0
        tmp_feature_list, tmp_label_list = self._flatten_img_feats_and_labels()
        if tmp_feature_list is None:
            raise ValueError("MemoryBank is empty, no features to compute SCC loss.")

        num_classes = prototypes.size(0)  # 类的数量

        # 遍历每个类，计算类内特征与原型的距离
        for c in range(num_classes):
            mask_c = (tmp_label_list == c)  # 找出属于类别 c 的所有特征
            features_c = tmp_feature_list[mask_c]

            if features_c.size(0) > 0:
                # 对每个特征计算与类原型的距离
                for feature in features_c:
                    if distance_type == 'mse':
                        # 均方误差 (MSE)
                        distance = torch.mean((feature - prototypes[c]) ** 2)
                    elif distance_type == 'euclidean':
                        # 欧氏距离
                        distance = torch.norm(feature - prototypes[c], p=2)  # 使用 PyTorch 的 norm
                    elif distance_type == 'cosine':
                        # 余弦相似度：1 - 余弦相似度（cosine similarity）
                        cosine_similarity = torch.nn.functional.cosine_similarity(feature, prototypes[c], dim=0)
                        distance = 1 - cosine_similarity  # 距离 = 1 - 相似度
                    else:
                        raise ValueError("Unsupported distance type. Choose 'mse', 'euclidean', or 'cosine'.")

                    #print(f"Distance from feature to class {c} prototype ({distance_type}): {distance.item()}")
                    scc_loss += distance  # 累加距离

        #print("scc_loss", scc_loss.item())
        return scc_loss

    def semantic_relationship_optimization(self):
        img_prototypes = self.compute_class_prototypes()

        # Compute SCC and SDC losses
        #scc_loss = self.compute_scc_loss(prototypes)
        img_loss = self.compute_sdc_loss(img_prototypes)
        text_loss = self.compute_sdc_loss(self.memory_text_feature)
        
        # Total SRO loss
        #sro_loss =  -sdc_loss*10 + scc_loss
        sro_loss =  -img_loss-text_loss
        return sro_loss 

class IBNetwork(nn.Module):
    def __init__(self, input_dim=512, bottleneck_dim=256):
        super(IBNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=bottleneck_dim, kernel_size=1)  # 压缩
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=bottleneck_dim, out_channels=input_dim, kernel_size=1)  # 恢复

    def forward(self, x):
        # x.shape: [num, cls, 512] -> [num, 512, cls] (转置)
        x = x.transpose(1, 2)
        x = self.conv1(x)  # [num, bottleneck_dim, cls]
        x = self.relu(x)
        x = self.conv2(x)  # [num, 512, cls]
        x = x.transpose(1, 2)  # 转置回 [num, cls, 512]
        return x

class IBMLPNetwork(nn.Module):
    def __init__(self, input_dim=512, bottleneck_dim=256):
        super(IBMLPNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, bottleneck_dim)  # 压缩
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(bottleneck_dim, input_dim)  # 恢复

    def forward(self, x):
        # x.shape: [cls, 512]
        x = self.fc1(x)  # [cls, bottleneck_dim]
        x = self.relu(x)
        x = self.fc2(x)  # [cls, 512]
        return x
    
class CustomCLIP(nn.Module):
    def __init__(
        self, cfg, classnames, clip_model, clip_model_distill, clip_prompt_weights
    ):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(
            cfg, classnames, clip_model, clip_model_distill
        )
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.ori_embedding = clip_prompt_weights
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.distill_criteria = cfg.TRAINER.DISTILL
        self.model_distill = clip_model_distill
        self.lambd = cfg.TRAINER.W
        self.adapter_image = Adapter(512, 4).to(clip_model.dtype)
        self.adapter_text = Adapter(512, 4).to(clip_model.dtype)
        self.image_adapter_m = 0.1
        self.text_adapter_m = 0.2
        
        self.W_dist = SinkhornDistance().cuda()
        self.softmax = torch.nn.Softmax(dim=1)
        
        
        self.IB = IBMLPNetwork()
        self.logit_scale = clip_model.logit_scale
        
    def forward_feature(self, image1, label=None):
        
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        (
            prompts,
            shared_ctx,
            deep_compound_prompts_text,
            deep_compound_prompts_vision,
        ) = self.prompt_learner()
        text_features = self.text_encoder(
        prompts, tokenized_prompts, deep_compound_prompts_text
        )
        image_features = self.image_encoder(
            image1.type(self.dtype), shared_ctx, deep_compound_prompts_vision
        )

        x_a = self.adapter_image(image_features)
        image_features = (
            self.image_adapter_m * x_a + (1 - self.image_adapter_m) * image_features
        )

        x_b = self.adapter_text(text_features)
        text_features = (
            self.text_adapter_m * x_b + (1 - self.text_adapter_m) * text_features
        )

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return image_features,text_features
#     def get_SBF_map(self, gradient, grid_size):
#         """
#         计算 SBF Saliency Map
#         """
#         # 平滑梯度
#         saliency = F.adaptive_avg_pool2d(gradient, grid_size)
#         bs_kernel = torch.ones((1, 1, 3, 3), device=gradient.device) / 9.0  # 3x3 平滑卷积核
#         bs_pad = 1

#         saliency = F.conv_transpose2d(saliency, bs_kernel, padding=bs_pad, stride=gradient.shape[-1] // grid_size)
#         saliency = F.interpolate(saliency, size=(gradient.shape[-2], gradient.shape[-1]), mode='bilinear', align_corners=True)

#         # 归一化到 [0, 1]
#         saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

#         return saliency

    def get_SBF_map(self, gradient, grid_size):
        """
        复杂数学公式版本的 SBF Saliency Map 计算
        """
        B, C, H, W = gradient.shape  # 获取 batch 维度

        # **(1) 计算梯度幅值 (Gradient Magnitude)**
        S = torch.sqrt(torch.mean(gradient ** 2, dim=1, keepdim=True))  # 形状 (B, 1, H, W)

        # **(2) 非均匀加权平滑**
        h_idx = torch.arange(H, device=gradient.device).float().view(1, 1, H, 1)
        w_idx = torch.arange(W, device=gradient.device).float().view(1, 1, 1, W)

        # 计算一个高斯加权窗口
        sigma_h = H / (2 * grid_size)
        sigma_w = W / (2 * grid_size)
        weight_h = torch.exp(-((h_idx - H / 2) ** 2) / (2 * sigma_h ** 2))
        weight_w = torch.exp(-((w_idx - W / 2) ** 2) / (2 * sigma_w ** 2))
        weight = weight_h * weight_w  # 形状 (1, 1, H, W)

        # 计算加权均值
        S_smooth = (S * weight).sum(dim=(2, 3), keepdim=True) / weight.sum(dim=(2, 3), keepdim=True)

        # **(3) 基于权重的重构**
        W_reconstruction = torch.exp(-((h_idx - H / 4) ** 2 + (w_idx - W / 4) ** 2) / (2 * sigma_h ** 2))
        S_reconstructed = S_smooth * W_reconstruction

        # **(4) 使用局部拉格朗日插值计算插值结果**
        S_interpolated = S_reconstructed * (1 - (h_idx / H)) * (1 - (w_idx / W)) + \
                         S_smooth * (h_idx / H) * (1 - (w_idx / W)) + \
                         S_smooth * (1 - (h_idx / H)) * (w_idx / W) + \
                         S_smooth * (h_idx / H) * (w_idx / W)

        # **(5) 归一化到 [0, 1]**
        saliency = (S_interpolated - S_interpolated.amin(dim=(2, 3), keepdim=True)) / \
                   (S_interpolated.amax(dim=(2, 3), keepdim=True) - S_interpolated.amin(dim=(2, 3), keepdim=True) + 1e-8)

        return saliency

    
    def forward(self, image1, image2=None, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        (
            prompts,
            shared_ctx,
            deep_compound_prompts_text,
            deep_compound_prompts_vision,
        ) = self.prompt_learner()
        text_features = self.text_encoder(
            prompts, tokenized_prompts, deep_compound_prompts_text
        )
        
        # **确保 image1 具有梯度**
        image1 = image1.clone().detach().requires_grad_(True)
        
        
        image_features = self.image_encoder(
            image1.type(self.dtype), shared_ctx, deep_compound_prompts_vision
        )
        #print("image_features",image_features.shape)

        #MVD  
        logits_raw = logit_scale * image_features @ text_features.t()
        
        
        x_a = self.adapter_image(image_features)
        image_features = (
            self.image_adapter_m * x_a + (1 - self.image_adapter_m) * image_features
        )

        x_b = self.adapter_text(text_features)
        text_features = (
            self.text_adapter_m * x_b + (1 - self.text_adapter_m) * text_features
        )

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.t()
        temperature = 1.0
        if self.prompt_learner.training:
            # vcd_loss = F.kl_div(
            #     input=F.log_softmax(logits_raw.detach() / temperature, dim=-1),
            #     target=F.softmax(logits / temperature, dim=-1),
            #     reduction='batchmean'  # 避免结果过大或过小
            # )
            # print('vcd_loss',vcd_loss)
            
            conventional_ML = self.W_dist(self.softmax(logits_raw), self.softmax(logits))
            #print("conventional_ML",conventional_ML)
            
            # print("self.ori_embedding.shape",self.ori_embedding.shape)
            pre_trained_text_features = self.ori_embedding[
                random.randint(0, self.ori_embedding.shape[0] - 1)
            ]
            original_trained_text_features = pre_trained_text_features.to(torch.float32)
            
            original_trained_text_features = (
                original_trained_text_features
                / original_trained_text_features.norm(dim=-1, keepdim=True)
            )
            
            pre_trained_text_features = self.IB(original_trained_text_features)
            
            pre_trained_image_features = self.model_distill.encode_image(image2)
            
            pre_trained_text_features = (
                pre_trained_text_features
                / pre_trained_text_features.norm(dim=-1, keepdim=True)
            )
            pre_trained_image_features = (
                pre_trained_image_features
                / pre_trained_image_features.norm(dim=-1, keepdim=True)
            )

            loss = F.cross_entropy(logits, label)
            
            #innerloop-calculate
            loss.backward(retain_graph=True)  # 反向传播计算梯度
            gradients = image1.grad
            # 保存梯度为 .pt 文件
            torch.save(gradients, 'grad1.pt')
            
            #innerloop-gradient
            #gradients = torch.autograd.grad(outputs=loss, inputs=image1, retain_graph=True, create_graph=False)[0]
            
            
            # **计算梯度并生成 saliency map**
            gradient = torch.sqrt(torch.mean(gradients ** 2, dim=1, keepdim=True)).detach()
            #print("gradient",gradient.shape)
            saliency = self.get_SBF_map(gradient, grid_size=3)  # 生成 SBF Saliency Map
            torch.save(gradients, 'grad2.pt')

            if self.distill_criteria == "cosine":
                    # 使用 KL 散度作为损失
                loss_kl_text = F.kl_div(
                    input=F.softmax(text_features, dim=-1),
                    target=F.softmax(pre_trained_text_features, dim=-1),
                    reduction='batchmean'
                )
                
                
                loss_vsd = F.kl_div(
                    input=F.softmax(text_features, dim=-1),
                    target=F.softmax(pre_trained_text_features, dim=-1),
                    reduction='batchmean'
                )
                
                
                loss_kl_image = F.kl_div(
                    input=F.log_softmax(image_features, dim=-1),
                    target=F.softmax(pre_trained_image_features, dim=-1),
                    reduction='batchmean'
                )
                #这里cos改成kl就是MV2D了？
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
                score = cos(text_features, pre_trained_text_features)
                loss_distill_text = 1.0 - torch.mean(score)

                score = cos(image_features, pre_trained_image_features)
                loss_distill_image = 1.0 - torch.mean(score)

                loss_distill = loss_distill_text + loss_distill_image# + loss_kl_text 
            else:
                loss_distill = F.mse_loss(
                    text_features, pre_trained_text_features
                ) + F.mse_loss(image_features, pre_trained_image_features)
               
            ###Outer Loop
            mixed_img = image1.detach() * saliency + image2 * (1 - saliency)
            
            image_features = self.image_encoder(
            mixed_img.type(self.dtype), shared_ctx, deep_compound_prompts_vision
            )
            x_a = self.adapter_image(image_features)
            image_features = (
                self.image_adapter_m * x_a + (1 - self.image_adapter_m) * image_features
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            perturbed_logits = logit_scale * image_features @ text_features.t()

            loss = F.cross_entropy(perturbed_logits, label)
            
            return loss + self.lambd * loss_distill + loss_vsd
        

        return logits, image_features, text_features, label


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
# FiLM Module for token enhancement
class FiLM(nn.Module):
    def __init__(self, dim, bias=True, use_sigmoid=False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.has_bias = bias
        self.use_sigmoid = use_sigmoid

    def forward(self, x):
        scale = self.scale.unsqueeze(0).type(x.dtype)
        bias = self.bias.unsqueeze(0).type(x.dtype) if self.has_bias else None

        x = scale * x
        if bias is not None:
            x = x + bias

        if self.use_sigmoid:
            return x.sigmoid()

        return x

# Modified encode_text function with FiLM enhancement
def encode_text_with_film(clip_model, text):
    print("encode_text_with_film!~!!")
    # Token embedding
    x = clip_model.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]

    # Add positional embedding
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    
    # Permute for transformer input format
    x = x.permute(1, 0, 2)  # NLD -> LND

    # Initialize FiLM module to enhance each token representation
    film = FiLM(dim=x.size(-1)).to(x.device)

    # Apply FiLM enhancement to each token representation before feeding to transformer
    x = film(x)

    # Pass through transformer
    x = clip_model.transformer(x)

    # Permute back to original format
    x = x.permute(1, 0, 2)  # LND -> NLD

    # Layer normalization
    x = clip_model.ln_final(x).type(clip_model.dtype)

    # Take features from the EOT embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x

def gpt_clip_classifier(classnames, gpt_prompts, clip_model, dataset_name):
    import os

    os.makedirs("cache/", exist_ok=True)

    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace("_", " ")
            texts = []
            for t in gpt_prompts[classname]:
                texts.append(t)
            texts = clip.tokenize(texts)
            if torch.cuda.is_available():
                clip_model = clip_model.cuda()
                texts = texts.cuda()
            # prompt ensemble for ImageNet
            class_embeddings = encode_text_with_film(clip_model, texts)
            #class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            clip_weights.append(class_embeddings)
            #print(f"Class embeddings shape for current class: {class_embeddings.shape}")


        clip_weights = torch.stack(clip_weights, dim=1)
        if torch.cuda.is_available():
            clip_weights = clip_weights.cuda()
        torch.save(clip_weights, f"cache/{dataset_name}_clip_weights_random.pt")
    return clip_weights


@TRAINER_REGISTRY.register()
class VcdPrompt(TrainerX):
    def run_epoch(self):

        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()


        print('------epoch-----')
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            #print(loss_summary)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % 1 == 0
            only_few_batches = self.num_batches < 1
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

        
    def check_cfg(self, cfg):
        assert cfg.TRAINER.CoPrompt.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
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
            # CLIP's default precision is fp16
            clip_model.float()
            clip_model_distill.float()

        with open(
            f"gpt_file/{dataset_name_mapping[cfg.DATASET.NAME]}_prompt.json"
        ) as f:
            gpt3_prompt = json.load(f)

        self.alpha = weights_mapping[cfg.DATASET.NAME]
        # Textual features
        # print("classnames",classnames)
        # print(" gpt3_prompt", gpt3_prompt)
        print("\nGetting textual features as CLIP's classifier.")
        clip_weights = gpt_clip_classifier(
            classnames, gpt3_prompt, clip_model_distill, cfg.DATASET.NAME
        )

        print("Building custom CLIP")
        self.model = CustomCLIP(
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
        # 使用 DATASET.NAME 来确定数据集，而不是从 OUTPUT_DIR 解析，
        # 以避免绝对路径/目录结构变化导致解析错误
        dataset_select = cfg.DATASET.NAME
        self.cls_num = dataset_clsnum_mapping.get(dataset_select)
        print("self.cls_num",self.cls_num)

    def forward_backward(self, batch):
        image1, image2, label = self.parse_batch_train(batch)


        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.CoPrompt.PREC
        


        
        self.set_model_mode("train")
        if prec == "amp":
            with autocast():
                loss = model(image1, image2, label).mean()
            
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image1, image2, label).mean()
            #prototypes
            #self.set_model_mode("eval")
            num_labeled_samples = 1000  # 根据实际数据大小设定
            memory_bank = MemoryBank(num_labeled_samples, self.cls_num)

            for self.batch_idx, batch in enumerate(self.train_loader_x):
                image, _, label = self.parse_batch_train(batch)
                model = self.model
                img_features, text_features = model.forward_feature(image)
                memory_bank.update_labeled_feature(img_features, text_features, label)
            scloss= memory_bank.semantic_relationship_optimization()
            #scloss = torch.tensor(scloss, dtype=torch.float32).to(self.device)
            print("scloss grad_fn:", scloss.item())
            print("soss grad_fn:", loss.item())
            optim.zero_grad()
            loss = loss + self.alpha * scloss
            #print("-----------------loss-----------------",loss)
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        image1, image2 = input[0], input[1]
        label = batch["label"]
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)
        label = label.to(self.device)
        return image1, image2, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        print("names",names)

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print(
                "Loading weights to {} "
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        self.all_results = [] 
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input, label)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def model_inference(self, input, label):
        #print('Label',label)
        logits, image_features, text_features, _ =  self.model(input,label) #, pre_trained_text_features, pre_trained_image_features
        self.all_results.append({
            'logits': logits,
            'image_features': image_features,
            'text_features': text_features,
            # 'pre_trained_text_features': pre_trained_text_features,
            # 'pre_trained_image_features': pre_trained_image_features,
            'label': label,
        })
        
        save_dir = "model_outputs"
        os.makedirs(save_dir, exist_ok=True)

        
        parts = os.path.normpath(self.cfg.OUTPUT_DIR).split(os.sep)  
        target_part = parts[5]
        modelname = parts[7]
        save_path = os.path.join(save_dir, f"PromptDVD_{target_part}_outputs.pt")
        torch.save(self.all_results, save_path)

        #print(f"Saved all results to {save_path}")
        
        return logits

    def load_pre_trained(self, model_path):
        if not osp.exists(model_path):
            raise FileNotFoundError('Model not found at "{}"'.format(model_path))

        checkpoint = torch.load(model_path)
        missing_keys = self.model.load_state_dict(checkpoint, strict=False)

        if len(missing_keys.missing_keys) > 0:
            print("Missing keys: {}".format(missing_keys.missing_keys))
        if len(missing_keys.unexpected_keys) > 0:
            print("Unexpected keys: {}".format(missing_keys.unexpected_keys))