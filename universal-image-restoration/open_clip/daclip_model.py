from typing import Optional

import logging
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import copy

from .transformer import (
    ControlTransformer
)
from .model import CLIP, CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower


class DaCLIP(nn.Module):
    def __init__(self, clip_model: CLIP):
        super().__init__()
        # 定义clip模型及其visual_control和visual_control.transformer
        self.clip = clip_model
        self.visual = clip_model.visual  # visual(image)返回图像特征
        self.visual_control = copy.deepcopy(clip_model.visual)
        # 替换原始编码器，CLIP图像编码器的副本，但是包含一些零初始化连接来向编码器添加控件，
        # 它操作所有编码器块的输出来控制图像编码器的预测
        self.visual_control.transformer = ControlTransformer(self.visual_control.transformer)
        self.logit_scale = copy.deepcopy(clip_model.logit_scale)

    def initial_controller(self):
        for (kv, param_v), (kc, param_c) in zip(self.clip.visual.named_parameters(),
                                                self.visual_control.named_parameters()):
            if 'transformer' not in kv:
                param_c.data.copy_(param_v.data)  # 将clip模型的视觉和文本编码器的参数复制到visual_control中

        for param_v, param_c in zip(self.clip.visual.transformer.parameters(),
                                    self.visual_control.transformer.parameters()):
            param_c.data.copy_(param_v.data)  # 将visual和visual_control的编码器参数复制到DaCLIP对象中

        self.logit_scale.data.copy_(self.clip.logit_scale.data)  # 将logit_scale的副本分配给DaCLIP对象

    def lock_clip(self):
        """
        锁定clip模型，将requires_grad属性设置为False，以防止计算梯度和更新参数
        """
        for param in self.clip.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.clip.visual.set_grad_checkpointing(enable)
        self.clip.transformer.grad_checkpointing = enable
        self.visual_control.set_grad_checkpointing(enable)

    def encode_image(self, image, control=False, normalize: bool = False):
        if control:
            # 嵌入层图像退化e^I_d和隐藏控件h_c
            degra_features, hiddens = self.visual_control(image, output_hiddens=True)
            image_features = self.clip.visual(image, control=hiddens)  # 控制image_encoder返回e^I_c
            # 归一化
            image_features = F.normalize(image_features, dim=-1) if normalize else image_features
            degra_features = F.normalize(degra_features, dim=-1) if normalize else degra_features
            return image_features, degra_features
        else:
            return self.clip.encode_image(image, normalize)  # 否则使用clip中原始的图像编码

    def encode_text(self, text, normalize: bool = False):
        return self.clip.encode_text(text, normalize)

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        """
        :return:图像特征、图像退化特征、文本特征、文本退化特征、logit_scale
        """
        (caption, degradation) = text.chunk(2, dim=-1) if text is not None else (None, None)
        image_features, image_degra_features = self.encode_image(image, control=True,
                                                                 normalize=True) if image is not None else None
        text_features = self.encode_text(caption, normalize=True) if text is not None else None
        text_degra_features = self.encode_text(degradation, normalize=True) if degradation is not None else None

        return {
            "image_features": image_features,
            "text_features": text_features,
            "image_degra_features": image_degra_features,
            "text_degra_features": text_degra_features,
            "logit_scale": self.logit_scale.exp()
        }
