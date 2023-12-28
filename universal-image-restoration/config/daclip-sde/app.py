import argparse
import os
import sys

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

import options as option
from models import create_model

sys.path.insert(0, "../../")
import open_clip
import utils as util

# options
parser = argparse.ArgumentParser()
# 测试图片位于universal-image-restoration/config/daclip-sde/images
parser.add_argument("-opt", type=str, default='options/test.yml', help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

# 默认加载预训练模型
model = create_model(opt)
device = model.device

# 加载clip模型
clip_model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=opt['path']['daclip'])
clip_model = clip_model.to(device)

# 加载IR-SDE
sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"],
                 eps=opt["sde"]["eps"], device=device)
sde.set_model(model.model)


def clip_transform(np_image, resolution=224):
    pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
    return Compose([
        # 使用Resize对图像进行缩放，将图像调整为指定的分辨率（默认为224x224像素）。（双三次插值方法InterpolationMode.BICUBIC）
        Resize(resolution, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(resolution),  # 使中心区域与原始图像相同
        ToTensor(),
        # 三个均值和三个方差值，分别对应于RGB三个通道
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])(pil_image)


def restore(image):
    # 处理图像：归一化和调整图像
    image = image / 255.
    img4clip = clip_transform(image).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        # 图像编码器生成图像特征和退化上下文
        image_context, degra_context = clip_model.encode_image(img4clip, control=True)
        image_context = image_context.float()
        degra_context = degra_context.float()

    LQ_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    noisy_tensor = sde.noise_state(LQ_tensor)
    model.feed_data(noisy_tensor, LQ_tensor, text_context=degra_context, image_context=image_context)
    model.test(sde)
    visuals = model.get_current_visuals(need_GT=False)
    output = util.tensor2img(visuals["Output"].squeeze())
    return output[:, :, [2, 1, 0]]


examples = [os.path.join(os.path.dirname(__file__), f"images/{i}.jpg") for i in range(1, 11)]
interface = gr.Interface(fn=restore, inputs="image", outputs="image", title="Image Restoration with DA-CLIP",
                         examples=examples)
# img1 = Image.open(interface.temp_file_sets[1])
interface.launch(share=True)
