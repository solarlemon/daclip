import logging

import torch
from models import modules as M

logger = logging.getLogger("base")


# Generator
def define_G(opt):
    """
    从给定的opt字典中提取有关生成器（Generator）网络的信息，然后创建一个生成器对象
    :param opt: 包含生成器（Generator）网络信息的字典
    """
    opt_net = opt["network_G"]
    which_model = opt_net["which_model_G"]
    setting = opt_net["setting"]
    netG = getattr(M, which_model)(** setting)
    return netG


# Discriminator
def define_D(opt):
    """
    从给定的参数中创建一个判别器网络
    首先从opt字典中提取有关网络设置的信息，然后使用getattr函数调用定义在M模块中的模型类
    :param opt: 包含判别器（Discriminator）网络信息的字典
    """
    opt_net = opt["network_D"]
    which_model = opt_net["which_model_D"]
    setting = opt_net["setting"]
    netD = getattr(M, which_model)(**setting)
    return netD


# Perceptual loss
def define_F(opt, use_bn=False):
    """
    返回预训练的VGG19-54网络，但只使用最后一个卷积层进行特征提取
    """
    # 确定是否使用GPU
    gpu_ids = opt["gpu_ids"]
    device = torch.device("cuda" if gpu_ids else "cpu")
    # PyTorch pretrained VGG19-54, before ReLU.
    # use_bn==True：使用VGG19-54网络的最后一个卷积层（特征层49）进行特征提取
    # use_bn==False：使用VGG19-54网络的最后一个卷积层（特征层34）进行特征提取
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    # 预训练的VGG19-54网络，但只使用最后一个卷积层进行特征提取。
    netF = M.VGGFeatureExtractor(
        feature_layer=feature_layer, use_bn=use_bn, use_input_norm=True, device=device
    )
    netF.eval()  # No need to train
    return netF  # 用于计算感知损失的网络
