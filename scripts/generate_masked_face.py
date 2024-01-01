# 将uncompleted的训练数据集的celeba_hq_256转换成LQ
import cv2
import numpy as np
import os


def add_random_mask(img, size=None, mask_root='inpainting_masks', mask_id=-1, n=100):
    """
    在图像上添加一个随机掩码，以模拟图像中的缺失部分
    Args:
        img:输入图像，要求为NumPy数组
        size:可选参数，指定输出图像的大小。如果未指定，则使用输入图像的大小
        mask_root:可选参数，指定包含随机mask的文件夹路径。默认为'inpainting_masks'
        mask_id:可选参数，指定要使用的随机掩码的ID。如果为-1，则随机选择一个ID
        n:可选参数，指定随机掩码的数量。默认为100
    """
    if mask_id < 0:
        mask_id = np.random.randint(n)  # 如果mask_id小于0，则随机选择一个ID

    mask = cv2.imread(os.path.join(mask_root, f'{mask_id:06d}.png')) / 255.  # 读取随机ID的掩码图像，并将其归一化到0到1的范围内
    if size is None:
        # 如果size未指定，则使用输入图像的大小调整掩码大小。
        # 否则，根据指定的size调整掩码大小，并在随机位置裁剪掩码
        mask = cv2.resize(mask, (img.shape[0], img.shape[1]), interpolation=cv2.INTER_AREA)
    else:
        mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_AREA)
        rnd_h = np.random.randint(0, max(0, size[0] - img.shape[0]))
        rnd_w = np.random.randint(0, max(0, size[1] - img.shape[1]))
        mask = mask[rnd_h: rnd_h + img.shape[0], rnd_w: rnd_w + img.shape[1]]

    return mask * img + (1. - mask)


target_dir = '.'
source_dir = '../images/'
im_name = 'Elon-Musk-256x256.jpg'
im = cv2.imread(os.path.join(source_dir, im_name)) / 255.
save_path = os.path.join(target_dir, im_name)
masked_im = add_random_mask(im) * 255
img = cv2.imwrite(save_path, masked_im)
print('finished')
