import os
import torch
from PIL import Image
import open_clip
from tqdm import tqdm

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# 从给定路径中返回图像文件的路径列表
def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


checkpoint = 'logs/daclip_ViT-B-32_b768x4_lr3e-5_e50/checkpoints/epoch_50.pt'

# 模型、图像预处理变换、根据模型名生成对应分词器
model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=checkpoint)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

val_root = 'datasets/universal/val'
degradations = ['motion-blurry', 'hazy', 'jpeg-compressed', 'low-light', 'noisy', 'raindrop', 'rainy', 'shadowed',
                'snowy', 'uncompleted']

text = tokenizer(degradations)  # 将文本数据转换为模型可以处理的格式
with torch.no_grad(), torch.cuda.amp.autocast():
    # 使用预训练的模型model对输入的文本进行编码，并将处理后的文本数据转换为向量形式
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)  # 编码后归一化处理

for i, degradation in enumerate(degradations):
    # 将验证集的根目录和当前degradation值连接起来，以创建包含LQ图像的目录路径
    root_path = os.path.join(val_root, degradation, 'LQ')
    image_paths = get_paths_from_images(root_path)
    acc = 0.0
    for im_path in tqdm(image_paths):  # tqdm()函数遍历image_paths列表中的每个图像路径(进度条)
        # 使用preprocess()函数对图像进行预处理，并将其unsqueeze到（1，C，H，W）格式
        image = preprocess(Image.open(im_path)).unsqueeze(0)
        with torch.no_grad(), torch.cuda.amp.autocast():
            _, image_features = model.encode_image(image, control=True)  # 图像编码，控制参数设为True
            image_features /= image_features.norm(dim=-1, keepdim=True)  # 将图像特征除以其范数，以便在进行比较之前将其归一化
            # 使用torch.argmax()函数计算图像特征与文本特征之间的相似性，并将其与索引i进行比较
            index = torch.argmax((image_features @ text_features.T)[0])
            acc += float(index == i)
    acc /= len(image_paths)
    print(f'degradation: {degradation},\t accuracy: {acc:.6f}')  # 打印出当前degradation值和对应的准确率
