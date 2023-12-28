from functools import partial
from itertools import islice
from typing import Callable, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F


def batched(iterable, n):
    """Batch data into lists of length *n*. The last batch may be shorter.
    NOTE based on more-itertools impl, to be replaced by python 3.12 itertools.batched impl
    """
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def build_zero_shot_classifier(
        model,
        tokenizer,
        classnames: Sequence[str],
        templates: Sequence[Union[Callable, str]],
        num_classes_per_batch: Optional[int] = 10,
        device: Union[str, torch.device] = 'cpu',
        use_tqdm: bool = False,
):
    """ 通过批量迭代类名来构建zero-shot分类器权重
    Args:
        model: CLIP模型实例
        tokenizer: CLIP分词实例
        classnames: 类（标签）名称序列模版
        templates: 可调用对象序列或format()友好字符串以生成每个类名的模板
        num_classes_per_batch: 每次转发中要批处理的类的数量，全部为None
        device: Device to use.
        use_tqdm: 启用TQDM进度条.
    """
    assert isinstance(templates, Sequence) and len(templates) > 0
    assert isinstance(classnames, Sequence) and len(classnames) > 0
    use_format = isinstance(templates[0], str)
    num_templates = len(templates)
    num_classes = len(classnames)
    if use_tqdm:
        # 导入tqdm库，并计算迭代次数。
        # 迭代次数取决于num_classes_per_batch，如果为None，则设置为1。
        # 然后，定义一个iter_wrap函数，它是tqdm。tqdm的partial函数，用于在迭代时显示进度条。
        import tqdm
        num_iter = 1 if num_classes_per_batch is None else ((num_classes - 1) // num_classes_per_batch + 1)
        iter_wrap = partial(tqdm.tqdm, total=num_iter, unit_scale=num_classes_per_batch)
    else:
        iter_wrap = iter

    # 对批量类别名称进行了处理，然后返回了处理后的类嵌入表示
    def _process_batch(batch_classnames):
        num_batch_classes = len(batch_classnames)
        # 对批量类别名称进行模版扩充
        texts = [template.format(c) if use_format else template(c) for c in batch_classnames for template in templates]
        texts = tokenizer(texts).to(device)  # 将文本转换成模型可以理解的形式
        class_embeddings = F.normalize(model.encode_text(texts), dim=-1)  # 对文本进行编码，并将class_embeddings在最后一个维度归一化
        # 重塑class_embeddings，并对第二维度取平均，获得每个类别的嵌入表示
        class_embeddings = class_embeddings.reshape(num_batch_classes, num_templates, -1).mean(dim=1)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)
        class_embeddings = class_embeddings.T
        return class_embeddings  # 每一行表示一个模板的嵌入表示，每一列表示一个类别的嵌入表示

    # 计算zero-shot权重并返回
    # zero - shot权重是一种用于评估模型性能的方法，
    # 通过将模型从未见过的类别中提取的特征与已知类别的特征进行比较来计算
    with torch.no_grad():  # 关闭梯度计算，提高计算速度
        if num_classes_per_batch:
            batched_embeds = [_process_batch(batch) for batch in iter_wrap(batched(classnames, num_classes_per_batch))]
            zeroshot_weights = torch.cat(batched_embeds, dim=1)
        else:
            zeroshot_weights = _process_batch(classnames)
    return zeroshot_weights


def build_zero_shot_classifier_legacy(
        model,
        tokenizer,
        classnames: Sequence[str],
        templates: Sequence[Union[Callable, str]],
        device: Union[str, torch.device] = 'cpu',
        use_tqdm: bool = False,
):
    """ Build zero-shot classifier weights by iterating over class names 1 by 1
    Args:
        model: CLIP model instance
        tokenizer: CLIP tokenizer instance
        classnames: A sequence of class (label) names
        templates: A sequence of callables or format() friendly strings to produce templates per class name
        device: Device to use.
        use_tqdm: Enable TQDM progress bar.
    """
    assert isinstance(templates, Sequence) and len(templates) > 0
    assert isinstance(classnames, Sequence) and len(classnames) > 0
    if use_tqdm:
        import tqdm
        iter_wrap = tqdm.tqdm
    else:
        iter_wrap = iter

    use_format = isinstance(templates[0], str)

    with torch.no_grad():
        zeroshot_weights = []
        for classname in iter_wrap(classnames):
            texts = [template.format(classname) if use_format else template(classname) for template in templates]
            texts = tokenizer(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    return zeroshot_weights
