import torch
from contextlib import suppress


# 根据传入的参数precision来选择合适的自动类型转换策略，以提高模型训练的性能
def get_autocast(precision):
    if precision == 'amp':
        return torch.cuda.amp.autocast  # 等于混合精度'amp'，返回torch.cuda.amp.autocast
    elif precision == 'amp_bfloat16' or precision == 'amp_bf16':
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress  # 这是一个占位符，表示不进行自动类型转换
