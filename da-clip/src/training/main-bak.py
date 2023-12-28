# import argparse
# import glob
# import logging
# import os
# import random
# import re
# import subprocess
# import sys
# from datetime import datetime
#
# import numpy as np
# import torch
# from torch import optim
# from torch.cuda.amp import GradScaler
#
# from training.data import get_data
# from training.distributed import is_master, init_distributed_device, broadcast_object
# from training.file_utils import pt_load, start_sync_process, remote_sync
# from training.logger import setup_logging
# from training.params import parse_args
# from training.scheduler import cosine_lr, const_lr, const_lr_cooldown
# from training.train import train_one_epoch, evaluate
#
# sys.path.insert(0, "../")
# from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
#
# try:
#     import wandb
# except ImportError:
#     wandb = None
#
# try:
#     import torch.utils.tensorboard as tensorboard
# except ImportError:
#     tensorboard = None
#
# try:
#     import horovod.torch as hvd
# except ImportError:
#     hvd = None
# sys.path.append('../open_clip')
#
# LATEST_CHECKPOINT_NAME = "epoch_latest.pt"
#
#
# def random_seed(seed=42, rank=0):
#     torch.manual_seed(seed + rank)
#     np.random.seed(seed + rank)
#     random.seed(seed + rank)
#
#
# def natural_key(string_):
#     """See http://www.codinghorror.com/blog/archives/001018.html"""
#     return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]
#
#
# def get_latest_checkpoint(path: str, remote: bool):
#     # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
#     if remote:
#         result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         print(result)
#         if result.returncode == 1:
#             return None
#         checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
#     else:
#         checkpoints = glob.glob(path + '**/*.pt', recursive=True)
#     if checkpoints:
#         checkpoints = sorted(checkpoints, key=natural_key)
#         return checkpoints[-1]
#     return None
#
#
# def main(args):
#     args = parse_args(args)
#
#     if torch.cuda.is_available():
#         # This enables tf32 on Ampere GPUs which is only 8% slower than
#         # float16 and almost as accurate as float32
#         # This was a default in pytorch until 1.12
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.benchmark = True
#         torch.backends.cudnn.deterministic = False
#
#     # fully initialize distributed device environment
#     device = init_distributed_device(args)
#
#     # get the name of the experiments
#     if args.name is None:
#         # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
#         model_name_safe = args.model.replace('/', '-')
#         date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
#         if args.distributed:
#             # 同步date_str从master到所有ranks
#             date_str = broadcast_object(args, date_str)
#         args.name = '-'.join([
#             date_str,
#             f"model_{model_name_safe}",
#             f"lr_{args.lr}",
#             f"b_{args.batch_size}",
#             f"j_{args.workers}",
#             f"p_{args.precision}",
#         ])
#
#     resume_latest = args.resume == 'latest'
#     log_base_path = os.path.join(args.logs, args.name)
#     args.log_path = None
#     if is_master(args, local=args.log_local):
#         os.makedirs(log_base_path, exist_ok=True)
#         log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
#         args.log_path = os.path.join(log_base_path, log_filename)
#         if os.path.exists(args.log_path) and not resume_latest:
#             print(
#                 "Error. Experiment already exists. Use --name {} to specify a new experiment."
#             )
#             return -1
#
#     # 日志文件
#     args.log_level = logging.DEBUG if args.debug else logging.INFO
#     setup_logging(args.log_path, args.log_level)
#
#     # 启动 wandb, tensorboard, checkpoint logging
#     args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
#     args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
#     args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
#     if is_master(args):
#         args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
#         for dirname in [args.tensorboard_path, args.checkpoint_path]:
#             if dirname:
#                 os.makedirs(dirname, exist_ok=True)
#     else:
#         args.tensorboard_path = ''
#
#     if resume_latest:
#         resume_from = None
#         checkpoint_path = args.checkpoint_path
#         # 如果使用remote_sync，需要检查远程checkpoint文件夹，而不是本地checkpoint文件夹
#         if args.remote_sync is not None:
#             checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
#             if args.save_most_recent:
#                 print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
#                 return -1
#             if args.remote_sync_protocol != 's3':
#                 print('Error. Sync protocol not supported when using resume latest.')
#                 return -1
#         if is_master(args):
#             # 仅通过主等级检查现有checkpoint。如果共享文件系统处于压力之下，
#             # 不同级别的进程可能会看到不同的文件，但是要完全解决这种情况是非常困难的。
#             if args.save_most_recent:
#                 # 如果设置了——save-most-recent，则在固定的文件名中查找latest
#                 resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
#                 if not os.path.exists(resume_from):
#                     # 如果还没有保存最新的checkpoint，不要尝试恢复
#                     resume_from = None
#             else:
#                 # 否则，列出checkpoint目录内容并选择最新的checkpoint
#                 resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
#             if resume_from:
#                 logging.info(f'Found latest resume checkpoint at {resume_from}.')
#             else:
#                 logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
#         if args.distributed:
#             # 同步找到所有等级的checkpoint路径
#             resume_from = broadcast_object(args, resume_from)
#         args.resume = resume_from
#
#     if args.copy_codebase:
#         copy_codebase(args)
#
#     # 如果remote-sync不为“None”，则启动同步进程
#     remote_sync_process = None
#     if is_master(args) and args.remote_sync is not None:
#         # 首次确认工作
#         result = remote_sync(
#             os.path.join(args.logs, args.name),
#             os.path.join(args.remote_sync, args.name),
#             args.remote_sync_protocol
#         )
#         if result:
#             logging.info('remote sync successful.')
#         else:
#             logging.info('Error: remote sync failed. Exiting.')
#             return -1
#         # 如果一切正常，每隔remote_sync_frequency秒就启动一个进程。
#         remote_sync_process = start_sync_process(
#             args.remote_sync_frequency,
#             os.path.join(args.logs, args.name),
#             os.path.join(args.remote_sync, args.name),
#             args.remote_sync_protocol
#         )
#         remote_sync_process.start()
#
#     if args.precision == 'fp16':
#         logging.warning(
#             # 'It is recommended to use AMP mixed-precision instead of FP16. '
#             '建议使用AMP混合精度代替FP16。'
#             'FP16 support needs further verification and tuning, especially for train.'
#             'FP16的支持需要进一步的验证和调整，特别是对于列车。')
#
#     if args.horovod:
#         # Horovod是分布式训练框架，对TensorFlow搭建的网络训练提速特别有效
#         logging.info(
#             f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
#             f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
#     elif args.distributed:
#         logging.info(
#             f'Running in distributed mode with multiple processes. Device: {args.device}.'
#             f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
#     else:
#         logging.info(f'Running with a single process. Device {args.device}.')
#
#     dist_model = None
#     args.distill = args.distill_model is not None and args.distill_pretrained is not None
#     if args.distill:
#         # FIXME: support distillation with grad accum.
#         # 在使用蒸馏（distillation）时，需要支持梯度累积（gradient accumulation）
#         assert args.accum_freq == 1
#         # FIXME: support distillation with coca.
#         # 表示没有使用coca库，需要修复
#         assert 'coca' not in args.model.lower()
#
#     if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
#         # 检查args.force_image_size是否为元组或列表，并且长度为1。
#         args.force_image_size = args.force_image_size[0]
#     random_seed(args.seed, 0)
#     # 获取到模型、训练图像处理变换、验证图像处理变换
#     model, preprocess_train, preprocess_val = create_model_and_transforms(
#         args.model,
#         args.pretrained,
#         precision=args.precision,
#         device=device,
#         jit=args.torchscript,
#         force_quick_gelu=args.force_quick_gelu,
#         force_custom_text=args.force_custom_text,
#         force_patch_dropout=args.force_patch_dropout,
#         force_image_size=args.force_image_size,
#         pretrained_image=args.pretrained_image,
#         image_mean=args.image_mean,
#         image_std=args.image_std,
#         aug_cfg=args.aug_cfg,
#         output_dict=True,
#     )
#     if args.distill:
#         # FIXME: currently assumes the model your distilling from has the same tokenizer & transforms.
#         # 目前假设您提取的模型具有相同的标记器和转换
#         # 创建蒸馏模型和图像变换器
#         dist_model, _, _ = create_model_and_transforms(
#             args.distill_model,
#             args.distill_pretrained,
#             device=device,
#             precision=args.precision,
#             output_dict=True,
#         )
#     if args.use_bnb_linear is not None:
#         # 使用bitsandbytes进行优化
#         print('=> using a layer from bitsandbytes.\n'
#               '   this is an experimental feature which requires two extra pip installs\n'
#               '   pip install bitsandbytes triton'
#               '   please make sure to use triton 2.0.0')
#         import bitsandbytes as bnb
#         from open_clip.utils import replace_linear
#         print(f'=> replacing linear layers with {args.use_bnb_linear}')
#         linear_replacement_cls = getattr(bnb.nn.triton_based_modules, args.use_bnb_linear)
#         replace_linear(model, linear_replacement_cls)
#         model = model.to(device)
#
#     random_seed(args.seed, args.rank)
#
#     if args.trace:
#         # 使用TorchScript对预训练模型进行优化，以便在部署和使用时获得更高的性能
#         model = trace_model(model, batch_size=args.batch_size, device=device)
#
#     if args.lock_image:
#         # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
#         # 锁定图像塔，根据论文LiT进行锁定
#         model.lock_image_tower(
#             unlocked_groups=args.lock_image_unlocked_groups,
#             freeze_bn_stats=args.lock_image_freeze_bn_stats)
#     if args.lock_text:
#         # 锁定文本塔的层，根据参数args.lock_text_unlocked_layers指定要保留的层。
#         # 冻结层归一化层，如果args.lock_text_freeze_layer_norm为真。
#         model.lock_text_tower(
#             unlocked_layers=args.lock_text_unlocked_layers,
#             freeze_layer_norm=args.lock_text_freeze_layer_norm)
#
#     if args.grad_checkpointing:
#         # 设置梯度检查点功能
#         model.set_grad_checkpointing()
#
#     if is_master(args):
#         # 输出模型，日志和参数
#         logging.info("Model:")
#         logging.info(f"{str(model)}")
#         logging.info("Params:")
#         params_file = os.path.join(args.logs, args.name, "params.txt")
#         with open(params_file, "w") as f:
#             for name in sorted(vars(args)):
#                 val = getattr(args, name)
#                 logging.info(f"  {name}: {val}")
#                 f.write(f"{name}: {val}\n")
#
#     if args.distributed and not args.horovod:
#         # 不使用horovod框架
#         # 初始化分布式训练所需的分布式数据并行模型（DDP）和蒸馏模型（dist_model）
#         if args.use_bn_sync:
#             model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
#         ddp_args = {}
#         if args.ddp_static_graph:
#             # this doesn't exist in older PyTorch, arg only added if enabled
#             ddp_args['static_graph'] = True
#         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
#
#         if args.distill:
#             dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)
#
#     # 创建 optimizer和scalar
#     optimizer = None
#     scalar = None
#
#     if args.train_data or args.dataset_type == "synthetic":
#         assert not args.trace, 'Cannot train with traced model'
#         # 如果参数的维度小于2（即参数是一个标量值），那么该参数将被排除
#         # 如果参数名中包含"bn"（批量归一化）或"ln"（层归一化）或"bias"（偏置项），那么该参数将被排除
#         # 如果参数名中包含"logit_scale"（用于生成logits的缩放因子），那么该参数将被排除
#         exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
#         # 如果exclude函数返回False，那么该参数将被包含，即exclude和include返回的结果相反
#         include = lambda n, p: not exclude(n, p)
#
#         named_parameters = list(model.named_parameters())  # 获取模型中所有参数的名称和值
#         # 使用exclude(n, p)函数排除某些参数，检查p是否需要梯度更新
#         gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
#         rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
#
#         optimizer = optim.AdamW(
#             [
#                 {"params": gain_or_bias_params, "weight_decay": 0.},
#                 {"params": rest_params, "weight_decay": args.wd},
#             ],
#             lr=args.lr,
#             betas=(args.beta1, args.beta2),
#             eps=args.eps,
#         )
#         if args.horovod:
#             # 如果使用Horovod进行分布式训练，则创建一个分布式优化器
#             # 并将模型和优化器的状态广播到所有工作进程
#             optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
#             hvd.broadcast_parameters(model.state_dict(), root_rank=0)
#             hvd.broadcast_optimizer_state(optimizer, root_rank=0)
#         # 如果使用混合精度AMP进行训练，则创建一个GradScaler，用于在训练过程中进行缩放，防止数值溢出
#         scalar = GradScaler() if args.precision == "amp" else None
#
#     # 选择地从检查点恢复
#     start_epoch = 0
#     if args.resume is not None:
#         checkpoint = pt_load(args.resume, map_location='cpu')  # 加载检查点文件
#         if 'epoch' in checkpoint:
#             # 如果检查点文件中包含epoch关键字，说明这是一个训练检查点，
#             # 将起始迭代设置为检查点中的epoch值
#             start_epoch = checkpoint["epoch"]
#             sd = checkpoint["state_dict"]  # 从检查点中提取模型状态字典（state_dict），并将其加载到模型上。
#             if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
#                 # 如果模型没有在分布式环境中训练，并且state_dict中的键以module.开头，则将它们从state_dict中移除
#                 sd = {k[len('module.'):]: v for k, v in sd.items()}
#             model.load_state_dict(sd)
#             if optimizer is not None:
#                 optimizer.load_state_dict(checkpoint["optimizer"])  # 加载optimizer
#             if scalar is not None and 'scalar' in checkpoint:
#                 scalar.load_state_dict(checkpoint['scalar'])  # 加载scalar
#             logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
#         else:
#             # 没有提供回复检查点文件的路径，则裸加载(仅模型)以进行微调或评估
#             model.load_state_dict(checkpoint)
#             logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
#
#     # 初始化数据集
#     data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=get_tokenizer(args.model))
#     assert len(data), 'At least one train or eval dataset must be specified.'  # 必须指定至少一个训练或eval数据集
#
#     # 如果是训练的话则创建scheduler
#     scheduler = None
#     if 'train' in data and optimizer is not None:
#         # 获取训练数据集的批处理数量（num_batches），并将其除以args.accum_freq以计算总训练步骤
#         total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
#         # 根据传入的args.lr_scheduler参数，选择相应的调度器
#         if args.lr_scheduler == "cosine":
#             scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)  # 创建一个余弦学习率调度器
#         elif args.lr_scheduler == "const":
#             scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)  # 创建一个常量学习率调度器
#         elif args.lr_scheduler == "const-cooldown":
#             # 使用const_lr_cooldown函数创建一个带有冷酷期的常量学习率调度器。
#             # 在这种情况下，代码需要args.epochs_cooldown参数来指定冷酷期的长度
#             assert args.epochs_cooldown is not None, \
#                 "Please specify the number of cooldown epochs for this lr schedule."
#             cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
#             scheduler = const_lr_cooldown(
#                 optimizer, args.lr, args.warmup, total_steps,
#                 cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
#         else:
#             # 如果三种情况都不是,报错退出
#             logging.error(
#                 f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
#             exit(1)
#
#     # 只有当rank == 0时,确定该工作者是否应该保存日志和检查点。
#     args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
#     writer = None
#     if args.save_logs and args.tensorboard:
#         assert tensorboard is not None, "Please install tensorboard."
#         writer = tensorboard.SummaryWriter(args.tensorboard_path)
#
#     if args.wandb and is_master(args):
#         # 检查args.wandb是否设置为True，并且当前进程是否是主进程
#         # 获取训练和验证数据的大小，以便在wandb中进行记录
#         assert wandb is not None, 'Please install wandb.'
#         logging.debug('Starting wandb.')
#         args.train_sz = data["train"].dataloader.num_samples
#         if args.val_data is not None:
#             args.val_sz = data["val"].dataloader.num_samples
#         # 配置wandb
#         wandb.init(
#             project=args.wandb_project_name,
#             name=args.name,
#             id=args.name,
#             notes=args.wandb_notes,
#             tags=[],
#             resume='auto' if args.resume == "latest" else None,
#             config=vars(args),
#         )
#         if args.debug:
#             wandb.watch(model, log='all')
#         wandb.save(params_file)
#         logging.debug('Finished loading wandb.')
#
#     if args.torchcompile:
#         logging.info('Compiling model...')  # 编译
#         model = torch.compile(model)
#
#     if 'train' not in data:
#         # 如果不是训练模式,说明是推理,则进行评估并return
#         # 如果使用INT8量化，将模型转换为推理模式
#         if args.use_bnb_linear is not None:
#             from open_clip.utils import convert_int8_model_to_inference_mode
#             convert_int8_model_to_inference_mode(model)
#         # 评估CLIP模型
#         evaluate(model, data, start_epoch, args, writer)
#         return
#
#     loss = create_loss(args)
#
#     for epoch in range(start_epoch, args.epochs):
#         if is_master(args):
#             logging.info(f'Start epoch {epoch}')
#
#         train_one_epoch(model, data, loss, epoch, optimizer, scalar, scheduler, dist_model, args, tb_writer=writer)
#         completed_epoch = epoch + 1
#
#         if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
#             evaluate(model, data, completed_epoch, args, writer)
#
#         # 保存checkpoints.
#         if args.save_logs:
#             # 存储存储当前训练状态，如当前轮数（epoch）、模型参数（state_dict）、优化器状态（optimizer）。
#             checkpoint_dict = {
#                 "epoch": completed_epoch,
#                 "name": args.name,
#                 "state_dict": model.state_dict(),
#                 "optimizer": optimizer.state_dict(),
#             }
#             if scalar is not None:
#                 checkpoint_dict["scalar"] = scalar.state_dict()
#
#             if completed_epoch == args.epochs or (
#                     args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
#             ):
#                 # 如果当前轮数是最后一轮或者满足保存频率要求（args.save_frequency > 0）
#                 # 则将 checkpoint_dict 保存到指定路径下，文件名为 epoch_{completed_epoch}.pt
#                 torch.save(
#                     checkpoint_dict,
#                     os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
#                 )
#             if args.delete_previous_checkpoint:
#                 # 删除上一轮checkpoint
#                 previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
#                 if os.path.exists(previous_checkpoint):
#                     os.remove(previous_checkpoint)
#
#             if args.save_most_recent:
#                 # 在保存新的检查点之前，先保存一个临时文件（tmp.pt）。
#                 # 如果保存失败，不会破坏现有的最新检查点。在保存完成后，将临时文件替换为最新的检查点
#                 tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
#                 latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
#                 torch.save(checkpoint_dict, tmp_save_path)
#                 os.replace(tmp_save_path, latest_save_path)
#
#     if args.wandb and is_master(args):
#         # 主进程且args.wandb==True
#         wandb.finish()
#
#     # run a final sync.
#     if remote_sync_process is not None:
#         logging.info('Final remote sync.')
#         remote_sync_process.terminate()
#         result = remote_sync(
#             os.path.join(args.logs, args.name),
#             os.path.join(args.remote_sync, args.name),
#             args.remote_sync_protocol
#         )
#         if result:
#             logging.info('Final remote sync successful.')
#         else:
#             logging.info('Final remote sync failed.')
#
#
# # 将当前代码库的副本复制到一个新的文件夹(args.logs和args.name的组合)中，以便在实验中使用
# def copy_codebase(args):
#     from shutil import copytree, ignore_patterns
#     new_code_path = os.path.join(args.logs, args.name, "code")
#     if os.path.exists(new_code_path):
#         print(
#             f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
#         )
#         return -1
#     print(f"Copying codebase to {new_code_path}")
#     current_code_path = os.path.realpath(__file__)
#     for _ in range(3):
#         current_code_path = os.path.dirname(current_code_path)
#     copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
#     print("Done copying code.")
#     return 1
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('arguments', nargs='*')  # nargs='*'表示接受任意个参数
#     parser.add_argument("--save-frequency", type=int, default=1)
#     parser.add_argument("--zeroshot-frequency", type=int, default=1)
#     parser.add_argument("--report_to", default="tensorboard")
#     parser.add_argument("--train_data", type=str, default="/root/autodl-fs/datasets/universal/daclip_train.csv")
#     parser.add_argument("--val_data", type=str, default="/root/autodl-fs/datasets/universal/daclip_val.csv")
#     parser.add_argument("--csv-img-key", default="filepath")
#     parser.add_argument("--csv-caption-key", default="title")
#     parser.add_argument("--warmup", type=int, default=100)
#     parser.add_argument("--batch-size", type=int, default=784)
#     parser.add_argument("--lr", type=int, default=2e-5)
#     parser.add_argument("--wd", type=float, default=0.05)
#     parser.add_argument("--epoch", type=int, default=30)
#     parser.add_argument("--worker", type=int, default=8)
#     parser.add_argument("--model", default="daclip_ViT-B-32")
#     parser.add_argument("--name", type=str, default="daclip_ViT-B-32-2023-09_b512x1_lr2e-5_e30_test")
#     parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
#     parser.add_argument("--da", type=str, default=None)
#     args = parser.parse_args()  # 转成数组
#     arguments = args.arguments
#     print(arguments)
#     # main(arguments)
#     main(sys.argv[1:])
