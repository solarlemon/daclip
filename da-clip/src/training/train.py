import json
import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast

sys.path.insert(0, "../")
from open_clip import get_input_dtype, CLIP, CustomTextCLIP


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


# 在计算损失（total_loss）之后执行反向传播，特别在使用混合精度计算时有助于加速计算
def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # 通过sampler或shared_epoch以进程安全的方式设置epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq  # 每个epoch中可以训练的批次数量
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        # 初始化累积图像、累积文本、累积特征
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    # 计算批量处理时间和数据加载时间
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        # a.获取图像和文本数据
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)
        images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            # 使用自动混合精度训练进行前向传播，计算损失并更新梯度
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss
            # 计算精度后反向传播，有助于加速计算
            backward(total_loss, scaler)
        else:
            # First,在没有任何梯度跟踪的情况下缓存特征。
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)
                    model_out.pop("logit_scale")
                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]
                # 将此次epoch的每个批次的images、texts累积(此时accum_freq>1)
                accum_images.append(images)
                accum_texts.append(texts)

            # 如果(i + 1) % accum_freq不为零，则直接进行下一批处理。
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                # 这使得数据时间记录在不可靠时积累
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()  # 梯度清零，重新计算
            for j in range(args.accum_freq):  # accum_freq表示累积梯度的频率
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)
                    logit_scale = model_out.pop("logit_scale")
                    inputs = {}  # 存储累积特征
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])
                    losses = loss(**inputs, logit_scale=logit_scale, output_dict=True)
                    del inputs
                    total_loss = sum(losses.values())  # 计算损失
                    losses["loss"] = total_loss
                backward(total_loss, scaler)

        # 主要涉及到机器学习中常用的梯度缩放和梯度裁剪技术
        if scaler is not None:
            # 使用了梯度缩放技术，需要对梯度进行缩放和平滑处理
            if args.horovod:
                optimizer.synchronize()  # 使用horovod库，进行分布式训练时进行同步
                scaler.unscale_(optimizer)  # 将优化器中的梯度进行取消缩放并应用裁剪
                if args.grad_clip_norm is not None:
                    # 如果设置了grad_clip_norm参数，会跳过同步步骤，使用缩放器进行梯度更新
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)  # 使用缩放器进行梯度更新
            else:
                # 未使用梯度缩放技术，代码会将优化器中的梯度进行取消缩放，并应用梯度裁剪
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            # scalar为None，说明没有使用梯度缩放技术，代码直接使用optimizer.step()进行梯度更新
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        if args.accum_freq > 1:
            # 重置梯度计算
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            # CLIP的原论文中将logit_scale限制在(0，ln(100))
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)  # 更新批量处理时间
        end = time.time()
        batch_count = i_accum + 1  # 当前迭代次数
        # 检查是否是主进程且(当前迭代可以被args.log_every_n_steps整除||已经遍历完训练集)
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch  # 训练进度

            # 损失是粗采样，只对主节点和每次日志更新进行采样
            for key, val in losses.items():
                # 使用losses_m记录每个损失的均值
                # 如果损失名称未在losses_m中找到，则创建一个新的AverageMeter对象
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)  # 更新losses_m中的每个损失的值

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            # 打印训练进度和损失信息，包括数据加载时间、批量处理时间和每秒的样本数以及学习率等
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # 节省训练损失等。使用非平均值计值作为记录器有自己的平滑
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})  # 保存到日志

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # 重置批量处理和数据加载时间计数器
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}  # 存储训练过程中的各种指标
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)  # 零次学习评估，模型在数据集上零样本推理的正确率
    metrics.update(zero_shot_metrics)  # 将 zero_shot_metrics 字典中的指标添加到 metrics 中,就可以看到每个轮数的指标变化。

    autocast = get_autocast(args.precision)  # 根据传入的参数precision来选择合适的自动类型转换策略，以提高模型训练的性能
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # All_image_features @ all_text_features会迅速增加内存和计算量
        cumulative_loss = 0.0  # 计算损失
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []  # 图像特征和文本特征
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():  # 混合精度计算，以在训练过程中节省显存
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]  # 缩放因子
                    # 特征在CPU张量中积累，否则GPU内存很快耗尽，
                    # 系统RAM很容易超过，计算时间就会出现问题
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()  # 将概率转换成标签
                    # 交叉熵total_loss
                    total_loss = (
                                         F.cross_entropy(logits_per_image, labels) +
                                         F.cross_entropy(logits_per_text, labels)
                                 ) / 2
                    # 计算生成模型损失
                    gen_loss = maybe_compute_generative_loss(model_out)
                # 累加损失，更新样本数量
                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        # 主进程的累加生成损失
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples  # 验证损失
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )  # 添加验证损失的值、当前训练的迭代次数、样本数量
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


# 计算CLIP模型的性能指标
def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


# model_out中包含"logits"和"labels"时计算模型损失
def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
