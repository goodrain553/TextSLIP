# Copyright (c) Meta Platforms, Inc. and affiliates

import json
import logging
import math
import os
import random
import time
from contextlib import suppress

import numpy as np
import torch
import torch.nn.functional as F

import collections
from collections import defaultdict

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import ClipLoss, get_mean_std
from .distributed import is_master, world_info_from_env
from .zero_shot import zero_shot_eval
from torchvision import transforms
from open_clip import loss


def save_checkpoint(model, optimizer, scaler, epoch, i, args, loss_fn=None):
    checkpoint_dict = {
        "epoch": epoch,
        "epoch_step": i,  # inner loop saves step and args.resume in main.py will decide if a checkpoint is saved by innerloop or epoch loop (in main).
        "name": args.name,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scaler is not None:
        checkpoint_dict["scaler"] = scaler.state_dict()
    
    # Save loss state (including queue for ESimCSE)
    if loss_fn is not None:
        checkpoint_dict["loss_state"] = loss_fn.state_dict()

    # Saving checkpoints. use eval_steps to save a checkpoint.
    if args.save_logs:  # master_only.
        # epoch saving is removed. only save `epoch_latest.pt`.
        if args.save_most_recent:
            torch.save(
                checkpoint_dict,
                os.path.join(args.checkpoint_path, f"epoch_latest.pt"),
            )


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


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def to_device(batch, device, args=None):
    """
    Move batch to device. Handles both standard CLIP and SLIP (triple augmentation) modes.
    """
    # Check if using SLIP with triple augmentation
    use_slip = args and hasattr(args, "loss") and args.loss == "SLIPLoss"
    
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        images, texts = batch
        
        # Check if images is a tuple (triple augmentation for SLIP from custom collate)
        if use_slip and isinstance(images, tuple) and len(images) == 3:
            # SLIP mode: images = (originals, aug1s, aug2s) - already batched tensors
            original, aug1, aug2 = images
            
            original = original.to(device=device, non_blocking=True)
            aug1 = aug1.to(device=device, non_blocking=True)
            aug2 = aug2.to(device=device, non_blocking=True)
            
            if args.inmem:
                # Apply normalization if in memory mode
                from open_clip import get_mean_std
                mean, std = get_mean_std(args)
                mean = torch.as_tensor(mean, device=original.device)[None, :, None, None]
                std = torch.as_tensor(std, device=original.device)[None, :, None, None]
                
                original = original.to(torch.float32).div_(255.).sub_(mean).div_(std)
                aug1 = aug1.to(torch.float32).div_(255.).sub_(mean).div_(std)
                aug2 = aug2.to(torch.float32).div_(255.).sub_(mean).div_(std)
            
            texts = texts.to(device=device, non_blocking=True)
            return original, aug1, aug2, texts
        else:
            # Standard CLIP mode - handle both tensor and list cases
            if isinstance(images, list):
                # If still a list, stack it
                images = torch.stack(images)
            images = images.to(device=device, non_blocking=True)
            
            if args and hasattr(args, "inmem") and args.inmem:
                images = images.to(torch.float32).div_(255.)
                from open_clip import get_mean_std
                mean, std = get_mean_std(args)
                mean = torch.as_tensor(mean, device=images.device)[None, :, None, None]
                std = torch.as_tensor(std, device=images.device)[None, :, None, None]
                images.sub_(mean).div_(std)
            
            texts = texts.to(device=device, non_blocking=True)
            return images, texts
    else:
        # Assuming batch contains just images, and is already a tuple/list
        images = batch[0]
        images = images.to(device=device, non_blocking=True)
        
        if args and hasattr(args, "inmem") and args.inmem:
            images = images.to(torch.float32).div_(255.)
            from open_clip import get_mean_std
            mean, std = get_mean_std(args)
            mean = torch.as_tensor(mean, device=images.device)[None, :, None, None]
            std = torch.as_tensor(std, device=images.device)[None, :, None, None]
            images.sub_(mean).div_(std)
        
        return (images,)



def train_one_epoch_ex(model, data, epoch, epoch_step, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress

    model.train()

    from open_clip import loss
    if hasattr(args, "loss"):
        loss_cls = getattr(loss, args.loss)
    else:
        loss_cls = getattr(loss, "ClipLoss")


    loss = loss_cls(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    if hasattr(args, "one_iter") and args.one_iter is True:
        # hack for big dataset using one iterator to run across 400M epoch.
        if not hasattr(data['train'], "dataloader_iter"):
            print(f"running dataloader across epochs ({args.train_num_samples} examples per epoch).")
            data['train'].dataloader_iter = iter(dataloader)
        batch_iter = data['train'].dataloader_iter
    else:
        batch_iter = iter(dataloader)

    for i in range(num_batches_per_epoch):
        if i < epoch_step:  # skip to the right i when resuming happens.
            continue
        batch = next(batch_iter)
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        
        # Debug log for first batch
        if i == 0 and is_master(args):
            logging.info(f"Starting epoch {epoch}, processing first batch...")

        batch_data = to_device(batch, device, args)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        
        # Initialize tracking variables to prevent NameError in logging
        local_batch_size = None
        global_batch_size = None
        FNMask = None
        images = None
        original = None

        if args.loss == 'SLIPLoss':
            # SLIP mode: batch_data = (original, aug1, aug2, texts)
            original, aug1, aug2, texts = batch_data
            with autocast():
                # Concatenate all images for a single forward pass to avoid DDP issues
                # Stack: [original, aug1, aug2]
                all_images = torch.cat([original, aug1, aug2], dim=0)
                
                # Single forward pass through image encoder
                model_unwrapped = unwrap_model(model)
                all_image_features = model_unwrapped.encode_image(all_images)
                
                # Split back into original, aug1, aug2
                batch_size = original.size(0)
                image_features_original = all_image_features[:batch_size]
                image_features_aug1 = all_image_features[batch_size:2*batch_size]
                image_features_aug2 = all_image_features[2*batch_size:]

                
                image_features_aug1 = model_unwrapped.simclr_projection(image_features_aug1)
                image_features_aug2 = model_unwrapped.simclr_projection(image_features_aug2)
                # Encode text separately
                text_features = model_unwrapped.encode_text(texts)
                
                # Get logit scale
                logit_scale = model_unwrapped.logit_scale.exp()
                


                # Compute SLIP loss (CLIP + SimCLR)
                total_loss = loss(
                    image_features_original,  # for CLIP (标准增强)
                    text_features,            # for CLIP
                    logit_scale,              # for CLIP
                    image_features_aug1,      # for SimCLR (强增强1)
                    image_features_aug2       # for SimCLR (强增强2)
                )
        elif args.loss == 'FNELoss':
            images, texts = batch_data
            local_batch_size = texts.size(0)
            device = texts.device
            
            # Gather texts from all GPUs to identify false negatives across all ranks
            if args.world_size > 1:
                # Gather all texts from all ranks
                if args.horovod:
                    import horovod.torch as hvd
                    all_texts = hvd.allgather(texts)
                else:
                    gathered_texts = [torch.zeros_like(texts) for _ in range(args.world_size)]
                    torch.distributed.all_gather(gathered_texts, texts)
                    all_texts = torch.cat(gathered_texts, dim=0)
                
                global_batch_size = local_batch_size * args.world_size
            else:
                all_texts = texts
                global_batch_size = local_batch_size
            
            # Create mask matrix: [local_batch, global_batch]
            # Compare local texts with all global texts to identify false negatives
            # Using vectorized operations for efficiency
            
            # Expand dimensions for broadcasting: texts [local_batch, 1, seq_len], all_texts [1, global_batch, seq_len]
            texts_expanded = texts.unsqueeze(1)  # [local_batch, 1, seq_len]
            all_texts_expanded = all_texts.unsqueeze(0)  # [1, global_batch, seq_len]
            
            # Compare all pairs: [local_batch, global_batch]
            FNMask = (texts_expanded == all_texts_expanded).all(dim=2).float()
            
            # Exclude positive pairs (same sample index globally)
            if args.world_size > 1:
                # Create diagonal mask for the current rank's portion
                for idx in range(local_batch_size):
                    global_idx = args.rank * local_batch_size + idx
                    FNMask[idx, global_idx] = 0.0
            else:
                # Single GPU: exclude diagonal
                FNMask.fill_diagonal_(0.0)
            
            with autocast():
                image_features, text_features, logit_scale = model(images, texts)
                total_loss = loss(image_features, text_features, logit_scale, FNMask)
        else:
            # Standard CLIP mode: batch_data = (images, texts)
            images, texts = batch_data
            with autocast():
                image_features, text_features, logit_scale = model(images, texts)
                total_loss = loss(image_features, text_features, logit_scale)

        if torch.isfinite(total_loss).all():
            if scaler is not None:
                scaler.scale(total_loss).backward()

                if args.horovod:
                    optimizer.synchronize()
                    scaler.unscale_(optimizer)
                    if args.norm_gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                    with optimizer.skip_synchronize():
                        scaler.step(optimizer)
                else:
                    if args.norm_gradient_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                    scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if args.norm_gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                optimizer.step()

            # Note: we clamp to 4.6052 = ln(100), as in the original paper.
            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))
        else:
            logging.warn(f"Loss is {total_loss}, skip back prop.")
            import sys
            sys.exit(1)  # protect the checkpoint for debugging.


        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            # Debug: confirm logging condition is reached
            if i == 0:
                logging.info(f"First batch log condition reached for epoch {epoch}")
            # Get batch size (handle different loss modes)
            if args.loss == 'SLIPLoss' and original is not None:
                batch_size = len(original)
            elif args.loss == 'FNELoss' and local_batch_size is not None:
                batch_size = local_batch_size
            elif images is not None:
                batch_size = len(images)
            else:
                # Fallback to config batch size
                batch_size = args.batch_size
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            
            
            # Log with false negative statistics for FNELoss
            if args.loss == 'FNELoss' and FNMask is not None:
                # Calculate FN statistics
                fn_count = FNMask.sum().item()
                fn_ratio = fn_count / (local_batch_size * global_batch_size) * 100
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                    f"Logit Scale: {logit_scale_scalar:.3f} "
                    f"FN: {fn_count:.0f} ({fn_ratio:.2f}%)"
                )
            elif args.loss == 'ESimCSECLIPLoss' and hasattr(loss_fn, 'esimcse_loss'):
                # Log with ESimCSE queue statistics
                queue_info = ""
                if loss_fn.esimcse_loss.queue is not None:
                    queue_size = loss_fn.esimcse_loss.queue.shape[0]
                    queue_info = f"Queue: {queue_size}/{loss_fn.esimcse_loss.neg_size}"
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                    f"Logit Scale: {logit_scale_scalar:.3f} "
                    f"{queue_info}"
                )
            else:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                    f"Logit Scale: {logit_scale_scalar:.3f}"
                )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            
            # Add FNELoss specific metrics
            if args.loss == 'FNELoss' and FNMask is not None:
                fn_count = FNMask.sum().item()
                fn_ratio = fn_count / (local_batch_size * global_batch_size) * 100
                log_data["fn_count"] = fn_count
                log_data["fn_ratio"] = fn_ratio
            
            # Add ESimCSE queue metrics
            if args.loss == 'ESimCSECLIPLoss' and hasattr(loss_fn, 'esimcse_loss'):
                if loss_fn.esimcse_loss.queue is not None:
                    queue_size = loss_fn.esimcse_loss.queue.shape[0]
                    log_data["queue_size"] = queue_size
            
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

        if hasattr(args, "save_steps") and (step + 1) % args.save_steps == 0:
            save_checkpoint(model, optimizer, scaler, epoch, i, args, loss_fn=loss)
    
        # TODO: copied from main.py, wrap as a function call.
        if hasattr(args, "eval_steps") and (step + 1) % args.eval_steps == 0: # TODO (huxu): put eval on master only?
            if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
                evaluate_ex(model, data, step, args, tb_writer)  # completed_epoch -> epoch, writer -> tb_writer
            save_checkpoint(model, optimizer, scaler, epoch, i, args, loss_fn=loss)
            model.train()  # evaluate won't turn model back to train."""
    # end for


def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress

    model.train()
    loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        images, texts = to_device(batch, device, args)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            total_loss = loss(image_features, text_features, logit_scale)

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.norm_gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.norm_gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.norm_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def text_duplicate_word(text, dup_rate=0.3):
    """
    Duplicate random words in text for data augmentation
    
    Args:
        text: input text string
        dup_rate: proportion of words to duplicate
    
    Returns:
        augmented text string
    """
    sent_list = text.split(' ')
    sent_len = len(sent_list)
    
    if sent_len > 0:
        # Randomly select number of words to duplicate
        add_len = random.randrange(min(10, sent_len, max(2, int(dup_rate * sent_len))))
        # Randomly select positions to duplicate
        dup = sorted(random.sample(range(0, sent_len-1), add_len))
        for i in dup:
            # Duplicate word directly
            sent_list[i] = sent_list[i] + ' ' + sent_list[i]
        new_text = ' '.join(sent_list)
    else:
        new_text = text
    
    return new_text


def train_one_epoch_textSLIP(model, data, epoch, epoch_step, optimizer, scaler, scheduler, args, tb_writer=None, loss_fn=None, tokenizer=None):
    """
    Training function for ESimCSECLIPLoss and standard CLIP loss.
    Removed SLIPLoss and FNELoss support for focused experimentation.
    """
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress

    model.train()

    # Create or reuse loss function (supports queue persistence across epochs)
    if loss_fn is None:
        from open_clip import loss as loss_module
        if hasattr(args, "loss"):
            loss_cls = getattr(loss_module, args.loss)
        else:
            loss_cls = getattr(loss_module, "ClipLoss")

        # Create loss with appropriate parameters
        loss_kwargs = {
            'local_loss': args.local_loss,
            'gather_with_grad': args.gather_with_grad,
            'cache_labels': True,
            'rank': args.rank,
            'world_size': args.world_size,
            'use_horovod': args.horovod
        }
        
        # Add ESimCSE specific parameters if using ESimCSECLIPLoss
        if hasattr(args, "loss") and args.loss == "ESimCSECLIPLoss":
            if hasattr(args, "esimcse_scale"):
                loss_kwargs['esimcse_scale'] = args.esimcse_scale
            if hasattr(args, "esimcse_temperature"):
                loss_kwargs['esimcse_temperature'] = args.esimcse_temperature
            if hasattr(args, "esimcse_neg_size"):
                loss_kwargs['esimcse_neg_size'] = args.esimcse_neg_size
        
        loss_fn = loss_cls(**loss_kwargs)
        logging.warning("Loss object created inside epoch loop - queue will reset each epoch! Consider passing loss_fn as parameter.")

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    esimcse_loss_m = AverageMeter()  # Track pure ESimCSE loss separately
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    if hasattr(args, "one_iter") and args.one_iter is True:
        # hack for big dataset using one iterator to run across 400M epoch.
        if not hasattr(data['train'], "dataloader_iter"):
            print(f"running dataloader across epochs ({args.train_num_samples} examples per epoch).")
            data['train'].dataloader_iter = iter(dataloader)
        batch_iter = data['train'].dataloader_iter
    else:
        batch_iter = iter(dataloader)

    for i in range(num_batches_per_epoch):
        if i < epoch_step:  # skip to the right i when resuming happens.
            continue
        batch = next(batch_iter)
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        
        # Debug log for first batch
        if i == 0 and is_master(args):
            logging.info(f"Starting epoch {epoch}, processing first batch...")

        batch_data = to_device(batch, device, args)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        
        images = None

        if args.loss == "ESimCSECLIPLoss":
            # ESimCSE + CLIP mode: batch_data = (images, texts)
            images, texts = batch_data
            
            with autocast():
                model_unwrapped = unwrap_model(model)
                
                # Encode image
                image_features = model_unwrapped.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                
                # Encode text with augmentation 1 (original)
                text_features_aug1 = model_unwrapped.encode_text(texts)
                text_features_aug1 = F.normalize(text_features_aug1, dim=-1)
                
                # Create augmentation 2: word duplication if tokenizer available, else dropout
                if tokenizer is not None and hasattr(args, 'text_aug_word_dup') and args.text_aug_word_dup:
                    # Word duplication augmentation
                    # Decode tokens to text (skip special tokens)
                    texts_str = tokenizer.tokenizer.batch_decode(texts, skip_special_tokens=True)
                    
                    # Apply word duplication
                    dup_rate = args.text_aug_dup_rate if hasattr(args, 'text_aug_dup_rate') else 0.3
                    texts_augmented = [text_duplicate_word(text, dup_rate=dup_rate) for text in texts_str]
                    
                    # Re-tokenize augmented texts
                    texts_aug_tokens = tokenizer(texts_augmented).to(device)
                    
                    # Encode augmented text (with gradient checkpointing if available)
                    if hasattr(args, 'grad_checkpointing') and args.grad_checkpointing:
                        from torch.utils.checkpoint import checkpoint
                        text_features_aug2 = checkpoint(model_unwrapped.encode_text, texts_aug_tokens, use_reentrant=False)
                    else:
                        text_features_aug2 = model_unwrapped.encode_text(texts_aug_tokens)
                    text_features_aug2 = F.normalize(text_features_aug2, dim=-1)
                else:
                    # Dropout-based augmentation (ESimCSE default)
                    # Use different dropout masks by running forward pass twice in train mode
                    # Memory optimization: recompute with checkpointing if enabled
                    model_unwrapped.train()  # Ensure dropout is active
                    if hasattr(args, 'grad_checkpointing') and args.grad_checkpointing:
                        from torch.utils.checkpoint import checkpoint
                        text_features_aug2 = checkpoint(model_unwrapped.encode_text, texts, use_reentrant=False)
                    else:
                        text_features_aug2 = model_unwrapped.encode_text(texts)
                    text_features_aug2 = F.normalize(text_features_aug2, dim=-1)
                
                # Get momentum text features (with dropout disabled)
                with torch.no_grad():
                    model_unwrapped.momentum_encoder.eval()
                    # HFTextEncoder returns tensor directly, not dict
                    momentum_text_features = model_unwrapped.momentum_encoder(texts)
                    momentum_text_features = F.normalize(momentum_text_features, dim=-1)
                    model_unwrapped.momentum_encoder.train()
                
                # Use aug1 features for CLIP
                text_features = text_features_aug1
                
                # Get logit scale
                logit_scale = model_unwrapped.logit_scale.exp()
                
                # Compute combined loss
                total_loss = loss_fn(
                    image_features=image_features,
                    text_features=text_features,
                    logit_scale=logit_scale,
                    text_features_aug1=text_features_aug1,
                    text_features_aug2=text_features_aug2,
                    momentum_text_features=momentum_text_features
                )
                
                # Compute pure ESimCSE loss for logging (no_grad to avoid extra computation graph)
                with torch.no_grad():
                    pure_esimcse_loss = loss_fn.esimcse_loss(
                        text_features_aug1,
                        text_features_aug2,
                        momentum_embeddings=momentum_text_features
                    )
                
                # Update queue after computing loss
                loss_fn.esimcse_loss.update_queue(momentum_text_features)
                
                # Update momentum encoder parameters
                model_unwrapped.update_momentum_encoder()
                
        else:
            # Standard CLIP mode: batch_data = (images, texts)
            images, texts = batch_data
            with autocast():
                image_features, text_features, logit_scale = model(images, texts)
                total_loss = loss_fn(image_features, text_features, logit_scale)

        if torch.isfinite(total_loss).all():
            if scaler is not None:
                scaler.scale(total_loss).backward()

                if args.horovod:
                    optimizer.synchronize()
                    scaler.unscale_(optimizer)
                    if args.norm_gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                    with optimizer.skip_synchronize():
                        scaler.step(optimizer)
                else:
                    if args.norm_gradient_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                    scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if args.norm_gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                optimizer.step()

            # Note: we clamp to 4.6052 = ln(100), as in the original paper.
            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))
        else:
            logging.warn(f"Loss is {total_loss}, skip back prop.")
            import sys
            sys.exit(1)  # protect the checkpoint for debugging.


        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            # Get batch size
            if images is not None:
                batch_size = len(images)
            else:
                batch_size = args.batch_size
            
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            
            # Log with ESimCSE queue statistics if applicable
            if args.loss == 'ESimCSECLIPLoss' and hasattr(loss_fn, 'esimcse_loss'):
                # Update ESimCSE loss meter
                if 'pure_esimcse_loss' in locals():
                    esimcse_loss_m.update(pure_esimcse_loss.item(), batch_size)
                
                queue_info = ""
                if loss_fn.esimcse_loss.queue is not None:
                    queue_size = loss_fn.esimcse_loss.queue.shape[0]
                    queue_info = f"Queue: {queue_size}/{loss_fn.esimcse_loss.neg_size}"
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Total Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"ESimCSE Loss: {esimcse_loss_m.val:#.5g} ({esimcse_loss_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                    f"Logit Scale: {logit_scale_scalar:.3f} "
                    f"{queue_info}"
                )
            else:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                    f"Logit Scale: {logit_scale_scalar:.3f}"
                )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            
            # Add ESimCSE queue metrics and pure ESimCSE loss
            if args.loss == 'ESimCSECLIPLoss' and hasattr(loss_fn, 'esimcse_loss'):
                if 'pure_esimcse_loss' in locals():
                    log_data["esimcse_loss"] = esimcse_loss_m.val
                if loss_fn.esimcse_loss.queue is not None:
                    queue_size = loss_fn.esimcse_loss.queue.shape[0]
                    log_data["queue_size"] = queue_size
            
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

        if hasattr(args, "save_steps") and (step + 1) % args.save_steps == 0:
            save_checkpoint(model, optimizer, scaler, epoch, i, args, loss_fn=loss_fn)
    
        # TODO: copied from main.py, wrap as a function call.
        if hasattr(args, "eval_steps") and (step + 1) % args.eval_steps == 0: # TODO (huxu): put eval on master only?
            if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
                evaluate_ex(model, data, step, args, tb_writer)  # completed_epoch -> epoch, writer -> tb_writer
            save_checkpoint(model, optimizer, scaler, epoch, i, args, loss_fn=loss_fn)
            model.train()  # evaluate won't turn model back to train."""
    # end for





# huxu: used inside train_epoch.
def evaluate_ex(model, data, step, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, 0, args)  # huxu: epoch = 0 as a trick to bypass checking.
    metrics.update(zero_shot_metrics)

    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    if 'val' in data:  # and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):  # huxu: val anytime called.
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = to_device(batch, device, args)

                with autocast():
                    image_features, text_features, logit_scale = model(images, texts)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Step: {step} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "step": step, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Step: {step} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val_step/{name}", val, step)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val_step/{name}": val, 'step': step})

    return metrics


def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = to_device(batch, device, args)

                with autocast():
                    image_features, text_features, logit_scale = model(images, texts)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

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


def get_metrics(image_features, text_features, logit_scale):
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
