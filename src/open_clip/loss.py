# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import utils

def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}


    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
                             F.cross_entropy(logits_per_image, labels) +
                             F.cross_entropy(logits_per_text, labels)
                     ) / 2
        return total_loss

class DispersiveLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        
        # First gather features from all GPUs if using distributed training
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # Calculate ground-truth and cache if enabled
        batch_size = logits_per_image.size(0)
        mask = torch.ones_like(logits_per_image, dtype=torch.bool)
        mask[torch.arange(batch_size), torch.arange(batch_size)] = False
        
        dispersive_loss_image = torch.sum(torch.exp(logits_per_image) * mask, dim=1) 
        dispersive_loss_text = torch.sum(torch.exp(logits_per_text) * mask, dim=1)
        
        # 对每个样本的 loss 取 log 后再取平均，确保返回标量
        dispersive_loss = torch.mean(
            (torch.log(dispersive_loss_image) + torch.log(dispersive_loss_text)) / 2
        )
        
        return dispersive_loss

class SIMCLRLoss(nn.Module):
    """
    This is the SimCLR loss in https://arxiv.org/abs/2002.05709
    The embedding vectors are assumed to have size (2 x batch_size, embedding_dim) and
    the memory layout that can be reshaped into shape (2, batch_size, embedding_dim).
    This memory layout is consistent with the SimCLR collator in
    https://github.com/facebookresearch/vissl/blob/master/vissl/data/collators/simclr_collator.py
    Config params:
        temperature (float): the temperature to be applied on the logits
    """

    def __init__(self, 
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            temperature=0.1):
        super().__init__()
        self.tau = temperature
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        
        # cache state
        self.labels = None
        self.masks = None
        self.last_local_batch_size = None

    def forward(self, q_a, q_b):
        """
        Args:
            q_a: [B, D] features from augmentation 1
            q_b: [B, D] features from augmentation 2
        """
        device = q_a.device
        
        # Normalize features
        q_a = F.normalize(q_a, dim=-1, p=2)
        q_b = F.normalize(q_b, dim=-1, p=2)

        local_batch_size = q_a.size(0)
        if isinstance(local_batch_size, torch.Tensor):
            local_batch_size = local_batch_size.item()

        # Gather features from all GPUs if distributed
        if self.world_size > 1:
            k_a, k_b = gather_features(
                q_a, q_b,
                self.local_loss, self.gather_with_grad, 
                self.rank, self.world_size, self.use_horovod
            )
        else:
            k_a, k_b = q_a, q_b

        # Create labels and masks for contrastive learning
        if (self.last_local_batch_size != local_batch_size or 
            self.labels is None or 
            self.labels.device != device):
            # Labels: diagonal elements are positive pairs
            self.labels = local_batch_size * self.rank + torch.arange(
                local_batch_size, device=device
            )
            total_batch_size = local_batch_size * self.world_size if self.world_size > 1 else local_batch_size
            # Masks: prevent comparing with itself
            self.masks = F.one_hot(self.labels, total_batch_size).float() * 1e9
            self.last_local_batch_size = local_batch_size

        logits_aa = torch.matmul(q_a, k_a.transpose(0, 1)) / self.tau
        logits_aa = logits_aa - self.masks
        logits_bb = torch.matmul(q_b, k_b.transpose(0, 1)) / self.tau
        logits_bb = logits_bb - self.masks
        logits_ab = torch.matmul(q_a, k_b.transpose(0, 1)) / self.tau
        logits_ba = torch.matmul(q_b, k_a.transpose(0, 1)) / self.tau

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), self.labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), self.labels)
        loss = (loss_a + loss_b) / 2  # divide by 2 to average over all samples

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(torch.cat([logits_ab, logits_aa], dim=1), dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return loss


class SLIPLoss(nn.Module):

    """
    SLIP: Self-supervision meets Language-Image Pre-training
    Combines CLIP loss with SimCLR self-supervised loss
    """
    def __init__(self,
            ssl_scale=0.5,
            ssl_temperature=0.1,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False):
        super().__init__()
        self.clip_loss = ClipLoss(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )
        self.ssl_loss = SIMCLRLoss(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod,
            temperature=ssl_temperature
        )
        self.ssl_scale = ssl_scale

    def forward(self, image_features, text_features, logit_scale, image_features_aug1, image_features_aug2):
        """
        Args:
            image_features: [B, D] image features for CLIP (from original or aug1)
            text_features: [B, D] text features for CLIP
            logit_scale: scalar for CLIP
            image_features_aug1: [B, D] features from first augmentation
            image_features_aug2: [B, D] features from second augmentation
        Returns:
            total_loss: scalar loss value for backprop
        """
        # CLIP loss
        clip_loss = self.clip_loss(image_features, text_features, logit_scale)
        
        # SimCLR self-supervised loss
        ssl_loss = self.ssl_loss(image_features_aug1, image_features_aug2)
        
        # Combined loss
        total_loss = clip_loss + self.ssl_scale * ssl_loss
        
        return total_loss


class FNELoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}


    def forward(self, image_features, text_features, logit_scale, mask_matrix):
        device = image_features.device
        local_batch_size = image_features.shape[0]
        
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            global_batch_size = local_batch_size * self.world_size
            
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
                
                # For local_loss, mask_matrix is already [local_batch, global_batch]
                # from train.py, so we can use it directly
                # No expansion needed
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
                
                # For non-local_loss, we need to gather masks from all ranks
                # Input mask_matrix shape: [local_batch, global_batch]
                # Output should be: [global_batch, global_batch]
                expanded_mask = torch.zeros(global_batch_size, global_batch_size, 
                                          device=device, dtype=mask_matrix.dtype)
                
                # Gather masks from all ranks
                if self.use_horovod:
                    import horovod.torch as hvd
                    # Stack masks from all ranks: [world_size, local_batch, global_batch]
                    all_masks = hvd.allgather(mask_matrix)
                    # Reshape to [global_batch, global_batch]
                    for r in range(self.world_size):
                        r_start = r * local_batch_size
                        r_end = r_start + local_batch_size
                        expanded_mask[r_start:r_end, :] = all_masks[r_start:r_end]
                else:
                    gathered_masks = [torch.zeros_like(mask_matrix) for _ in range(self.world_size)]
                    torch.distributed.all_gather(gathered_masks, mask_matrix)
                    # Concatenate masks from all ranks along the first dimension
                    for r, r_mask in enumerate(gathered_masks):
                        r_start = r * local_batch_size
                        r_end = r_start + local_batch_size
                        expanded_mask[r_start:r_end, :] = r_mask
                
                mask_matrix = expanded_mask
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        # Ensure mask shape matches logits shape
        assert mask_matrix.shape == logits_per_image.shape, \
            f"Mask shape {mask_matrix.shape} doesn't match logits shape {logits_per_image.shape}"
        
        # Mask out false negatives by setting their logits to -inf
        # This prevents them from contributing to the loss
        logits_per_image = logits_per_image.masked_fill(mask_matrix.bool(), float('-inf'))
        # logits_per_text = logits_per_text.masked_fill(mask_matrix.bool(), float('-inf'))

        total_loss = (
                             F.cross_entropy(logits_per_image, labels) +
                             F.cross_entropy(logits_per_text, labels)
                     ) / 2
        return total_loss

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class ESimCSELoss(nn.Module):
    """
    ESimCSE Loss for sentence embeddings with momentum contrast
    Based on trainers.py and models.py implementation
    
    NOTE: momentum_encoder needs to be in the main model architecture
    This loss expects to receive momentum_embeddings as input
    """
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            temperature=0.05,
            neg_size=160,
            # NOTE: momentum parameter is not used in loss - 
            # momentum encoder update should be done in model forward pass
            ):
        super().__init__()
        
        # Queue for storing historical embeddings from momentum encoder
        self.register_buffer("queue", None)
        self.neg_size = neg_size
        self.sim = Similarity(temperature)
        
        # Distribution training parameters
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.rank = rank
        self.gather_with_grad = gather_with_grad
        self.local_loss = local_loss
        
        # Cache for labels
        self.cache_labels = cache_labels
        if cache_labels:
            self.register_buffer('labels', None, persistent=False)
    
    def update_queue(self, momentum_embeddings):
        """
        Update the queue with new momentum embeddings (FIFO)
        
        Args:
            momentum_embeddings: [batch_size, hidden_dim] from momentum encoder
        
        NOTE: Call this AFTER computing loss to update queue for next iteration
        """
        with torch.no_grad():
            if self.queue is None:
                # Initialize queue
                self.queue = momentum_embeddings.clone().detach()
            else:
                # FIFO update: append new embeddings and keep last neg_size
                self.queue = torch.cat([self.queue, momentum_embeddings.clone().detach()], dim=0)
                if self.queue.size(0) > self.neg_size:
                    self.queue = self.queue[-self.neg_size:]
    
    def forward(self, z1, z2, momentum_embeddings=None):
        """
        Compute ESimCSE loss
        
        Args:
            z1: [batch_size, hidden_dim] embeddings from main encoder (view 1)
            z2: [batch_size, hidden_dim] embeddings from main encoder (view 2)
            momentum_embeddings: [batch_size, hidden_dim] embeddings from momentum encoder
                                 Used to update queue after loss computation
        
        Returns:
            loss: scalar loss value
        
        NOTE: After getting the loss, call update_queue(momentum_embeddings) 
              to update the queue for the next iteration
        """
        device = z1.device
        local_batch_size = z1.size(0)
        
        # Gather features from all GPUs if distributed training
        if self.world_size > 1:
            all_z1, all_z2 = gather_features(
                z1, z2,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod
            )
        else:
            all_z1 = z1
            all_z2 = z2
        
        # Compute cosine similarity matrix [batch_size, batch_size]
        cos_sim = self.sim(all_z1.unsqueeze(1), all_z2.unsqueeze(0))
        
        # Add queue negatives if available
        if self.neg_size > 0 and self.queue is not None:
            # Compute similarity with queue [batch_size, queue_size]
            queue_sim = self.sim(all_z1.unsqueeze(1), self.queue.unsqueeze(0))
            # Concatenate: [batch_size, batch_size + queue_size]
            cos_sim = torch.cat([cos_sim, queue_sim], dim=1)
        
        # Create labels
        batch_size = cos_sim.size(0)
        if self.cache_labels and self.labels is not None and self.labels.size(0) == batch_size:
            labels = self.labels
        else:
            labels = torch.arange(batch_size, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels = labels
        
        # Compute cross entropy loss
        loss = F.cross_entropy(cos_sim, labels)
        
        return loss


class ESimCSECLIPLoss(nn.Module):
    """
    Combined ESimCSE and CLIP loss
    Similar to SLIP but uses ESimCSE for text-text contrastive learning
    
    Usage:
        loss_fn = ESimCSECLIPLoss(esimcse_scale=0.5)
        
        # In training loop:
        # 1. Get image features from image encoder
        # 2. Get text features (z1, z2) from main text encoder with augmentation
        # 3. Get momentum_text_features from momentum text encoder
        
        loss = loss_fn(
            image_features=image_features,
            text_features=text_features_for_clip,  # Can be z1 or average of z1,z2
            logit_scale=logit_scale,
            text_features_aug1=z1,
            text_features_aug2=z2,
            momentum_text_features=momentum_text_features
        )
        
        # 4. After loss, update queue
        loss_fn.esimcse_loss.update_queue(momentum_text_features)
    """
    def __init__(
            self,
            esimcse_scale=0.5,
            esimcse_temperature=0.05,
            esimcse_neg_size=160,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False
    ):
        super().__init__()
        
        # CLIP loss for image-text alignment
        self.clip_loss = ClipLoss(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )
        
        # ESimCSE loss for text-text contrastive learning
        self.esimcse_loss = ESimCSELoss(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod,
            temperature=esimcse_temperature,
            neg_size=esimcse_neg_size
        )
        
        self.esimcse_scale = esimcse_scale
    
    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            text_features_aug1,
            text_features_aug2,
            momentum_text_features=None
    ):
        """
        Args:
            image_features: [B, D] image features for CLIP
            text_features: [B, D] text features for CLIP (can be text_features_aug1 or averaged)
            logit_scale: scalar for CLIP logit scaling
            text_features_aug1: [B, D] text features from augmentation 1 (e.g., original text)
            text_features_aug2: [B, D] text features from augmentation 2 (e.g., word duplicated)
            momentum_text_features: [B, D] text features from momentum encoder
                                    NOTE: Should be encoded with disable_dropout=True
        
        Returns:
            total_loss: scalar loss value for backprop
        
        NOTE: After forward, remember to call:
              self.esimcse_loss.update_queue(momentum_text_features)
        """
        # CLIP loss for image-text alignment
        clip_loss = self.clip_loss(image_features, text_features, logit_scale)
        
        # ESimCSE loss for text-text contrastive learning
        # NOTE: momentum_embeddings is passed but queue update happens outside
        esimcse_loss = self.esimcse_loss(
            text_features_aug1,
            text_features_aug2,
            momentum_embeddings=momentum_text_features
        )
        
        # Combined loss
        total_loss = clip_loss + self.esimcse_scale * esimcse_loss
        
        return total_loss
