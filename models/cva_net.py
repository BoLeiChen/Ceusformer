"""
Model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
import copy
import matplotlib.pyplot as plt
import model_v1
import utils
from timesformer.models.vit import TimeSformer
from .conformer import Conformer


class SetCriterion(nn.Module):
    def __init__(self, weight_dict, losses, focal_alpha=0.25):
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets):

        src_logits = outputs['label']

        labels = []
        for t in targets:
            labels.append(t['labels'])
        target_classes_onehot = torch.stack(labels)

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot.float())
        losses = {
            'loss_ce': loss_ce,
            }

        return losses

    def loss_masks(self, outputs, targets):

        src_masks_ceus = outputs["seg_ceus"]

        target_masks_ceus = []
        for t in targets:
            target_masks_ceus.append(t['masks_ceus'])
        target_masks_ceus = torch.stack(target_masks_ceus).squeeze(1)

        src_masks_us = outputs["seg_us"]

        target_masks_us = []
        for t in targets:
            target_masks_us.append(t['masks_us'])
        target_masks_us = torch.stack(target_masks_us).squeeze(1)

        losses = {
            "loss_mask": utils.loss.ce_dice_iou_loss(src_masks_ceus, target_masks_ceus)
                         + utils.loss.ce_dice_iou_loss(src_masks_us, target_masks_us),
        }
        return losses

    def get_loss(self, loss, outputs, targets):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets)

    def forward(self, outputs, targets):
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        return losses


def build(args):
    device = torch.device(args.device)

    # model = model_v1.Attention_Unet_Vgg(num_class=2, in_channels=3)
    # model = TimeSformer(img_size=512, num_classes=1, num_frames=9, attention_type='divided_space_time')

    model = Conformer(patch_size=16, channel_ratio=2, num_med_block=4, embed_dim=384, img_size=512, num_frames=7, attention_type='divided_space_time',
                      depth=21, num_heads=4, mlp_ratio=2, qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.0, drop_path_rate=0.1)

    weight_dict = {
                'loss_ce': args.cls_loss_coef,
                'loss_mask': args.mask_loss_coef,
                }
    losses = ['labels', 'masks']

    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(weight_dict, losses, focal_alpha=0.25)
    criterion.to(device)

    return model, criterion
