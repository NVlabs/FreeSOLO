# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

# -------------------------------------------------------------------------
# Copyright (c) 2019 the AdelaiDet authors
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Modified by Xinlong Wang
# -------------------------------------------------------------------------

import logging
import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec, batched_nms, cat, paste_masks_in_image
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import log_first_n
from fvcore.nn import sigmoid_focal_loss_jit

from .utils import imrescale, center_of_mass, point_nms, mask_nms, matrix_nms, dice_coefficient, compute_pairwise_term
from .loss import dice_loss, FocalLoss

__all__ = ["SOLOv2"]


@META_ARCH_REGISTRY.register()
class SOLOv2(nn.Module):
    """
    SOLOv2 model. Creates FPN backbone, instance branch for kernels and categories prediction,
    mask branch for unified mask features.
    Calculates and applies proper losses to class and masks.
    """

    def __init__(self, cfg):
        super().__init__()

        # get the device of the model
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.scale_ranges = cfg.MODEL.SOLOV2.FPN_SCALE_RANGES
        self.strides = cfg.MODEL.SOLOV2.FPN_INSTANCE_STRIDES
        self.sigma = cfg.MODEL.SOLOV2.SIGMA
        # Instance parameters.
        self.num_classes = cfg.MODEL.SOLOV2.NUM_CLASSES
        self.num_kernels = cfg.MODEL.SOLOV2.NUM_KERNELS
        self.num_grids = cfg.MODEL.SOLOV2.NUM_GRIDS
        self.num_embs = 128

        self.instance_in_features = cfg.MODEL.SOLOV2.INSTANCE_IN_FEATURES
        self.instance_strides = cfg.MODEL.SOLOV2.FPN_INSTANCE_STRIDES
        self.instance_in_channels = cfg.MODEL.SOLOV2.INSTANCE_IN_CHANNELS  # = fpn.
        self.instance_channels = cfg.MODEL.SOLOV2.INSTANCE_CHANNELS

        # Mask parameters.
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_in_features = cfg.MODEL.SOLOV2.MASK_IN_FEATURES
        self.mask_in_channels = cfg.MODEL.SOLOV2.MASK_IN_CHANNELS
        self.mask_channels = cfg.MODEL.SOLOV2.MASK_CHANNELS
        self.num_masks = cfg.MODEL.SOLOV2.NUM_MASKS

        # Inference parameters.
        self.max_before_nms = cfg.MODEL.SOLOV2.NMS_PRE
        self.score_threshold = cfg.MODEL.SOLOV2.SCORE_THR
        self.update_threshold = cfg.MODEL.SOLOV2.UPDATE_THR
        self.mask_threshold = cfg.MODEL.SOLOV2.MASK_THR
        self.max_per_img = cfg.MODEL.SOLOV2.MAX_PER_IMG
        self.nms_kernel = cfg.MODEL.SOLOV2.NMS_KERNEL
        self.nms_sigma = cfg.MODEL.SOLOV2.NMS_SIGMA
        self.nms_type = cfg.MODEL.SOLOV2.NMS_TYPE

        # build the backbone.
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.is_freemask = cfg.MODEL.SOLOV2.IS_FREEMASK

        if not self.is_freemask:
            # build the ins head.
            instance_shapes = [backbone_shape[f] for f in self.instance_in_features]
            self.ins_head = SOLOv2InsHead(cfg, instance_shapes)

            # build the mask head.
            mask_shapes = [backbone_shape[f] for f in self.mask_in_features]
            self.mask_head = SOLOv2MaskHead(cfg, mask_shapes)

        if cfg.MODEL.SOLOV2.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")
            #for p in self.mask_head.parameters():
            #    p.requires_grad = False
            #print("froze mask head parameters")

        # loss
        self.ins_loss_weight = cfg.MODEL.SOLOV2.LOSS.DICE_WEIGHT
        self.focal_loss_alpha = cfg.MODEL.SOLOV2.LOSS.FOCAL_ALPHA
        self.focal_loss_gamma = cfg.MODEL.SOLOV2.LOSS.FOCAL_GAMMA
        self.focal_loss_weight = cfg.MODEL.SOLOV2.LOSS.FOCAL_WEIGHT

        # free
        self.bottom_pixels_removed = 10
        self.pairwise_size = 3
        self.pairwise_dilation = 2
        self.pairwise_color_thresh = 0.3
        self._warmup_iters = 1000
        self.register_buffer("_iter", torch.zeros([1]))

        # image transform
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DetectionTransform` .
                Each item in the list contains the inputs for one image.
            For now, each item in the list is a dict that contains:
                image: Tensor, image in (C, H, W) format.
                instances: Instances
                Other information that's included in the original dicts, such as:
                    "height", "width" (int): the output resolution of the model, used in inference.
                        See :meth:`postprocess` for details.
         Returns:
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)


        # ins branch
        ins_features = [features[f] for f in self.instance_in_features]
        ins_features = self.split_feats(ins_features)
        cate_pred, kernel_pred = self.ins_head(ins_features)

        # mask branch
        mask_features = [features[f] for f in self.mask_in_features]
        mask_pred = self.mask_head(mask_features)

        if self.training:
            """
            get_ground_truth.
            return loss and so on.
            """
            mask_feat_size = mask_pred.size()[-2:]
            targets = self.get_ground_truth(gt_instances, mask_feat_size)
            losses = self.loss(cate_pred, kernel_pred, mask_pred, targets)
            return losses
        else:
            # point nms.
            cate_pred = [point_nms(cate_p.sigmoid(), kernel=2).permute(0, 2, 3, 1)
                         for cate_p in cate_pred]
            # do inference for results.
            results = self.inference(cate_pred, kernel_pred, mask_pred, images.image_sizes, batched_inputs)
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @torch.no_grad()
    def get_ground_truth(self, gt_instances, mask_feat_size=None):
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list, cate_soft_label_list = [], [], [], [], []
        if len(gt_instances) and gt_instances[0].has('image_color_similarity'):
            image_color_similarity_list = [gt_instance.image_color_similarity for gt_instance in gt_instances]
        else:
            image_color_similarity_list = []

        for img_idx in range(len(gt_instances)):
            cur_ins_label_list, cur_cate_label_list, \
            cur_ins_ind_label_list, cur_grid_order_list, cur_cate_soft_label_list = \
                self.get_ground_truth_single(img_idx, gt_instances,
                                             mask_feat_size=mask_feat_size)
            ins_label_list.append(cur_ins_label_list)
            cate_label_list.append(cur_cate_label_list)
            ins_ind_label_list.append(cur_ins_ind_label_list)
            grid_order_list.append(cur_grid_order_list)
            cate_soft_label_list.append(cur_cate_soft_label_list)
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list, cate_soft_label_list, image_color_similarity_list

    def get_ground_truth_single(self, img_idx, gt_instances, mask_feat_size):
        gt_bboxes_raw = gt_instances[img_idx].gt_boxes.tensor
        gt_labels_raw = gt_instances[img_idx].gt_classes
        gt_masks_raw = gt_instances[img_idx].gt_masks
        device = gt_labels_raw.device
        if hasattr(gt_instances[img_idx], 'gt_embs'):
            gt_embs_raw = gt_instances[img_idx].gt_embs
        else: # empty soft labels
            gt_embs_raw = torch.zeros([gt_labels_raw.shape[0], self.num_embs], device=device)
        if not torch.is_tensor(gt_masks_raw):
            gt_masks_raw = gt_masks_raw.tensor

        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        grid_order_list = []
        emb_label_list = []
        for (lower_bound, upper_bound), stride, num_grid \
                in zip(self.scale_ranges, self.strides, self.num_grids):

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            num_ins = len(hit_indices)

            ins_label = []
            grid_order = []
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            cate_label = torch.fill_(cate_label, self.num_classes)
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)
            emb_label = torch.zeros([num_grid, num_grid, self.num_embs], device=device)

            if num_ins == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
                emb_label_list.append(emb_label)
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices, ...]
            gt_embs = gt_embs_raw[hit_indices]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            # mass center
            center_ws, center_hs = center_of_mass(gt_masks)
            valid_mask_flags = gt_masks.sum(dim=-1).sum(dim=-1) > 0

            output_stride = 4
            gt_masks = gt_masks.permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()
            gt_masks = imrescale(gt_masks, scale=1. / output_stride)
            if len(gt_masks.shape) == 2:
                gt_masks = gt_masks[..., None]
            gt_masks = torch.from_numpy(gt_masks).to(dtype=torch.uint8, device=device).permute(2, 0, 1)
            for seg_mask, gt_label, gt_emb, half_h, half_w, center_h, center_w, valid_mask_flag in zip(gt_masks, gt_labels, gt_embs,
                                                                                               half_hs, half_ws,
                                                                                               center_hs, center_ws,
                                                                                               valid_mask_flags):
                if not valid_mask_flag:
                    continue
                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(coord_w - 1, left_box)
                right = min(right_box, coord_w + 1)

                cate_label[top:(down + 1), left:(right + 1)] = gt_label
                emb_label[top:(down + 1), left:(right + 1)] = gt_emb
                #cate_label[coord_h, coord_w] = gt_label
                #cate_soft_label[coord_h, coord_w] = gt_soft_label
                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        label = int(i * num_grid + j)

                        cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                                    device=device)
                        cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_label.append(cur_ins_label)
                        ins_ind_label[label] = True
                        grid_order.append(label)
            if len(ins_label) == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
            else:
                ins_label = torch.stack(ins_label, 0)
            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)
            emb_label_list.append(emb_label)
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list, emb_label_list

    def loss(self, cate_preds, kernel_preds, emb_preds, ins_pred, targets, pseudo=False):
        self._iter += 1
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list, emb_label_list, image_color_similarity_list = targets

        # ins
        ins_labels = [torch.cat([ins_labels_level_img
                                 for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]
        if len(image_color_similarity_list):
            image_color_similarity = []
            for level_idx in range(len(ins_label_list[0])):
                level_image_color_similarity = []
                for img_idx in range(len(ins_label_list)):
                    num = ins_label_list[img_idx][level_idx].shape[0]
                    cur_image_color_sim = image_color_similarity_list[img_idx][[0]].expand(num, -1, -1, -1)
                    level_image_color_similarity.append(cur_image_color_sim)
                image_color_similarity.append(torch.cat(level_image_color_similarity))
        else:
            image_color_similarity = ins_labels.copy()


        kernel_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                         for kernel_preds_level_img, grid_orders_level_img in
                         zip(kernel_preds_level, grid_orders_level)]
                        for kernel_preds_level, grid_orders_level in zip(kernel_preds, zip(*grid_order_list))]
        # generate masks
        ins_pred_list = []
        for b_kernel_pred in kernel_preds:
            b_mask_pred = []
            for idx, kernel_pred in enumerate(b_kernel_pred):

                if kernel_pred.size()[-1] == 0:
                    continue
                cur_ins_pred = ins_pred[idx, ...]
                H, W = cur_ins_pred.shape[-2:]
                N, I = kernel_pred.shape
                cur_ins_pred = cur_ins_pred.unsqueeze(0)
                kernel_pred = kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                cur_ins_pred = F.conv2d(cur_ins_pred, kernel_pred, stride=1).view(-1, H, W)
                b_mask_pred.append(cur_ins_pred)
            if len(b_mask_pred) == 0:
                b_mask_pred = None
            else:
                b_mask_pred = torch.cat(b_mask_pred, 0)
            ins_pred_list.append(b_mask_pred)

        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()

        # dice loss
        loss_ins = []
        loss_ins_max = []
        loss_pairwise = []
        for input, target, cur_image_color_similarity  in zip(ins_pred_list, ins_labels, image_color_similarity):
            if input is None:
                continue
            input_scores = torch.sigmoid(input)
            box_target = target.max(dim=1, keepdim=True)[0].expand(-1, target.shape[1], -1) * target.max(dim=2, keepdim=True)[0].expand(-1, -1, target.shape[2])

            mask_losses_y = dice_coefficient(
                input_scores.max(dim=1, keepdim=True)[0],
                target.max(dim=1, keepdim=True)[0]
            )
            mask_losses_x = dice_coefficient(
                input_scores.max(dim=2, keepdim=True)[0],
                target.max(dim=2, keepdim=True)[0]
            )
            loss_ins_max.append((mask_losses_y + mask_losses_x).mean())

            mask_losses_y = dice_coefficient(
                input_scores.mean(dim=1, keepdim=True),
                target.float().mean(dim=1, keepdim=True),
            )
            mask_losses_x = dice_coefficient(
                input_scores.mean(dim=2, keepdim=True),
                target.float().mean(dim=2, keepdim=True),
            )
            loss_ins.append((mask_losses_y + mask_losses_x).mean())

            pairwise_losses = compute_pairwise_term(
                input[:, None, ...], self.pairwise_size,
                self.pairwise_dilation
            )
            weights = (cur_image_color_similarity >= self.pairwise_color_thresh).float() * box_target[:, None, ...].float()
            #weights = (image_color_similarity >= self.pairwise_color_thresh).float()
            cur_loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)
            warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)
            cur_loss_pairwise = cur_loss_pairwise * warmup_factor
            loss_pairwise.append(cur_loss_pairwise)

        if not loss_ins_max:
            loss_ins_max = 0 * ins_pred.sum()
        else:
            loss_ins_max = torch.stack(loss_ins_max).mean()
            loss_ins_max = loss_ins_max * self.ins_loss_weight * 1.0
        if not loss_ins:
            loss_ins = 0 * ins_pred.sum()
        else:
            loss_ins = torch.stack(loss_ins).mean()
            loss_ins = loss_ins * self.ins_loss_weight * 0.1
        if not loss_pairwise:
            loss_pairwise = 0 * ins_pred.sum()
        else:
            loss_pairwise = torch.stack(loss_pairwise).mean()
            loss_pairwise =  1. * loss_pairwise


        # cate
        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)
        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)
        # prepare one_hot
        pos_inds = torch.nonzero((flatten_cate_labels != self.num_classes) & (flatten_cate_labels != -1)).squeeze(1)
        num_ins = len(pos_inds)

        flatten_cate_labels_oh = torch.zeros_like(flatten_cate_preds)
        flatten_cate_labels_oh[pos_inds, flatten_cate_labels[pos_inds]] = 1

        if pseudo:
            flatten_cate_labels_oh = flatten_cate_labels_oh[pos_inds]
            flatten_cate_preds = flatten_cate_preds[pos_inds]

        if len(flatten_cate_preds):
            loss_cate = self.focal_loss_weight * sigmoid_focal_loss_jit(flatten_cate_preds, flatten_cate_labels_oh,
                                                                        gamma=self.focal_loss_gamma,
                                                                        alpha=self.focal_loss_alpha,
                                                                        reduction="sum") / (num_ins + 1)
        else:
            loss_cate = 0 * flatten_cate_preds.sum()

        emb_labels = [
            torch.cat([emb_labels_level_img.reshape(-1, self.num_embs)
                       for emb_labels_level_img in emb_labels_level])
            for emb_labels_level in zip(*emb_label_list)
        ]
        flatten_emb_labels = torch.cat(emb_labels)
        emb_preds = [
            emb_pred.permute(0, 2, 3, 1).reshape(-1, self.num_embs)
            for emb_pred in emb_preds
        ]
        flatten_emb_preds = torch.cat(emb_preds)

        if num_ins:
            flatten_emb_labels = flatten_emb_labels[pos_inds]
            flatten_emb_preds = flatten_emb_preds[pos_inds]
            flatten_emb_preds = flatten_emb_preds / flatten_emb_preds.norm(dim=1, keepdim=True)
            flatten_emb_labels = flatten_emb_labels / flatten_emb_labels.norm(dim=1, keepdim=True)
            loss_emb = 1 - (flatten_emb_preds * flatten_emb_labels).sum(dim=-1)
            loss_emb = loss_emb.mean() * 4.0
        else:
            loss_emb = 0 * flatten_emb_preds.sum()

        return {'loss_ins': loss_ins,
                'loss_ins_max': loss_ins_max,
                'loss_pairwise': loss_pairwise,
                'loss_emb': flatten_emb_preds.sum()*0.,
                'loss_cate': loss_cate}

    @staticmethod
    def split_feats(feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear'))

    def inference(self, pred_cates, pred_kernels, pred_embs, pred_masks, cur_sizes, images, keep_train_size=False):
        assert len(pred_cates) == len(pred_kernels)

        results = []
        num_ins_levels = len(pred_cates)
        for img_idx in range(len(images)):
            # image size.
            ori_img = images[img_idx]
            height, width = ori_img["height"], ori_img["width"]
            ori_size = (height, width)

            # prediction.
            pred_cate = [pred_cates[i][img_idx].view(-1, self.num_classes).detach()
                         for i in range(num_ins_levels)]
            pred_kernel = [pred_kernels[i][img_idx].permute(1, 2, 0).view(-1, self.num_kernels).detach()
                           for i in range(num_ins_levels)]
            pred_emb = [pred_embs[i][img_idx].view(-1, self.num_embs).detach()
                         for i in range(num_ins_levels)]
            pred_mask = pred_masks[img_idx, ...].unsqueeze(0)

            pred_cate = torch.cat(pred_cate, dim=0)
            pred_kernel = torch.cat(pred_kernel, dim=0)
            pred_emb = torch.cat(pred_emb, dim=0)

            # inference for single image.
            result = self.inference_single_image(pred_cate, pred_kernel, pred_emb, pred_mask,
                                                 cur_sizes[img_idx], ori_size, keep_train_size)
            results.append({"instances": result})
        return results

    def inference_single_image(
            self, cate_preds, kernel_preds, emb_preds, seg_preds, cur_size, ori_size, keep_train_size=False
    ):
        # overall info.
        h, w = cur_size
        f_h, f_w = seg_preds.size()[-2:]
        ratio = max(math.ceil(h / f_h), math.ceil(w / f_w))
        upsampled_size_out = (int(f_h * ratio), int(f_w * ratio))

        # process.
        inds = (cate_preds > self.score_threshold)
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            results = Instances(ori_size)
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            results.scores = torch.tensor([])
            results.category_scores = torch.tensor([])
            results.maskness = torch.tensor([])
            results.pred_embs = torch.tensor([])
            return results

        # cate_labels & kernel_preds
        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]
        emb_preds = emb_preds[inds[:, 0]]
        
        if keep_train_size: # used in self-training
            # sort and keep top nms_pre
            sort_inds = torch.argsort(cate_scores, descending=True)
            max_pseudo_labels =self.max_before_nms
            if len(sort_inds) > max_pseudo_labels:
                sort_inds = sort_inds[:max_pseudo_labels]
            cate_scores = cate_scores[sort_inds]
            cate_labels = cate_labels[sort_inds]
            kernel_preds = kernel_preds[sort_inds]
            emb_preds = emb_preds[sort_inds]
            inds = inds[sort_inds]

        # trans vector.
        size_trans = cate_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(size_trans[-1])

        n_stage = len(self.num_grids)
        strides[:size_trans[0]] *= self.instance_strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.instance_strides[ind_]
        strides = strides[inds[:, 0]]

        # mask encoding.
        N, I = kernel_preds.shape
        kernel_preds = kernel_preds.view(N, I, 1, 1)
        seg_preds = F.conv2d(seg_preds, kernel_preds, stride=1).squeeze(0).sigmoid()

        # mask.
        seg_masks = seg_preds > self.mask_threshold
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            results = Instances(ori_size)
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            results.scores = torch.tensor([])
            results.category_scores = torch.tensor([])
            results.maskness = torch.tensor([])
            results.pred_embs = torch.tensor([])
            return results

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]
        emb_preds = emb_preds[keep, :]

        # maskness.
        seg_masks = seg_preds > self.mask_threshold
        maskness = (seg_preds * seg_masks.float()).sum((1, 2)) / seg_masks.sum((1, 2))
        
        scores = cate_scores * maskness

        # sort and keep top nms_pre
        sort_inds = torch.argsort(scores, descending=True)
        if len(sort_inds) > self.max_before_nms:
            sort_inds = sort_inds[:self.max_before_nms]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        scores = scores[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]
        maskness = maskness[sort_inds]
        emb_preds = emb_preds[sort_inds]

        if self.nms_type == "matrix":
            # matrix nms & filter.
            scores = matrix_nms(cate_labels, seg_masks, sum_masks, scores,
                                     sigma=self.nms_sigma, kernel=self.nms_kernel)
            keep = scores >= self.update_threshold
        elif self.nms_type == "mask":
            # original mask nms.
            keep = mask_nms(cate_labels, seg_masks, sum_masks, scores,
                            nms_thr=self.mask_threshold)
        else:
            raise NotImplementedError

        if keep.sum() == 0:
            results = Instances(ori_size)
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            results.scores = torch.tensor([])
            results.category_scores = torch.tensor([])
            results.maskness = torch.tensor([])
            results.pred_embs = torch.tensor([])
            return results

        seg_preds = seg_preds[keep, :, :]
        scores = scores[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]
        maskness = maskness[keep]
        emb_preds = emb_preds[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(scores, descending=True)
        if len(sort_inds) > self.max_per_img:
            sort_inds = sort_inds[:self.max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        scores = scores[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]
        maskness = maskness[sort_inds]
        emb_preds = emb_preds[sort_inds]

        # reshape to original size.
        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]
        if keep_train_size: # for self-training
            seg_masks = seg_preds.squeeze(0)
        else:
            seg_masks = F.interpolate(seg_preds,
                                      size=ori_size,
                                      mode='bilinear').squeeze(0)
        seg_masks = seg_masks > self.mask_threshold

        sum_masks = seg_masks.sum((1, 2)).float()
        # filter.
        keep = sum_masks > 0
        scores = scores[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]
        seg_masks = seg_masks[keep]
        maskness = maskness[keep]
        emb_preds = emb_preds[keep]

        results = Instances(seg_masks.shape[1:])
        results.pred_classes = cate_labels
        results.scores = scores
        results.category_scores = cate_scores
        results.maskness = maskness
        results.pred_masks = seg_masks
        # normalize the embeddings
        results.pred_embs = emb_preds / emb_preds.norm(dim=-1, keepdim=True)

        # get bbox from mask
        #pred_boxes = torch.zeros(seg_masks.size(0), 4)
        #for i in range(seg_masks.size(0)):
        #    mask = seg_masks[i].squeeze()
        #    ys, xs = torch.where(mask)
        #    pred_boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()]).float()

        width_proj = seg_masks.max(1)[0]
        height_proj = seg_masks.max(2)[0]
        width, height = width_proj.sum(1), height_proj.sum(1)
        center_ws, _ = center_of_mass(width_proj[:, None, :])
        _, center_hs = center_of_mass(height_proj[:, :, None])
        pred_boxes = torch.stack([center_ws-0.5*width, center_hs-0.5*height, center_ws+0.5*width, center_hs+0.5*height], 1)
        results.pred_boxes = Boxes(pred_boxes)

        return results


class SOLOv2InsHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        SOLOv2 Instance Head.
        """
        super().__init__()
        # fmt: off
        self.num_classes = cfg.MODEL.SOLOV2.NUM_CLASSES
        self.num_kernels = cfg.MODEL.SOLOV2.NUM_KERNELS
        self.num_grids = cfg.MODEL.SOLOV2.NUM_GRIDS
        self.instance_in_features = cfg.MODEL.SOLOV2.INSTANCE_IN_FEATURES
        self.instance_strides = cfg.MODEL.SOLOV2.FPN_INSTANCE_STRIDES
        self.instance_in_channels = cfg.MODEL.SOLOV2.INSTANCE_IN_CHANNELS  # = fpn.
        self.instance_channels = cfg.MODEL.SOLOV2.INSTANCE_CHANNELS
        self.num_embs = 128
        # Convolutions to use in the towers
        self.type_dcn = cfg.MODEL.SOLOV2.TYPE_DCN
        self.num_levels = len(self.instance_in_features)
        assert self.num_levels == len(self.instance_strides), \
            print("Strides should match the features.")
        # fmt: on

        head_configs = {"cate": (cfg.MODEL.SOLOV2.NUM_INSTANCE_CONVS,
                                 cfg.MODEL.SOLOV2.USE_DCN_IN_INSTANCE,
                                 False),
                        "kernel": (cfg.MODEL.SOLOV2.NUM_INSTANCE_CONVS,
                                   cfg.MODEL.SOLOV2.USE_DCN_IN_INSTANCE,
                                   cfg.MODEL.SOLOV2.USE_COORD_CONV)
                        }

        norm = None if cfg.MODEL.SOLOV2.NORM == "none" else cfg.MODEL.SOLOV2.NORM
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, \
            print("Each level must have the same channel!")
        in_channels = in_channels[0]
        assert in_channels == cfg.MODEL.SOLOV2.INSTANCE_IN_CHANNELS, \
            print("In channels should equal to tower in channels!")

        for head in head_configs:
            tower = []
            num_convs, use_deformable, use_coord = head_configs[head]
            for i in range(num_convs):
                conv_func = nn.Conv2d
                if i == 0:
                    if use_coord:
                        chn = self.instance_in_channels + 2
                    else:
                        chn = self.instance_in_channels
                else:
                    chn = self.instance_channels

                tower.append(conv_func(
                    chn, self.instance_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None
                ))
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, self.instance_channels))
                tower.append(nn.ReLU(inplace=True))
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.cate_pred = nn.Conv2d(
            self.instance_channels, self.num_classes,
            kernel_size=3, stride=1, padding=1
        )
        self.kernel_pred = nn.Conv2d(
            self.instance_channels, self.num_kernels,
            kernel_size=3, stride=1, padding=1
        )
        self.emb_pred = nn.Conv2d(
            self.instance_channels, self.num_embs,
            kernel_size=3, stride=1, padding=1
        )

        if cfg.MODEL.SOLOV2.FREEZE:
            #for modules in [self.cate_tower, self.kernel_tower, self.kernel_pred]:
            for modules in [self.cate_tower, self.kernel_tower]:
                for p in modules.parameters():
                    p.requires_grad = False
            print("froze ins head parameters")

        for modules in [
            self.cate_tower, self.kernel_tower,
            self.cate_pred, self.kernel_pred, self.emb_pred
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.SOLOV2.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cate_pred.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.
        Returns:
            pass
        """
        cate_pred = []
        kernel_pred = []
        emb_pred = []

        for idx, feature in enumerate(features):
            ins_kernel_feat = feature
            # concat coord
            x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], device=ins_kernel_feat.device)
            y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], device=ins_kernel_feat.device)
            y, x = torch.meshgrid(y_range, x_range)
            y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
            x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
            coord_feat = torch.cat([x, y], 1)
            ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)

            # individual feature.
            kernel_feat = ins_kernel_feat
            seg_num_grid = self.num_grids[idx]
            kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear')
            cate_feat = kernel_feat[:, :-2, :, :]

            # kernel
            kernel_feat = self.kernel_tower(kernel_feat)
            kernel_pred.append(self.kernel_pred(kernel_feat))

            # cate
            cate_feat = self.cate_tower(cate_feat)
            cate_pred.append(self.cate_pred(cate_feat))

            # emb
            emb_pred.append(self.emb_pred(cate_feat))
        return cate_pred, kernel_pred, emb_pred


class SOLOv2MaskHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        SOLOv2 Mask Head.
        """
        super().__init__()
        # fmt: off
        self.mask_on = cfg.MODEL.MASK_ON
        self.num_masks = cfg.MODEL.SOLOV2.NUM_MASKS
        self.mask_in_features = cfg.MODEL.SOLOV2.MASK_IN_FEATURES
        self.mask_in_channels = cfg.MODEL.SOLOV2.MASK_IN_CHANNELS
        self.mask_channels = cfg.MODEL.SOLOV2.MASK_CHANNELS
        self.num_levels = len(input_shape)
        assert self.num_levels == len(self.mask_in_features), \
            print("Input shape should match the features.")
        # fmt: on
        norm = None if cfg.MODEL.SOLOV2.NORM == "none" else cfg.MODEL.SOLOV2.NORM

        self.convs_all_levels = nn.ModuleList()
        for i in range(self.num_levels):
            convs_per_level = nn.Sequential()
            if i == 0:
                conv_tower = list()
                conv_tower.append(nn.Conv2d(
                    self.mask_in_channels, self.mask_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None
                ))
                if norm == "GN":
                    conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                conv_tower.append(nn.ReLU(inplace=False))
                convs_per_level.add_module('conv' + str(i), nn.Sequential(*conv_tower))
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    chn = self.mask_in_channels + 2 if i == 3 else self.mask_in_channels
                    conv_tower = list()
                    conv_tower.append(nn.Conv2d(
                        chn, self.mask_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=norm is None
                    ))
                    if norm == "GN":
                        conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                    conv_tower.append(nn.ReLU(inplace=False))
                    convs_per_level.add_module('conv' + str(j), nn.Sequential(*conv_tower))
                    upsample_tower = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module(
                        'upsample' + str(j), upsample_tower)
                    continue
                conv_tower = list()
                conv_tower.append(nn.Conv2d(
                    self.mask_channels, self.mask_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None
                ))
                if norm == "GN":
                    conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                conv_tower.append(nn.ReLU(inplace=False))
                convs_per_level.add_module('conv' + str(j), nn.Sequential(*conv_tower))
                upsample_tower = nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=False)
                convs_per_level.add_module('upsample' + str(j), upsample_tower)

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                self.mask_channels, self.num_masks,
                kernel_size=1, stride=1,
                padding=0, bias=norm is None),
            nn.GroupNorm(32, self.num_masks),
            nn.ReLU(inplace=True)
        )

        for modules in [self.convs_all_levels, self.conv_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.
        Returns:
            pass
        """
        assert len(features) == self.num_levels, \
            print("The number of input features should be equal to the supposed level.")

        # bottom features first.
        feature_add_all_level = self.convs_all_levels[0](features[0])
        for i in range(1, self.num_levels):
            mask_feat = features[i]
            if i == 3:  # add for coord.
                x_range = torch.linspace(-1, 1, mask_feat.shape[-1], device=mask_feat.device)
                y_range = torch.linspace(-1, 1, mask_feat.shape[-2], device=mask_feat.device)
                y, x = torch.meshgrid(y_range, x_range)
                y = y.expand([mask_feat.shape[0], 1, -1, -1])
                x = x.expand([mask_feat.shape[0], 1, -1, -1])
                coord_feat = torch.cat([x, y], 1)
                mask_feat = torch.cat([mask_feat, coord_feat], 1)
            # add for top features.
            feature_add_all_level += self.convs_all_levels[i](mask_feat)

        mask_pred = self.conv_pred(feature_add_all_level)
        return mask_pred
