# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

import argparse
import glob
import multiprocessing as mp
import time
import tqdm
import json
import pycocotools.mask as mask_util

import numpy as np
import torch
import torch.nn.functional as F

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

import sys
sys.path.append('.')
from freesolo import add_solo_config
from freesolo.modeling.solov2.utils import matrix_nms, center_of_mass

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_solo_config(cfg)
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        #nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--split",
        type=int,
        default=0,
        help="Split id.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)


    ann_dict = dict()
    ann_dict_ = json.load(open('datasets/coco/annotations/instances_train2017.json'))
    ann_dict['categories'] = ann_dict_['categories']

    images_list = []
    anns_list = []
    ann_id = 0

    paths = glob.glob(args.input + '/*g')
    split = args.split
    if split == -1:
        save_path = args.output
        cur_paths = paths
    else:
        num_each_split = 15000                                          
        image_range = [split*num_each_split, (split+1) * num_each_split]
        save_path = args.output + str(split)
        assert image_range[0] < len(paths)                   
        cur_paths = paths[image_range[0]:min(len(paths), image_range[1])]
    for path in tqdm.tqdm(cur_paths, disable=not args.output):
        # use PIL, to be consistent with evaluation
        try:
            img = read_image(path, format="BGR")
        except:
            continue
        height, width, _ = img.shape
        img_path = path
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)

        keys = predictions['res5'][0]
        scale_factors = [1.0, 0.5, 0.25]
        queries_list = []
        for scale_factor in scale_factors:
            cur_queries = F.interpolate(keys[None, ...], scale_factor=scale_factor, mode='bilinear')[0].reshape(keys.shape[0], -1).permute(1, 0)
            num_q = len(cur_queries)
            queries_list.append(cur_queries)
        queries = torch.cat(queries_list)
        _, H, W = keys.shape
        keys = keys / keys.norm(dim=0, keepdim=True)
        queries = queries / queries.norm(dim=1, keepdim=True)
        attn = queries @ keys.reshape(keys.shape[0], -1)
        # normalize
        attn -= attn.min(-1, keepdim=True)[0]
        attn /= attn.max(-1, keepdim=True)[0]

        attn = attn.reshape(attn.shape[0], H, W)

        soft_masks = attn
        masks = soft_masks >= 0.5

        # downsample queries
        queries = F.interpolate(queries[None, ...], size=128, mode='linear')[0]

        sum_masks = masks.sum((1,2))
        keep = sum_masks > 1
        if keep.sum() == 0:
            continue
        masks = masks[keep]
        soft_masks = soft_masks[keep]
        sum_masks = sum_masks[keep]
        queries = queries[keep]

        # Matrix NMS
        maskness = (soft_masks * masks.float()).sum((1, 2)) / sum_masks
        sort_inds = torch.argsort(maskness, descending=True)
        maskness = maskness[sort_inds]
        masks = masks[sort_inds]
        sum_masks = sum_masks[sort_inds]
        soft_masks = soft_masks[sort_inds]
        queries = queries[sort_inds]
        maskness = matrix_nms(maskness*0, masks, sum_masks, maskness, sigma=2, kernel='gaussian')

        sort_inds = torch.argsort(maskness, descending=True)
        if len(sort_inds) > 20:
            sort_inds = sort_inds[:20]
        masks = masks[sort_inds]
        maskness = maskness[sort_inds]
        soft_masks = soft_masks[sort_inds]
        queries = queries[sort_inds]

        soft_masks = F.interpolate(soft_masks[None, ...], size=(height, width), mode='bilinear')[0]
        masks = (soft_masks >= 0.5).float()
        sum_masks = masks.sum((1, 2))

        # mask to box
        width_proj = masks.max(1)[0]
        height_proj = masks.max(2)[0]
        box_width, box_height = width_proj.sum(1), height_proj.sum(1)
        center_ws, _ = center_of_mass(width_proj[:, None, :])
        _, center_hs = center_of_mass(height_proj[:, :, None])
        boxes = torch.stack([center_ws-0.5*box_width, center_hs-0.5*box_height, center_ws+0.5*box_width, center_hs+0.5*box_height], 1)
        #boxes = []
        #for mask in masks.cpu().numpy():
        #    ys, xs = np.where(mask)
        #    box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        #    boxes.append(box)
        #boxes = torch.tensor(boxes, device = maskness.device)

        # filter masks on the top border or with large width
        keep = center_hs > 0.2 * height
        keep_2 = (boxes[:, 2] - boxes[:, 0]) < 0.95 * width
        keep_3 = maskness >= 0.7
        keep = keep & keep_2 & keep_3
        #
        if keep.sum() == 0:
            continue
        masks = masks[keep]
        maskness = maskness[keep]
        boxes = boxes[keep]
        queries = queries[keep]

        # coco format
        img_name = img_path.split('/')[-1].split('.')[0]
        try:
            img_id = int(img_name)
        except:
            img_id = int(img_name.split('_')[-1])
        cur_image_dict = {'file_name': img_path.split('/')[-1],
                          'height': height,
                          'width': width,
                          'id':  img_id}
        images_list.append(cur_image_dict)


        masks = masks.cpu().numpy()
        maskness = maskness.cpu().numpy()
        boxes = boxes.tolist()
        queries = queries.tolist()
        rles = [mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
                            for mask in masks]
        for idx in range(len(masks)):
            rle = rles[idx]
            rle['counts'] = rle['counts'].decode('ascii')
            cur_ann_dict = {'segmentation': rle,
                            'bbox': boxes[idx],
                            'score': float(maskness[idx]),
                            'emb': queries[idx],
                            'iscrowd': 0,
                            'image_id': img_id,
                            'category_id': 1,
                            'id':  ann_id}
            ann_id += 1
            anns_list.append(cur_ann_dict)


    ann_dict['images'] = images_list
    ann_dict['annotations'] = anns_list
    json.dump(ann_dict, open(save_path, 'w'))
    #json.dump(anns_list, open(save_path+'ann', 'w'))
    print("Done: {} images, {} annotations.".format(len(images_list), len(anns_list)))

