# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

import argparse
import json
import os

def split_json():
    dataset_dir = '../datasets/coco/annotations'
    ann_path = os.path.join(dataset_dir, 'instances_train2017.json')
    save_path_labeled = os.path.join(dataset_dir, 'instances_train2017_sup1_seed1_usemask.json')

    split_txt_file = '../dataseed/COCO_supervision.txt'
    split = json.load(open(split_txt_file))
    selected_img_idx = split['1.0']['1'] # sup10 and seed 1

    ann = json.load(open(ann_path))
    images = ann['images']
    anns = ann['annotations']

    labeled_images, unlabeled_images = [], []
    # split images
    for idx, cur_image in enumerate(images):
        if idx in selected_img_idx: # labeled
            labeled_images.append(cur_image)
        else:
            unlabeled_images.append(cur_image)

    # split annotations
    selected_images_ids = [image['id'] for image in labeled_images]
    selected_anns = []
    cnt = 0
    for cur_ann in anns:
        if cur_ann['image_id'] in selected_images_ids:
            #selected_anns.append(cur_ann)
            cur_ann['usemask'] = 1
            cnt += 1
        else:
            cur_ann['usemask'] = 0
    print(cnt, len(anns))
    json.dump(ann, open(save_path_labeled, 'w'))


if __name__ == '__main__':
    split_json()          
