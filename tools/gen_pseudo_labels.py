# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

import os
import json
import glob
from tqdm import tqdm

save_path = 'datasets/coco/annotations/instances_train2017_unlabeled2017_freesolo_pl.json'

ann_dict = json.load(open('datasets/coco/annotations/instances_train2017_unlabeled2017_densecl_r101.json'))
anns = json.load(open('training_dir/FreeSOLO_pl_1/inference/coco_instances_results.json'))
print('original {} images, {} objects.'.format(len(ann_dict['images']), len(ann_dict['annotations'])))

start_id = len(ann_dict['annotations'])
new_anns = []
for id, ann in enumerate(anns):
    # filter
    box = ann['bbox']
    h, w = ann['segmentation']['size']
    if (box[2] - box[0]) >= 0.95 * w:
        continue
    ann['id'] = id + start_id
    new_anns.append(ann)

ann_dict['annotations'] = new_anns
#ann_dict['annotations'].extend(new_anns)
json.dump(ann_dict, open(save_path, 'w'))
print('{} images, {} objects.'.format(len(ann_dict['images']), len(ann_dict['annotations'])))




