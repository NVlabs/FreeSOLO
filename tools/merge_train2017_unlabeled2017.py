# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

import os
import json
import glob
from tqdm import tqdm

save_path = 'datasets/coco/annotations/instances_train2017_unlabel2017_densecl_r101.json'

ann_dict_train = json.load(open('training_dir/instances_train2017_densecl_r101.json'))
ann_dict_unlabeled = json.load(open('training_dir/instances_unlabeled2017_densecl_r101.json'))

for image in ann_dict_train['images']:
    image['file_name'] = 'train2017/' + image['file_name']

for image in ann_dict_unlabeled['images']:
    image['file_name'] = 'unlabeled2017/' + image['file_name']

ann_dict_train['images'].extend(ann_dict_unlabeled['images'])
ann_dict_train['annotations'].extend(ann_dict_unlabeled['annotations'])

json.dump(ann_dict_train, open(save_path, 'w'))




