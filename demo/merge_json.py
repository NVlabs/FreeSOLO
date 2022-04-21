# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

import os
import json
import glob
from tqdm import tqdm

paths = glob.glob('coco_unlabeled2017/*')

save_path = 'instances_unlabeled2017_densecl_r101.json'

ann_dict = dict()
ann_dict_ = json.load(open('../datasets/coco/annotations/instances_train2017.json'))
ann_dict['categories'] = ann_dict_['categories']

images_list = []
anns_list = []

for json_path in tqdm(paths):
    cur_ann_dict = json.load(open(json_path))
    images_list.extend(cur_ann_dict['images'])
    anns_list.extend(cur_ann_dict['annotations'])

print("Done: {} images; {} anns.".format(len(images_list), len(anns_list)))
ann_dict['images'] = images_list
ann_dict['annotations'] = anns_list

json.dump(ann_dict, open(save_path, 'w'))
