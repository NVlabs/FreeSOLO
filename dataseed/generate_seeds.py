# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

import numpy as np
import json
import os

def subsample_idx(num_all):
    #SupPercent = [0.01, 0.1,0.5, 1, 2, 5, 10]
    SupPercent = [20.0, 25.0, 30.0]
    run_times = 5
    dict_all = {}
    for sup_p in SupPercent:
        dict_all[sup_p] = {}
        for run_i in range(run_times):
            num_label = int(sup_p / 100. * num_all)
            labeled_idx = np.random.choice(range(num_all), size=num_label, replace=False)
            dict_all[sup_p][run_i] = labeled_idx.tolist() 
    return dict_all


def gen_seeds():
    dataset_dir = '../datasets/coco/annotations'
    ann_path = os.path.join(dataset_dir, 'instances_train2017.json')
    save_path = './COCO_supervision.txt'

    ann = json.load(open(ann_path))
    images = ann['images']
    
    num_images = len(images)
    subsample_dict = subsample_idx(num_images)

    old_dict = json.load(open(save_path))
    old_dict.update(subsample_dict)
    json.dump(old_dict, open(save_path, 'w'))


if __name__ == '__main__':
    gen_seeds()
