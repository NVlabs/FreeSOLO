# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

# generate free masks for train2017 images
python demo/inference_freemask.py --config-file configs/freesolo/freemask.yaml \
	--input datasets/coco/train2017/ \
	--output training_dir/instances_train2017_densecl_r101.json \
	--split -1 \
	--opts MODEL.WEIGHTS training_dir/pre-trained/DenseCL/densecl_r101_imagenet_200ep.pkl \

# generate free masks for unlabeled2017 images
python demo/inference_freemask.py --config-file configs/freesolo/freemask.yaml \
	--input datasets/coco/unlabeled2017/ \
	--output training_dir/instances_unlabeled2017_densecl_r101.json \
	--split -1 \
	--opts MODEL.WEIGHTS training_dir/pre-trained/DenseCL/densecl_r101_imagenet_200ep.pkl \

# merge train2017 and unlabeled2017 json files
python tools/merge_train2017_unlabeled2017.py
