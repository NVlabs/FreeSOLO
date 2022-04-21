# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

#export GLOO_SOCKET_IFNAME=eth0
python train_net.py \
	--dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 50000 )) \
	--num-gpus 8 \
	--config configs/freesolo/freesolo_30k.yaml \
	DATASETS.TRAIN '("coco_2017_train_unlabeled_freesolo_pl",)' \
	OUTPUT_DIR training_dir/FreeSOLO_pl\
