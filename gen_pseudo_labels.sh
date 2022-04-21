# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

#export GLOO_SOCKET_IFNAME=eth0
python train_net.py \
	--dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 50000 )) \
	--eval-only \
	--num-gpus 8 \
	--config configs/freesolo/freesolo_30k.yaml \
	OUTPUT_DIR training_dir/FreeSOLO_pl_1 \
	DATASETS.TEST  '("coco_2017_train_unlabeled_densecl_r101",)' \
	MODEL.SOLOV2.UPDATE_THR 0.3 \
	MODEL.SOLOV2.MAX_PER_IMG 20 \
	MODEL.WEIGHTS training_dir/FreeSOLO/model_0029999.pth

# convert to annotation format
python tools/gen_pseudo_labels.py
