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

from detectron2.config import CfgNode as CN


def add_solo_config(cfg):
    """
    Add config for solov2.
    """
    _C = cfg
    _C.MODEL.SOLOV2 = CN()
    # Instance hyper-parameters
    _C.MODEL.SOLOV2.INSTANCE_IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    _C.MODEL.SOLOV2.FPN_INSTANCE_STRIDES = [8, 8, 16, 32, 32]
    _C.MODEL.SOLOV2.FPN_SCALE_RANGES = ((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048))
    _C.MODEL.SOLOV2.SIGMA = 0.2
    # Channel size for the instance head.
    _C.MODEL.SOLOV2.INSTANCE_IN_CHANNELS = 256
    _C.MODEL.SOLOV2.INSTANCE_CHANNELS = 512
    # Convolutions to use in the instance head.
    _C.MODEL.SOLOV2.NUM_INSTANCE_CONVS = 4
    _C.MODEL.SOLOV2.USE_DCN_IN_INSTANCE = False
    _C.MODEL.SOLOV2.TYPE_DCN = 'DCN'
    _C.MODEL.SOLOV2.NUM_GRIDS = [40, 36, 24, 16, 12]
    # Number of foreground classes.
    _C.MODEL.SOLOV2.NUM_CLASSES = 80
    _C.MODEL.SOLOV2.NUM_KERNELS = 256
    _C.MODEL.SOLOV2.NORM = "GN"
    _C.MODEL.SOLOV2.USE_COORD_CONV = True
    _C.MODEL.SOLOV2.PRIOR_PROB = 0.01

    # Mask hyper-parameters.
    # Channel size for the mask tower.
    _C.MODEL.SOLOV2.MASK_IN_FEATURES = ["p2", "p3", "p4", "p5"]
    _C.MODEL.SOLOV2.MASK_IN_CHANNELS = 256
    _C.MODEL.SOLOV2.MASK_CHANNELS = 128
    _C.MODEL.SOLOV2.NUM_MASKS = 256

    # Test cfg.
    _C.MODEL.SOLOV2.NMS_PRE = 500
    _C.MODEL.SOLOV2.SCORE_THR = 0.1
    _C.MODEL.SOLOV2.UPDATE_THR = 0.05
    _C.MODEL.SOLOV2.MASK_THR = 0.5
    _C.MODEL.SOLOV2.MAX_PER_IMG = 100
    # NMS type: matrix OR mask.
    _C.MODEL.SOLOV2.NMS_TYPE = "matrix"
    # Matrix NMS kernel type: gaussian OR linear.
    _C.MODEL.SOLOV2.NMS_KERNEL = "gaussian"
    _C.MODEL.SOLOV2.NMS_SIGMA = 2

    # Loss cfg.
    _C.MODEL.SOLOV2.LOSS = CN()
    _C.MODEL.SOLOV2.LOSS.FOCAL_USE_SIGMOID = True
    _C.MODEL.SOLOV2.LOSS.FOCAL_ALPHA = 0.25
    _C.MODEL.SOLOV2.LOSS.FOCAL_GAMMA = 2.0
    _C.MODEL.SOLOV2.LOSS.FOCAL_WEIGHT = 1.0
    _C.MODEL.SOLOV2.LOSS.DICE_WEIGHT = 3.0

    # Freeze
    _C.MODEL.SOLOV2.FREEZE = False

    # Flag for Free Mask
    _C.MODEL.SOLOV2.IS_FREEMASK = False

