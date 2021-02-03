# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_centernet_config(cfg):
    """
    Add config for tridentnet.
    """

    cfg.MODEL.CENTERNET = CN()
    cfg.MODEL.CENTERNET.DECONV_CHANNEL = [512, 256, 128, 64]
    cfg.MODEL.CENTERNET.DECONV_KERNEL = [4, 4, 4]
    cfg.MODEL.CENTERNET.NUM_CLASSES = 1
    cfg.MODEL.CENTERNET.MODULATE_DEFORM = True
    cfg.MODEL.CENTERNET.BIAS_VALUE = -2.19
    cfg.MODEL.CENTERNET.DOWN_SCALE = 4
    cfg.MODEL.CENTERNET.MIN_OVERLAP = 0.7
    cfg.MODEL.CENTERNET.TENSOR_DIM = 128
    cfg.MODEL.CENTERNET.IN_FEATURES = ["res5"]
    cfg.MODEL.CENTERNET.OUTPUT_SIZE = [128, 128]
    cfg.MODEL.CENTERNET.TEST_PIPELINES = []
    cfg.MODEL.CENTERNET.LOSS = CN()
    cfg.MODEL.CENTERNET.LOSS.CLS_WEIGHT = 1
    cfg.MODEL.CENTERNET.LOSS.WH_WEIGHT = 0.1
    cfg.MODEL.CENTERNET.LOSS.REG_WEIGHT = 1
    cfg.INPUT.FORMAT = "RGB"

    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.SOLVER.TRAIN_PIPELINES = [
        ("CenterAffine", dict(boarder=128, output_size=(512, 512), random_aug=True)),
        ("RandomBrightness", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomContrast", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomSaturation", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomLighting", dict(scale=0.1)),
    ]

