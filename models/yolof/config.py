from detectron2.config import CfgNode as CN


def add_yolof_config(cfg):
    """
    Add config for yolof.
    """

    # anchor_generator
    cfg.MODEL.YOLOF = CN()

    cfg.MODEL.YOLOF.ENCODER = CN()

    cfg.MODEL.YOLOF.ENCODER.BACKBONE_LEVEL = "res5"
    cfg.MODEL.YOLOF.ENCODER.IN_CHANNELS = 2048
    cfg.MODEL.YOLOF.ENCODER.NUM_CHANNELS = 512
    cfg.MODEL.YOLOF.ENCODER.BLOCK_MID_CHANNELS = 128
    cfg.MODEL.YOLOF.ENCODER.NUM_RESIDUAL_BLOCKS = 4
    cfg.MODEL.YOLOF.ENCODER.BLOCK_DILATIONS = [2, 4, 6, 8]
    cfg.MODEL.YOLOF.ENCODER.NORM = "BN"
    cfg.MODEL.YOLOF.ENCODER.ACTIVATION = "ReLU"

    cfg.MODEL.YOLOF.DECODER = CN()

    cfg.MODEL.YOLOF.DECODER.IN_CHANNELS = 512
    cfg.MODEL.YOLOF.DECODER.NUM_CLASSES = 1
    cfg.MODEL.YOLOF.DECODER.NUM_ANCHORS = 5
    cfg.MODEL.YOLOF.DECODER.CLS_NUM_CONVS = 2
    cfg.MODEL.YOLOF.DECODER.REG_NUM_CONVS = 4
    cfg.MODEL.YOLOF.DECODER.NORM = "BN"
    cfg.MODEL.YOLOF.DECODER.ACTIVATION = "ReLU"
    cfg.MODEL.YOLOF.DECODER.PRIOR_PROB = 0.01

    # YOLOF box2box transform
    cfg.MODEL.YOLOF.BOX_TRANSFORM = CN()
    cfg.MODEL.YOLOF.BOX_TRANSFORM.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    cfg.MODEL.YOLOF.BOX_TRANSFORM.ADD_CTR_CLAMP = True
    cfg.MODEL.YOLOF.BOX_TRANSFORM.CTR_CLAMP = 32

    cfg.MODEL.YOLOF.MATCHER = CN()

    cfg.MODEL.YOLOF.MATCHER.TOPK = 4
    # YOLOF ignore thresholds
    cfg.MODEL.YOLOF.POS_IGNORE_THRESHOLD = 0.15
    cfg.MODEL.YOLOF.NEG_IGNORE_THRESHOLD = 0.7

    # YOLOF losses
    cfg.MODEL.YOLOF.LOSSES = CN()
    cfg.MODEL.YOLOF.LOSSES.FOCAL_LOSS_GAMMA = 2.0
    cfg.MODEL.YOLOF.LOSSES.FOCAL_LOSS_ALPHA = 0.25
    cfg.MODEL.YOLOF.LOSSES.BBOX_REG_LOSS_TYPE = "giou"

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0]]
    cfg.MODEL.ANCHOR_GENERATOR.OFFSET = 0.0

    # YOLOF test
    cfg.MODEL.YOLOF.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.YOLOF.TOPK_CANDIDATES_TEST = 1000
    cfg.MODEL.YOLOF.NMS_THRESH_TEST = 0.6
    cfg.MODEL.YOLOF.DETECTIONS_PER_IMAGE = 100

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # transform.
    cfg.SOLVER.TRAIN_PIPELINES = [
        ("CenterAffine", dict(boarder=128, output_size=(512, 512), random_aug=True)),
        ("RandomBrightness", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomContrast", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomSaturation", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomLighting", dict(scale=0.1)),
    ]
