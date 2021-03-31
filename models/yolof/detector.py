#
# Modified by chang liu
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List
import copy
import torch
import torch.nn.functional as F
from torch import Tensor,nn
from typing import Dict, List, Tuple
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.layers import batched_nms, cat, nonzero_tuple

from detectron2.structures import Boxes, ImageList, Instances
from .decoder import Decoder
from .encoder import DilatedEncoder
from .utils import permute_to_N_HWA_K, YOLOFBox2BoxTransform, UniformMatcher
from .loss import Losses


__all__ = ["YOLOF"]


@META_ARCH_REGISTRY.register()
class YOLOF(nn.Module):
    """
    Implement YOLOF: https://arxiv.org/pdf/2103.09460.pdf
    """
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone_level = cfg.MODEL.YOLOF.ENCODER.BACKBONE_LEVEL
        self.backbone = build_backbone(cfg)
        self.nums_classes = cfg.MODEL.YOLOF.DECODER.NUM_CLASSES

        # build anchor generator
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[self.backbone_level]]
        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)

        # build encode decode
        self.encoder = DilatedEncoder(cfg, backbone_shape)
        self.decoder = Decoder(cfg)

        # prepare ground truth
        self.box2box_transform = YOLOFBox2BoxTransform(
                weights=cfg.MODEL.YOLOF.BOX_TRANSFORM.BBOX_REG_WEIGHTS,
                add_ctr_clamp=cfg.MODEL.YOLOF.BOX_TRANSFORM.ADD_CTR_CLAMP,
                ctr_clamp=cfg.MODEL.YOLOF.BOX_TRANSFORM.CTR_CLAMP
            )
        self.anchor_matcher = UniformMatcher(cfg.MODEL.YOLOF.MATCHER.TOPK)

        # build loss
        self.losses = Losses(cfg)

        # get normalizer
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts,
                such as:
                * "height", "width" (int): the output resolution of the model,
                  used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            in training, dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used
                during training only.
            in inference, the standard output format, described in
            :doc:`/tutorials/models`.
        """
        nums_images = len(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[self.backbone_level]]
        anchor_image = self.anchor_generator(features)
        anchors = [copy.deepcopy(anchor_image) for _ in range(nums_images)]
        pred_logits, pred_anchor_deltas = self.decoder(
            self.encoder(features[0])
        )

        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(pred_logits, self.nums_classes)]
        pred_anchor_deltas = [permute_to_N_HWA_K(pred_anchor_deltas, 4)]

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[
                0], "Instance annotations are missing in training"
            gt_instances = [x["instances"].to(self.device) for x in
                            batched_inputs]
            indices = self.get_ground_truth(
                anchors, pred_anchor_deltas, gt_instances)
            loss = self.losses(
                indices, gt_instances, anchors,
                pred_logits, pred_anchor_deltas
            )
            return loss
        else:
            results = self.inference(

            )

    def inference(self):


    @torch.no_grad()
    def get_ground_truth(self, anchors, bbox_preds, targets):
        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
        N = len(anchors)

        all_anchors = Boxes.cat(anchors).tensor.reshape(N, -1, 4)

        box_delta = cat(bbox_preds, dim=1)

        # box_pred: xyxy; targets: xyxy
        box_pred = self.box2box_transform.apply_deltas(box_delta, all_anchors)
        indices = self.anchor_matcher(box_pred, all_anchors, targets)
        return indices

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        # padding
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        return images




