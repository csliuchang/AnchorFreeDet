#
# Modified by chang liu
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List
import copy
import torch
import torch.nn.functional as F
from torch import Tensor, nn
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
        self.test_score_thresh = 0.05
        self.test_nms_thresh = 0.6
        self.test_topk_candidates = 1000
        self.max_detections_per_image = 100

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
        anchors_image = self.anchor_generator(features)
        anchors = [copy.deepcopy(anchors_image) for _ in range(nums_images)]
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
                anchors_image,
                pred_logits,
                pred_anchor_deltas,
                images.image_sizes

            )
            if torch.jit.is_scripting():
                return results
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def inference(self,
                  anchors: List[Boxes],
                  pred_logits: List[Tensor],
                  pred_anchor_deltas: List[Tensor],
                  image_sizes: List[Tuple[int, int]],
                  ):
        """
    Arguments:
        anchors (list[Boxes]): A list of #feature level Boxes.
            The Boxes contain anchors of this image on the specific
            feature level.
        pred_logits, pred_anchor_deltas: list[Tensor], one per level. Each
            has shape (N, Hi * Wi * Ai, K or 4)
        image_sizes (List[(h, w)]): the input image sizes
    Returns:
        results (List[Instances]): a list of #images elements.
    """
        results: List[Instances] = []
        for img_idx, image_sizes in enumerate(image_sizes):
            pred_logits_per_image = [x[img_idx] for x in pred_logits]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            results_per_image = self.inference_single_image(
                anchors, pred_logits_per_image, deltas_per_image, image_sizes
            )
            results.append(results_per_image)
        return results

    def inference_single_image(
            self,
            anchors: List[Boxes],
            box_cls: List[Tensor],
            box_delta: List[Tensor],
            image_size: Tuple[int, int]
    ):
        """
        Single-image inference. Return bounding-box detection results by
        thresholdingon scores and applying non-maximum suppression (NMS).

        Args:
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature
                level.
            box_cls (list[Tensor]): list of #feature levels. Each entry
                contains tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K
                becomes 4.
            image_size (tuple(H, W)): a tuple of the image height and width.
        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta,
                                                   anchors):
            # (HxWxAxK,)
            predicted_prob = box_cls_i.flatten().sigmoid_()

            # Apply two filtering below to make NMS faster.
            # 1. Keep boxes with confidence score higher than threshold
            keep_idxs = predicted_prob > self.test_score_thresh
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = nonzero_tuple(keep_idxs)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_topk_candidates, topk_idxs.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, idxs = predicted_prob.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[idxs[:num_topk]]

            anchor_idxs = topk_idxs // self.nums_classes
            classes_idxs = topk_idxs % self.nums_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(
                box_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all,
                           self.test_nms_thresh)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

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
