
import torch
from fvcore.nn import sigmoid_focal_loss_jit, giou_loss
from torch import Tensor, nn
import torch.distributed as dist
from torchvision.ops.boxes import box_iou

from detectron2.layers import batched_nms, cat, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils import comm

from .utils import YOLOFBox2BoxTransform


class Losses(nn.Module):
    def __init__(self, cfg):
        super(Losses, self).__init__()
        self.num_classes = cfg.MODEL.YOLOF.DECODER.NUM_CLASSES
        self.box2box_transform = YOLOFBox2BoxTransform(
                weights=cfg.MODEL.YOLOF.BOX_TRANSFORM.BBOX_REG_WEIGHTS,
                add_ctr_clamp=cfg.MODEL.YOLOF.BOX_TRANSFORM.ADD_CTR_CLAMP,
                ctr_clamp=cfg.MODEL.YOLOF.BOX_TRANSFORM.CTR_CLAMP
            )
        self.neg_ignore_thresh = cfg.MODEL.YOLOF.NEG_IGNORE_THRESHOLD
        self.pos_ignore_thresh = cfg.MODEL.YOLOF.POS_IGNORE_THRESHOLD
        self.focal_loss_alpha = cfg.MODEL.YOLOF.LOSSES.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.YOLOF.LOSSES.FOCAL_LOSS_GAMMA


    def forward(self,
               indices,
               gt_instances,
               anchors,
               pred_class_logits,
               pred_anchor_deltas):
        pred_class_logits = cat(
            pred_class_logits, dim=1).view(-1, self.num_classes)
        pred_anchor_deltas = cat(pred_anchor_deltas, dim=1).view(-1, 4)

        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
        N = len(anchors)
        # list[Tensor(R, 4)], one for each image
        all_anchors = Boxes.cat(anchors).tensor
        # Boxes(Tensor(N*R, 4))
        predicted_boxes = self.box2box_transform.apply_deltas(
            pred_anchor_deltas, all_anchors)
        predicted_boxes = predicted_boxes.reshape(N, -1, 4)

        ious = []
        pos_ious = []
        for i in range(N):
            src_idx, tgt_idx = indices[i]
            iou = box_iou(predicted_boxes[i, ...],
                          gt_instances[i].gt_boxes.tensor)
            if iou.numel() == 0:
                max_iou = iou.new_full((iou.size(0),), 0)
            else:
                max_iou = iou.max(dim=1)[0]
            a_iou = box_iou(anchors[i].tensor,
                            gt_instances[i].gt_boxes.tensor)
            if a_iou.numel() == 0:
                pos_iou = a_iou.new_full((0,), 0)
            else:
                pos_iou = a_iou[src_idx, tgt_idx]
            ious.append(max_iou)
            pos_ious.append(pos_iou)
        ious = torch.cat(ious)
        ignore_idx = ious > self.neg_ignore_thresh
        pos_ious = torch.cat(pos_ious)
        pos_ignore_idx = pos_ious < self.pos_ignore_thresh

        src_idx = torch.cat(
            [src + idx * anchors[0].tensor.shape[0] for idx, (src, _) in
             enumerate(indices)])
        gt_classes = torch.full(pred_class_logits.shape[:1],
                                self.num_classes,
                                dtype=torch.int64,
                                device=pred_class_logits.device)
        gt_classes[ignore_idx] = -1
        target_classes_o = torch.cat(
            [t.gt_classes[J] for t, (_, J) in zip(gt_instances, indices)])
        target_classes_o[pos_ignore_idx] = -1
        gt_classes[src_idx] = target_classes_o

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        if comm.get_world_size() > 1:
            dist.all_reduce(num_foreground)
        num_foreground = num_foreground * 1.0 / comm.get_world_size()

        # cls loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )
        # reg loss
        target_boxes = torch.cat(
        [t.gt_boxes.tensor[i] for t, (_, i) in zip(gt_instances, indices)],
        dim=0)
        target_boxes = target_boxes[~pos_ignore_idx]
        matched_predicted_boxes = predicted_boxes.reshape(-1, 4)[
            src_idx[~pos_ignore_idx]]
        loss_box_reg = giou_loss(
            matched_predicted_boxes, target_boxes, reduction="sum")

        return {
            "loss_cls": loss_cls / max(1, num_foreground),
            "loss_box_reg": loss_box_reg / max(1, num_foreground),
        }