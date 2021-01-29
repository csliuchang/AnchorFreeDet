from detectron2.data import MetadataCatalog
import os
from data.coco_class import register_coco_class


def register_all_coco_class(root):
    SPLITS = [
        ("balloon_train",
         "datasets/balloon/annotations/instances_train.json",
         "datasets/balloon/train/",
         ["balloon",]),
        ("balloon_val",
         "datasets/balloon/annotations/instances_val.json",
         "datasets/balloon/val/",
         ["balloon",]),
        ("coco_car_train",
         "datasets/coco/annotations/instances_train2017.json",
         "datasets/coco/train2017/",
         ["car",]),
        ("coco_car_val",
         "datasets/coco/annotations/instances_val2017.json",
         "datasets/coco/val2017/",
         ["", "car", ]),
        ("coco_person_car_val",
         "datasets/coco/annotations/instances_val2017.json",
         "datasets/coco/val2017/",
         ["person", "car"]),
    ]
    for name, json_file, image_root, class_names in SPLITS:
        register_coco_class(name, json_file, image_root, class_names)
        MetadataCatalog.get(name).evaluator_type = "coco_class"

