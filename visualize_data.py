import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances

"""
add your dataset path 
"""
register_coco_instances('self_balloon_train', {},
                        './datasets/balloon/annotations/instances_train.json',
                       './datasets/balloon/train')
register_coco_instances('self_balloon_val', {},
                        '/home/changliu/PycharmProjects/SparseR-CNN/output/inference/coco_instances_results.json',
                       './datasets/balloon/val')

coco_val_metadata = MetadataCatalog.get("self_balloon_train")
dataset_dicts = DatasetCatalog.get("self_balloon_train")


import random

for d in random.sample(dataset_dicts, 1):
    img_path = d["file_name"]
    img = cv2.imread(img_path, 1)
    visualizer = Visualizer(img[:, :, ::-1], metadata=coco_val_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow('rr', vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)
