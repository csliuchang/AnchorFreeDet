## This is a github for collecting anchor-free methods 


for honest, I am just begin this github, and i will compete it soon
####whats coming soon：
1. augment pipeline  
2. anchor-free methods centernet fcos
3. rotation methods DAL BBA

first, you need to install detectron2
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
####[√] support sparse-rcnn
####[ ] support centernet
####[√] support onenet

### train 
```
python3 train_net.py --num-gpus 1 --config-file [your config file] --model [your model]
```

### eval
```
python train_net.py --num-gpus 1 --config-file [your config file] --model [your model]/ 
                    --eval-only MODEL.WEIGHTS path/to/model.pth
```
###Use tensorboard 
```
cd [outputdir]
tensorboard --logdir=[outputdir]
```
SparseRCNN

| Backbone                 |   AP balloon    |  AP IOU=0.5    | inference time |
| ----------------         | ---------------- | -------------- | ----- |
| res50                  | 42.3             |     53.2          |        |
| res50 + augment        |                  |                 |        |
| res50 + coco_pretrained| 80.3             |     89.1        |       | 

OneNet 

| Backbone                 |   AP balloon    |  AP IOU=0.5    | inference time |
| ----------------         | ---------------- | -------------- | ----- |
| res50                  | 48.2            |      64.8          |        |
| res50 + augment        |                  |                 |        |
| res50 + coco_pretrained|                  |                 |       | 

centernet

| Backbone                 |   AP balloon    |  AP IOU=0.5    | inference time |
| ----------------         | ---------------- | -------------- | ----- |
| res50                  | 42.1            |      65.1          |        |
| res50 + augment        |                  |                 |         |


Reference:  
https://github.com/PeizeSun/SparseR-CNN  
https://github.com/facebookresearch/detectron2  
https://github.com/aim-uofa/AdelaiDet  
https://github.com/JDAI-CV/centerX  
https://github.com/PeizeSun/OneNet  


