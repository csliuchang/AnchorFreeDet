## This is a github for collecting anchor-free methods 


for honest, I am just begin this github, and i will compete it soon
#### what's coming soon：
1. augment pipeline  
2. other anchor-free methods such as  fcos
3. rotation methods DAL BBA

first, you need to install detectron2
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
#### [·] support sparse-rcnn
#### [·] support centernet
#### [·] support onenet
#### [·] support yolof

### train 
```
python3 train_net.py --num-gpus 1 --config-file [your config file] --models [your model]
```

### eval
```
python train_net.py --num-gpus 1 --config-file [your config file] --models [your model]/ 
                    --eval-only MODEL.WEIGHTS path/to/model.pth
```
### Use tensorboard 
```
cd [outputdir]
tensorboard --logdir=[outputdir]
```
#### SparseRCNN
##### balloon

| Backbone                 |    AP        |  AP50    |  FPS|
| ----------------         | ---------------- | -------------- | ----- |
| res50     +2.5e-5+ADAMW            | 42.3     |    53.2      | 12  |
| res50+augment +2.5e-5+ADAMW    |     50.0     |    58.2      |    12   |       
| res50 + augment+1e-4+ADAMW  |        56.0    |        68.9  |     12  |
| res50 + coco_pretrained| 80.3             |     89.1        |     12  | 
##### bead
| Backbone                 |    AP        |  AP50    |  FPS|
| ----------------         | ---------------- | -------------- | ----- |
| res50 + coco_pretrained| 19.3             |     57.7        |     10  | 

#### OneNet(balloon)
| Backbone                 |   AP     |  AP50    | FPS |
| ----------------         | ---------------- | -------------- | ----- |
| res50                  | 48.2            |      64.8          |   30     |
| res50 + augment        |                  |                 |        |
| res50 + coco_pretrained|                  |                 |       | 

#### centernet(balloon)

| Backbone                 |   AP    |  AP50    | FPS |
| ----------------         | ---------------- | -------------- | ----- |
| res50                  | 42.1            |      65.1          |        |
| res50 + augment        |                  |                 |         |


### Reference:  
+ [SparseRCNN](https://github.com/PeizeSun/SparseR-CNN)  
+ [detectron2](https://github.com/facebookresearch/detectron2)  
+ [AdelaiDet](https://github.com/aim-uofa/adet) some detection methods including FCOS, BlendMask
+ [CenterNet](https://github.com/JDAI-CV/centerX)  
+ [OneNet](https://github.com/PeizeSun/OneNet), keypoint detection 
+ [Res2Net backbones](https://github.com/Res2Net/Res2Net-detectron2)
+ [VoVNet backbones](https://github.com/youngwanLEE/vovnet-detectron2)  


