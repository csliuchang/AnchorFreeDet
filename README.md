## This is a github for collecting anchor-free methods 


for honest, I am just begin this github, and i will compete it soon
### whats coming soon
1. augment papeline  
2. center fcos detr   
3. rotation methods: DAL BBA

first, you need to install detectron2
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

#### support sparse-rcnn
#### support centernet
#### support onenet

### train 
```
python3 train_net.py --config-file [your config file] --models [your model]
```

### eval
```
python train_net.py --num-gpus 1 --config-file [your config file] --model [your model]/ 
                    --eval-only MODEL.WEIGHTS path/to/model.pth
```

### Use tensorboard
```
cd [outputdir]
tensorboard --logdir=[outputdir]
```

Reference:  
https://github.com/PeizeSun/SparseR-CNN  
https://github.com/facebookresearch/detectron2  
https://github.com/aim-uofa/AdelaiDet  
https://github.com/JDAI-CV/centerX  
https://github.com/PeizeSun/OneNet  


