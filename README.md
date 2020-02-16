# TF_CenterNet
## Introduction
This my implementation of CenterNet([Objects as Points](https://arxiv.org/abs/1904.07850)) in pure TensorFlow.You can refer to the official [code](https://github.com/xingyizhou/CenterNet).Because I use light backbone without DCN, the performance of this implementation is worse than official version. The main special features of this repo inlcude:
* tf.data pipeline
* light backbone：resnet18, mobilenetV2
* all codes were writen by pure TensorFlow ops (no keras or slim) 
* support training on your own dataset.

## Requirements
* python3
* tensorflow>=1.12
* opencv-python
* tqdm

## Train on voc dataset

### 1. Make dataset file  
Download Pascal VOC Dataset and reorganize the directory as follows:
```
VOC
├── test
|    └──VOCdevkit
|        └──VOC2007
└── train
     └──VOCdevkit
         └──VOC2007
         └──VOC2012
```
Generate `./data/dataset/voc_train.txt` and `./data/dataset/voc_test.txt`, some codes of this part are from [yolov3](https://github.com/YunYang1994/tensorflow-yolov3/blob/master/scripts/voc_annotation.py):  
```
$ cd ./data/dataset
$ python voc_annotation.py --data_path D:/dataset/VOC
```

### 2. Download pre-train weights  
You can get pre-train weights of [resnet](https://github.com/MioChiu/ResNet_TensorFlow) or [mobilenet](https://github.com/MioChiu/MobileNet_V2_TensorFlow) from my other repo.  Put npy file in `pretrained_weights` folder. 

### 3. Modify `cfg.py` and run `train.py`  
```
$ python train.py
```

### 4. Inference  
Update `ckpt_path` in `inference.py`,and run demo:  
```
$ python inference.py
```
The result for the example images should look like:  
![demo_img1](https://github.com/MioChiu/TF_CenterNet/blob/master/img/1.png)  
![demo_img2](https://github.com/MioChiu/TF_CenterNet/blob/master/img/2.png)  
![demo_img3](https://github.com/MioChiu/TF_CenterNet/blob/master/img/3.png)  

### 5.Visualization
```
$ tensorboard --logdir=./log
```

## Reference
[1] official [code](https://github.com/xingyizhou/CenterNet) and [paper](https://arxiv.org/abs/1904.07850)  
[2] YunYang1994's [YOLOv3 repo](https://github.com/YunYang1994/tensorflow-yolov3)
