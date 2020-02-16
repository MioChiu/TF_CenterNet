# TF_CenterNet
## Introduction
This my implementation of CenterNet([Objects as Points](https://arxiv.org/abs/1904.07850)) in pure TensorFlow.The main special features of this repo inlcude:
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
1. Make dataset file  
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

2. Modify `cfg.py` and run `train.py`  
```
$ python train.py
```

3. Inference  
Update `ckpt_path` in `inference.py`,and run demo:  
```
$ python inference.py
```
The result for the example images should look like:

4.Visualization
```
$ tensorboard --logdir=./log
```
