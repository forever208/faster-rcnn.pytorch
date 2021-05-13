Running the repo in Colab is recommended, copy the file [Faster R-CNN Colab](https://colab.research.google.com/drive/1zYICq7vZlaQ6LSAOpE20qerS0d0viK1-?usp=sharing), then run it on Colab. (remember to change the runtime type to GPU in Colab)


## 【1】Installation

clone the code
```
git clone https://github.com/forever208/faster-rcnn.pytorch.git
```

create a folder to save the pretrained model 
```
cd faster-rcnn.pytorch && mkdir data
cd data && mkdir pretrained_model
```

#### Compilation

Install all the python dependencies using pip:
```
cd ..
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:
```
cd lib
python setup.py build develop
```
It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop.

#### Install coco_python_API
```
cd data
git clone https://github.com/pdollar/coco.git 
cd coco/PythonAPI
make
cd ../../..
```


## 【2】Run demo (detect images)

#### Download pre-trained weights manually

let's say, you downloaded the model [Res-101](https://www.dropbox.com/s/4v3or0054kzl19q/faster_rcnn_1_7_10021.pth?dl=0) whose filename is `faster_rcnn_1_7_10021.pth` and you have put it into folder `./data/pretrained_model/`


#### Download pre-trained weights by command 
I have saved the file in my Gdrive, so you can download by:
```
cd data/pretrained_model/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1n-YUaO0O2aJhWwZ_7DVF-xFJXK5JaZbR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1n-YUaO0O2aJhWwZ_7DVF-xFJXK5JaZbR" -O faster_rcnn_1_7_10021.pth && rm -rf /tmp/cookies.txt
cd ../..
```

#### run demo
put your images into folder  `./images`, then run demo
```
python demo.py --net res101  --model_dir ./data/pretrained_model --model_weights faster_rcnn_1_7_10021.pth
```
you will find the detection results in folder `/images`



## 【3】Download Datasets

#### PASCAL_VOC_2007
download and unzip the VOC2007 dataset by the command:
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar

tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar

rm -rf VOCtrainval_06-Nov-2007.tar
rm -rf VOCtest_06-Nov-2007.tar
rm -rf VOCdevkit_08-Jun-2007.tar
```

remember to move the dataset to the folder `./data` and rename it as `VOCdevkit2007`.
you can do it by the command:
```
mv VOCdevkit/ ./data/VOCdevkit2007
```


#### COCO
Please also follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare the data.



## 【4】Pretrained Model

#### Download pre-trained weights by command
I have saved the resnet101 weigths in my Gdrive, so you can download the weights by command:
```
cd data/pretrained_model/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Pyv546ss5q4idvcXE_Z_GFn_qOhDz10H' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Pyv546ss5q4idvcXE_Z_GFn_qOhDz10H" -O resnet101_caffe.pth && rm -rf /tmp/cookies.txt
```

#### Download pre-trained weights manually 

We used two pretrained models in our experiments, VGG and ResNet101. You can download these two models from:

* VGG16: [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

* ResNet101: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

Download them and put them into the folder `./data/pretrained_model/`



## 【5】Train on VOC

#### Train Faster R-CNN in Colab
I recommend you to train the model in Colab for the environment convinience.

To train a faster R-CNN model with resnet101 on pascal_voc, simply run:
```
python trainval_net_Adam.py --dataset pascal_voc --net res101 --bs 8 --nw 2 --cuda
```

where 'bs' is the batch size; nw is the number_of_workers



## 【6】Test on VOC

first creat a folder for your weight file (e.g. download the weights file and move it into the folder)
```
mkdir -p ./models/res101/pascal_voc

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1n-YUaO0O2aJhWwZ_7DVF-xFJXK5JaZbR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1n-YUaO0O2aJhWwZ_7DVF-xFJXK5JaZbR" -O faster_rcnn_1_7_10021.pth && rm -rf /tmp/cookies.txt
mv faster_rcnn_1_7_10021.pth models/res101/pascal_voc
```

then run the command:
```
python test_net_Colab.py --dataset pascal_voc --net res101 --checksession 1 --checkepoch 7 --checkpoint 10021 --cuda
```
where SESSION=1, EPOCH=7, CHECKPOINT=10021.



## 【7】Benchmarking

We benchmark our code thoroughly on three datasets: pascal voc, coco and imagenet-200, using two different network architecture: vgg16 and resnet101. Below are the results:

1). PASCAL VOC 2007 (Train/Test: 07trainval/07test, scale=600, ROI Align)

model    | #GPUs | batch size | lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP
---------|--------|-----|--------|-----|-----|-------|--------|-----
[VGG-16](https://www.dropbox.com/s/6ief4w7qzka6083/faster_rcnn_1_6_10021.pth?dl=0)     | 1 | 1 | 1e-3 | 5   | 6   |  0.76 hr | 3265MB   | 70.1
[VGG-16](https://www.dropbox.com/s/cpj2nu35am0f9hp/faster_rcnn_1_9_2504.pth?dl=0)     | 1 | 4 | 4e-3 | 8   | 9  |  0.50 hr | 9083MB   | 69.6
[VGG-16](https://www.dropbox.com/s/1a31y7vicby0kvy/faster_rcnn_1_10_625.pth?dl=0)     | 8 | 16| 1e-2 | 8   | 10  |  0.19 hr | 5291MB   | 69.4
[VGG-16](https://www.dropbox.com/s/hkj7i6mbhw9tq4k/faster_rcnn_1_11_416.pth?dl=0)     | 8 | 24| 1e-2 | 10  | 11  |  0.16 hr | 11303MB  | 69.2
[Res-101](https://www.dropbox.com/s/4v3or0054kzl19q/faster_rcnn_1_7_10021.pth?dl=0)   | 1 | 1 | 1e-3 | 5   | 7   |  0.88 hr | 3200 MB  | 75.2
[Res-101](https://www.dropbox.com/s/8bhldrds3mf0yuj/faster_rcnn_1_10_2504.pth?dl=0)    | 1 | 4 | 4e-3 | 8   | 10  |  0.60 hr | 9700 MB  | 74.9
[Res-101](https://www.dropbox.com/s/5is50y01m1l9hbu/faster_rcnn_1_10_625.pth?dl=0)    | 8 | 16| 1e-2 | 8   | 10  |  0.23 hr | 8400 MB  | 75.2 
[Res-101](https://www.dropbox.com/s/cn8gneumg4gjo9i/faster_rcnn_1_12_416.pth?dl=0)    | 8 | 24| 1e-2 | 10  | 12  |  0.17 hr | 10327MB  | 75.1  


2). COCO (Train/Test: coco_train+coco_val-minival/minival, scale=800, max_size=1200, ROI Align)

model     | #GPUs | batch size |lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP
---------|--------|-----|--------|-----|-----|-------|--------|-----
VGG-16     | 8 | 16    |1e-2| 4   | 6  |  4.9 hr | 7192 MB  | 29.2
[Res-101](https://www.dropbox.com/s/5if6l7mqsi4rfk9/faster_rcnn_1_6_14657.pth?dl=0)    | 8 | 16    |1e-2| 4   | 6  |  6.0 hr    |10956 MB  | 36.2
[Res-101](https://www.dropbox.com/s/be0isevd22eikqb/faster_rcnn_1_10_14657.pth?dl=0)    | 8 | 16    |1e-2| 4   | 10  |  6.0 hr    |10956 MB  | 37.0

**NOTE**. Since the above models use scale=800, you need add "--ls" at the end of test command.

3). COCO (Train/Test: coco_train+coco_val-minival/minival, scale=600, max_size=1000, ROI Align)

model     | #GPUs | batch size |lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP
---------|--------|-----|--------|-----|-----|-------|--------|-----
[Res-101](https://www.dropbox.com/s/y171ze1sdw1o2ph/faster_rcnn_1_6_9771.pth?dl=0)    | 8 | 24    |1e-2| 4   | 6  |  5.4 hr    |10659 MB  | 33.9
[Res-101](https://www.dropbox.com/s/dpq6qv0efspelr3/faster_rcnn_1_10_9771.pth?dl=0)    | 8 | 24    |1e-2| 4   | 10  |  5.4 hr    |10659 MB  | 34.5

4). Visual Genome (Train/Test: vg_train/vg_test, scale=600, max_size=1000, ROI Align, category=2500)

model     | #GPUs | batch size |lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP
---------|--------|-----|--------|-----|-----|-------|--------|-----
[VGG-16](http://data.lip6.fr/cadene/faster-rcnn.pytorch/faster_rcnn_1_19_48611.pth)    | 1 P100 | 4    |1e-3| 5   | 20  |  3.7 hr    |12707 MB  | 4.4

Thanks to [Remi](https://github.com/Cadene) for providing the pretrained detection model on visual genome!

* Click the links in the above tables to download our pre-trained faster r-cnn models.
* If not mentioned, the GPU we used is NVIDIA Titan X Pascal (12GB).



## Authorship

This project is equally contributed by [Jianwei Yang](https://github.com/jwyang) and [Jiasen Lu](https://github.com/jiasenlu), and many others (thanks to them!).


## Citation

    @article{jjfaster2rcnn,
        Author = {Jianwei Yang and Jiasen Lu and Dhruv Batra and Devi Parikh},
        Title = {A Faster Pytorch Implementation of Faster R-CNN},
        Journal = {https://github.com/jwyang/faster-rcnn.pytorch},
        Year = {2017}
    }

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }
