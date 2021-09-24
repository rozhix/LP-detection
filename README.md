## License-plate-detection
This project using yolov3 to detection car license plate

## requirements
Python 3.8 or later with the following pip3 install -U -r requirements.txt packages:

Numpy == 1.21.2

Opencv-python == 4.5.3.56

## Step 1: Prepare the dataset
I downloaded my car license plate dataset which contains 433 .png and .xml files, from Kaggle. It can be found and download from here

https://www.kaggle.com/andrewmvd/car-plate-detection#

I augmented dataset and my dataset turned into 690 images.
You can augment your dataset here

https://app.roboflow.com/


## Step 2 : train the model on Google Colab

1. upload our dataset on google drive

2. mount the Google Drive to Google Colab

```
from google.colab import drive
drive.mount('/content/drive')
```

3. clone the darknet git repository
```
! git clone https://github.com/AlexeyAB/darknet
```

4. Open darknet/Makefile and put 1 in front of GPU, CUDNN, and OPENCV instead of 0. These changes are required if you wanted to use GPU on Google Colab.
```
%cd /content/darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
```

5. Install the base darknet framework with the below commands.
```
%cd darknet 
! make
```


6. Make a copy of yolov3.cfg
```
!cp cfg/yolov3.cfg cfg/yolov3_training.cfg
```


7. Change lines on yolov3.cfg file
```
!sed -i 's/batch=1/batch=64/' cfg/yolov3_training.cfg
!sed -i 's/subdivisions=1/subdivisions=16/' cfg/yolov3_training.cfg
!sed -i 's/max_batches = 500200/max_batches = 6000/' cfg/yolov3_training.cfg
!sed -i '610 s@classes=80@classes=2@' cfg/yolov3_training.cfg
!sed -i '696 s@classes=80@classes=2@' cfg/yolov3_training.cfg
!sed -i '783 s@classes=80@classes=2@' cfg/yolov3_training.cfg
!sed -i '603 s@filters=255@filters=21@' cfg/yolov3_training.cfg
!sed -i '689 s@filters=255@filters=21@' cfg/yolov3_training.cfg
!sed -i '776 s@filters=255@filters=21@' cfg/yolov3_training.cfg
```


8. Create .names and .data file


9. Upload dataset in Colab


10. Download pre-trained weights for the convolutional layers file

```
!wget https://pjreddie.com/media/files/darknet53.conv.74
```


11. start training
```
!./darknet detector train data/obj.data cfg/yolov3_training.cfg darknet53.conv.74 -dont_show
```


## Step 3: prediction

 I couldn’t push my final weights file  (yolov3_training_final.weights 234MB)
 
you can download it from here

https://colab.research.google.com/drive/1mAbzVafP1BYH8U2OsFbKdCTUBMqYKrQe?usp=sharing

Download my LP-detection repository and .weights file
Test on a single image:
```
Python  yolo.py --image test.jpg
```
