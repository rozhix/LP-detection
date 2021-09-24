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

1.upload our dataset on google drive

2.mount the Google Drive to Google Colab

```
from google.colab import drive
drive.mount('/content/drive')
```

3.clone the darknet git repository
```
! git clone https://github.com/AlexeyAB/darknet
```
