import numpy as np
import pandas as pd 
import os
import gc
import h5py
import keras
import cv2
from keras.optimizers import Adam
from tqdm import tqdm_notebook
from keras.preprocessing.image import ImageDataGenerator
from efficientnet import EfficientNetB3 # import efficientnet

# PATHS to directories
model_path = '../input/with-c-v-kernel-no-1-aptos-old-data-s300/cv_old_data_effnet.h5'
train_csv = "../input/aptos2019-blindness-detection/train.csv"
test_csv = "../input/aptos2019-blindness-detection/test.csv"
train_dir = "../input/aptos2019-blindness-detection/train_images/"
test_dir = "../input/aptos2019-blindness-detection/test_images/"

size = 300,300 # input image size

df = pd.read_csv(train_csv) # CSV containing training images id and label

train_paths = [train_dir + str(x) + str(".png") for x in df["id_code"]] # list contining image paths

labels = pd.get_dummies(df["diagnosis"]).values

# creating multilabel target
y_train_multi = np.empty(labels.shape, dtype=labels.dtype)
y_train_multi[:, 4] = labels[:, 4]
for i in range(3, -1, -1):
    y_train_multi[:, i] = np.logical_or(labels[:, i], y_train_multi[:, i+1])
train_labels = y_train_multi

# load MODEL pretrained on the old competition data
model = keras.models.load_model(model_path)

#compile
optimizer=Adam(lr = 0.00005)
loss = "binary_crossentropy"
model.compile(loss = loss, optimizer = optimizer, metrics = ["accuracy"])

# image cropping function
def crop_image_from_gray(img,tol=7):
    """
    Crop out black borders
    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
    """  
    
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img>tol        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

# image loading+preprocessing function
def get_image(path):
    image = cv2.imread(path)
    image = crop_image_from_gray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    return image

# load images
images = np.empty((len(df),300,300,3), dtype = np.uint8)
for i, im in tqdm_notebook(enumerate(train_paths)):
    images[i,:,:,:] = get_image(im)

# image data generator
train_aug = ImageDataGenerator(
        zoom_range=0.25,
        rotation_range = 360,
        vertical_flip=True,
        horizontal_flip=True)

train_generator = train_aug.flow(images, train_labels, batch_size = 8)

# TRAINING
model.fit_generator(train_generator, epochs = 10, steps_per_epoch = len(train_generator))

# save model
model.save('crop_v1_new_old_effnet.h5')