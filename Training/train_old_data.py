import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import random
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import gc
from efficientnet import EfficientNetB3


size = 300,300 # Image size

resized_train = '../input/diabetic-retinopathy-resized/resized_train/resized_train/' # Training Data path
train_csv = pd.read_csv('../input/diabetic-retinopathy-resized/trainLabels.csv') # csv file containing image id and label

'''
CONFIG...
multilabel classification
Image size 			: 300x300x3
loss 				: binary crossentropy
output activation 	: sigmoid
optimizer 			: Adam (learning rate = 5e-5)
color mode 			: RGB
'''

# training and validation split function
def split_data(images, labels): 
    x_train = images[:-3512]
    y_train = labels[:-3512]
    xval = images[-3512:]
    yval = labels[-3512:]
    return x_train, xval, y_train, yval

train_paths = [resized_train + str(x) + str(".jpeg") for x in train_csv["image"]] # list contining paths to images 
labels = pd.get_dummies(train_csv["level"]).values # pandas dummies for labels 

# creating multilabel target
y_train_multi = np.empty(labels.shape, dtype=labels.dtype)
y_train_multi[:, 4] = labels[:, 4]
for i in range(3, -1, -1):
    y_train_multi[:, i] = np.logical_or(labels[:, i], y_train_multi[:, i+1])
train_labels = y_train_multi

train_paths, val_paths, train_labels, val_labels = split_data(train_paths, train_labels) # split data

# Shuffling data randomly
zipped = list(zip(train_paths, train_labels))
random.shuffle(zipped)
train_paths, train_labels = zip(*zipped)

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

# image loading and preprocessing function
def get_image(path):
    image = cv2.imread(path)
    image = crop_image_from_gray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    return image

# load validation data 
X_val = np.empty((len(val_paths), 300,300,3), dtype = np.uint8)
for i, path in tqdm_notebook(enumerate(val_paths)):
    X_val[i,:,:,:] = get_image(path)

# image data generator 
train_aug = ImageDataGenerator(
        zoom_range=0.25,
        rotation_range = 25,
        vertical_flip=True,
        horizontal_flip = True)

# Model 
base_model = EfficientNetB3(weights=None, include_top=False,input_shape=(300,300,3))
base_model.load_weights('../input/efficientnet-keras-weights-b0b5/efficientnet-b3_imagenet_1000_notop.h5')
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(5, activation='sigmoid'))

optimizer=Adam(lr = 0.00005)
loss = "binary_crossentropy"

# callbacks - early stopping and reduce learning rate on plateau
es = EarlyStopping(monitor='val_loss', mode='min', patience = 5, verbose = 1, restore_best_weights=False)
rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience = 2, factor = 0.5, min_lr=0.00001, verbose = 1)    
callback_list = [es, rlrop]

# TRAINING 

n_batches = 10
image_index = 0
for loop_batch in range(9):
    images = np.empty((gen_batch_size,300,300,3), dtype = np.uint8)
    curr_labels = train_labels[image_index : image_index+gen_batch_size]
    for i,img_path in tqdm_notebook(enumerate(train_paths[image_index : image_index+gen_batch_size])):
        images[i,:,:,:] = get_image(img_path)
        
    #training
    model.compile(loss = loss, optimizer = optimizer, metrics = ["accuracy"])
    train_generator = train_aug.flow(images, curr_labels, batch_size = 32)
    model.fit_generator(generator = train_generator,
                        epochs = 10,
                        steps_per_epoch = len(train_generator),
                        validation_data = (X_val, val_labels),
                        callbacks = callback_list)
    
    del train_generator
    del images
    gc.collect()
    image_index = image_index + gen_batch_size

# train a few epochs on validation data as well
model.compile(loss = loss, optimizer = optimizer, metrics = ["accuracy"])
train_val_generator = train_aug.flow(X_val, val_labels, batch_size = 32)
model.fit_generator(train_val_generator,
                    epochs = 2,
                    steps_per_epoch = len(train_val_generator))

# saving model
model.save("cv_old_data_effnet.h5")