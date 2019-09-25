import numpy as np
import pandas as pd 
import os
import cv2
import keras
import h5py
import gc
from tqdm import tqdm_notebook
import random
from statistics import mode
from efficientnet import EfficientNetB3

# PATHS to all the pretrained models
model_path_1 = '../input/model-f-with-cv-k2-aptos-old-new-version-2/crop_v1_new_old_effnet.h5'
model_path_2 = '../input/crop-v1-aptos-old-data-new-data-train-model/crop_v1_new_old_effnet.h5'
model_path_3 = '../input/with-c-v-k1-aptos-old-new-model-ver1/crop_v1_new_old_effnet.h5'
model_path_4 = '../input/size300-v1-aptos-old-new-model-v1/crop_v1_new_old_effnet.h5'
model_path_5 = '../input/size300-crop-v2-aptos-old-new-model/crop_v1_new_old_effnet.h5'
test_csv = "../input/aptos2019-blindness-detection/test.csv"
test_dir = "../input/aptos2019-blindness-detection/test_images/"

df = pd.read_csv(test_csv)

# crop images
def crop_image_from_gray(img,tol=7):  
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

# load images
def get_image_crop(path):
    image = cv2.imread(path)
    image = crop_image_from_gray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (300,300))
    return image

# make predictions on one image
def get_predictions(test, model):
    predictions = model.predict(test) > 0.5
    predictions = predictions.astype(int).sum(axis=1) - 1
    return predictions

# rotate image by a degree
def rotate_image(image, degree):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width/2,height/2), degree, 1)
    image_transformed = cv2.warpAffine(image, rotation_matrix, (width, height))
    return image_transformed


# make predictions on and image for a particular model and perform TTA by rotating through certain degree
def get_predictions_on_model(model, image):
    predictions_1 = get_predictions(image, model)
    
    
    temp_image = rotate_image(image[0], random.randint(1,5))
    predictions_2 = get_predictions(temp_image.reshape(1,300,300,3), model)
    
    temp_image = rotate_image(image[0], random.randint(6,10))
    predictions_3 = get_predictions(temp_image.reshape(1,300,300,3), model)
    
    del temp_image

    return predictions_1[0], predictions_2[0], predictions_3[0]

# returns mode of a list of multiple numbers (1st mode)
def return_mode(modelist):
    countlist = [0,0,0,0,0]
    for x in modelist:
        countlist[x]+=1
    ans = np.argmax(countlist)
    equallist = [0,0,0,0,0]
    for i in range(len(countlist)):
        if i==ans:
            pass
        else:
            if countlist[i] == countlist[ans]:
                equallist[i] = 1
    best = None
    for i in range(len(equallist)):
        if equallist[i] == 1:
            best = i
    if best == None:
        best = ans
    return best

# find mode of the predictions from all the models on all types of TTA
def find_mode(i):
    curr_mode = None
    if len(set(i)) == 3:
        curr_mode = i[0]
    else:
        curr_mode = mode(i)
    return curr_mode

# make predictions on one image for all the models and return the final predictions for that image
def predict_on_one_image(image):
    m1_predictions_1, m1_predictions_2, m1_predictions_3 = get_predictions_on_model(model1, image)
    m2_predictions_1, m2_predictions_2, m2_predictions_3 = get_predictions_on_model(model2, image)
    m3_predictions_1, m3_predictions_2, m3_predictions_3 = get_predictions_on_model(model3, image)
    m4_predictions_1, m4_predictions_2, m4_predictions_3 = get_predictions_on_model(model4, image)
    m5_predictions_1, m5_predictions_2, m5_predictions_3 = get_predictions_on_model(model5, image)
    
    mode1 = return_mode([m1_predictions_1, m2_predictions_1, m3_predictions_1, m4_predictions_1, m5_predictions_1])
    mode2 = return_mode([m1_predictions_2, m2_predictions_2, m3_predictions_2, m4_predictions_2, m5_predictions_2])
    mode3 = return_mode([m1_predictions_3, m2_predictions_3, m3_predictions_3, m4_predictions_3, m5_predictions_3])
    
    final_mode = find_mode([mode1, mode2, mode3])
    return final_mode

# load all the trained models 
model1 = keras.models.load_model(model_path_1)
model2 = keras.models.load_model(model_path_2)
model3 = keras.models.load_model(model_path_3)
model4 = keras.models.load_model(model_path_4)
model5 = keras.models.load_model(model_path_5)

test_paths = [test_dir + str(x) + str(".png") for x in df["id_code"]] # create paths for images

# prediction pipeline - load images and make predictions one image at a time
final_predictions = []
for path in tqdm_notebook(test_paths):
    image = get_image_crop(path)
    prediction = predict_on_one_image(image.reshape(1,300,300,3))
    final_predictions.append(prediction)

# submission
id_code = df["id_code"].values.tolist()
subfile = pd.DataFrame({"id_code":id_code, "diagnosis":final_predictions})

subfile.to_csv('submission.csv',index=False)