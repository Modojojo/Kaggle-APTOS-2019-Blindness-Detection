## Configurations for Different Models used in ensembling
#### No. of models used in ensembling - 5 <br/>Each of the five models were trained on different folds of training data and validated on different validation sets.
#### The following configurations were same for all the models:
1. Image size - 300x300
2. Multilabel target
3. Augmentations - 
    * Rotation (25 - 360 degree on different models)
    * vertical flip
    * horizontal flip
    * Zoom range - (0.25 - 0.35)
4. optimizer - Adam
5. Loss function - Binary Crossentropy
6. output activation - Sigmoid
7. callback - Early Stopping (patience = 5)

### 1. MODEL 1 
* Training on old competition data
  * Training on a total of about 35000 images divided into 9 batches, each batch trained for 10 epochs.
  * Total no. of epochs : 92 (divided into 9 batches of 10 epochs for training data), (also trained 2 epochs on the validation data).
  * Trained on validation data as well (only for 2 epochs)
  * augmentations - rotation_range = 25, zoom_range = 0.25, vertial and horizontal flip  
  * preprocessing - only cropping
  * train-validation-split - 90% training and 10% validation
  * batch size : 32
* Training on current competition data 
  * Trained the above model on about 3600 images for 10 epochs
  * preprocessing - only cropping
  * No validation on new data
* Added TTA on test data
#### Model 1 Performance : 
* Training Accuracy around 95%
* Validation accuracy around 94%
* Leaderboard score - Public : 0.797  Private : 0.910

### 2. MODEL 2
* Training on old competition data : 
   * Total no. of EPOCHS : 75 (8 epochs for 9 batches and 3 epochs while training on the validation set)
   * preprocessing - Cropping
   * augmentations - rotation_range = 25, zoom_range = 0.35, vertial and horizontal flip 
   * batch size = 32
   * Train-validation-split - 90% training and 10% validation
* Training on current competition data
   * Total no. of epochs : 12
   * Learning Rate : 1e-4 with ReduceLrOnPlateau(patience = 1, factor = 0.5, min_lr = 1e-6)
   * 90% training data and 10% validation data
   * preprocessing - Cropping (crop_from_gray)
* Added TTA on test data
#### Model 2 Performance
* Training Accuracy around 94%
* validation accuracy around 94.5%
* Leaderboard score - Public : 0.790   Private : 0.912

### 3. MODEL 3
* Training on old competition data : 
   * Total no. of EPOCHS : 72 (8 epochs for 9 batches)
   * augmentations - rotation_range = 360, zoom_range = 0.25, vertial and horizontal flip 
   * learning rate = 1e-4 with ReduceLrOPlateau(patience = 1, factor = 0.5, min_lr = 5e-5)
   * EarlyStopping(patience = 5)
   * preprocessing - Cropping
   * batch size = 32
   * Train-validation-split - 90% training and 10% validation
* Training on current competition data
   * Total no. of epochs : 8
   * Learning Rate : 5e-5
   * 90% training data and 10% validation data
   * preprocessing - Cropping (crop_from_gray)
* Added TTA on test data
#### Model 3 Performance
* Training Accuracy around 95%
* validation accuracy around 96%
* Leaderboard score - Public : 0.801   Private : 0.917

### 4. MODEL 4
* Training on old competition data : 
   * Total no. of EPOCHS : 77 (8 epochs for 9 batches and 5 epochs while training on the validation set)
   * augmentations - rotation_range = 360, zoom_range = 0.35, vertial and horizontal flip 
   * No preprocessing
   * batch size = 32
   * Train-validation-split - 90% training and 10% validation
* Training on current competition data
   * Total no. of epochs : 10
   * Learning Rate : 5e-5
   * 90% training data and 10% validation data
   * No preprocessing
#### Model 4 Performance
* Training Accuracy around 94%
* validation accuracy around 93%
* Leaderboard score - Public : 0.797   Private : 0.914

### 5. MODEL 5
* Training on old competition data : 
   * Total no. of EPOCHS : 100
   * preprocessing - Cropping
   * Learning Rate : 5e-5 
   * batch size = 32
   * no Validation
* Training on current competition data
   * Total no. of epochs : 10
   * Learning Rate : 5e-5 
   * 90% training data and 10% validation data
   * No preprocessing
#### Model 5 Performance
* Training Accuracy around 96%
* validation accuracy around 94%
* Leaderboard score - Public : 0.794   Private : 0.915

### Ensembling and TTA : 
* Each model gives predictions on 1. original image, 2. image rotated at 1-6 degrees, 3. image rotated at 7-12 degrees
* stacking the predictions in such a way that for each type of tta we get one list of 5 elements (1 prediction from each model)
   * explanation - m1_p1 : model 1 predictions 1 on original image <br/> m1_p2 : model 1 prediction 2 on rotated image 
* this gives 3 lists of predictions such as : [m1_p1, m2_p1, m3_p1, m4_p1, m5_p1]<br/> [m1_p2, m2_p2, m3_p2, m4_p2, m5_p2]<br/> [m1_p3, m2_p3, m3_p3, m4_p3, m5_p3]
* mode of each list is taken (this gives 3 values) and then mode of these 3 values is taken and this prediction is considered final.
