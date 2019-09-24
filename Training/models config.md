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

### MODEL 1 
* Training on old competition data
  * Training on a total of about 35000 images divided into 9 batches, each batch trained for 10 epochs.
  * Total no. of epochs : 92 (divided into 9 batches of 10 epochs for training data), (also trained 2 epochs on the validation data).
  * Trained on validation data as well (only for 2 epochs)
  * preprocessing - only cropping
  * train-validation-split - 90% training and 10% validation
  * batch size : 32
* Training on current competition data 
  * Trained the above model on about 3600 images for 10 epochs
  * preprocessing - only cropping
  * No validation on new data
#### Model 1 Performance : 
* Training Accuracy around 95%
* Validation accuracy around 94%
* Leaderboard score - Public : 0.797  Private : 0.910
