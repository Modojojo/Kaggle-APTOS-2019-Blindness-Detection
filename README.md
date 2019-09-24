## About the Repository
The repository contains the description of the basic methodology used to build a model that could correctly predict the severity of **Diabetic Retinopathy** on a scale from 0-4. Originally completed as a Kaggle competition.
### APTOS 2019 Blindness Detection - Kaggle
Competition link [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
### Competition Results - 108th/2943 | top 4% | Silver medal
### Overview 
- API : KERAS
- Model used - EfficientNet pretrained on Imagenet
- Data - Using the Competition data and the old competition data (2015 competition data)
- Primarily Trained on 2015 competition data
- Validated on 2019 Data and then trained on the same
- Used Test Time Augmentation 
- Ensembled 5 models (All EfficientNet B3)
- Multilabel Classification
- Image size - 300x300
- Image Preprocessing - Image cropping, to crop out part of image that contained no information (black borders).
- Augmentations - Rotation(360), Horizontal and Vertical Flips, Zooming
  
### Architecture and Approach
- Architecture - The final model was built using EfficientNet B3 architecture pretrained on the Imagenet Data. The model was Trained on
the data from the old competition as well as this competition. Final result was an ensemble of 5 models Trained using different validation splits and hyperparameters.
  - **Optimimzer** : Adam
  - **Output activation**  : Sigmoid
  - **Loss function** : Binary Crossentropy
  - **Learning Ratge** : 5e-5, One thing that i sticked to, throughout the competition was to train the model with a constant learning rate of 5e-5, also experimented different learning rates ranging from 1e-3 to 1e-6 but 5e-5 worked the best.
  - **Batch Size** : 32
  - **Keras callbacks used** : ReduceLrOnPlateau, EarlyStopping
  
- Approach - Followed the basic approach of training the model primarily on the 2015 competition data and then training again on the new competition data using different validation splits for each of the 5 models used for ensembling. Also the one thing that boosted my score was applying Test Time Augmentation (rotating the image randomly about 1-6 degrees and again rotating the original image to about 12 degrees) while making a prediction.
  - Making Prediction on an image - Created the function that takes a model and an image as an input, and returns 3 predictions - one for original image and 2 more predictions for randomly rotated images at different angles, this is done for all the five models. Now the main idea was to stact the input of the original, and randomly rotated images from all 5 models, this gave 3 lists in which two lists contained the predictions made on images rotated by the same range of angles and one list contained predictions made by all 5 models on the original image. Now Calculated the MODE of predictions for each of the 3 lists and again stack them up in a single list and again calculate MODE of that list.
### Ensemble
- 5x Efficientnet B3 + 3x TTA
### Multilabel Classification 
- Instead of predicting a single label we will change the target to be a multilabel problem i.e, if the target is a certain label, then it encompasses all the labels before it.
  - Explanation : if target label after one hot encoding is [0,0,1,0,0], then it will be [1,1,1,0,0] for the multilabel version. The reason this version converts all previous labels to 1 is that Diabetic retinopathy in this competition is calculated in 5 levels (0-4) in increasing severity for each level.
  - References - https://www.kaggle.com/xhlulu/aptos-2019-densenet-keras-starter
### Other experiments (not used in the final model)
- Tried using different architectures - ResNet50, DenseNet121
- Tried using different image sizes - 224x224, 256x256, 512x512 (300x300 gave better results) 
- Treating the problem as simple classification (multilabel classification gave better results)
- Converting to regression problem (didnt gave better results)
- Preprocessing using Circle Crop
- Experimenting with LR scheduler


### Published Kernels 
https://www.kaggle.com/modojj/densenet121-and-cropping-aptos-2019 (published during competition, simple baseline model) 

### kaggle profile 
https://www.kaggle.com/modojj
