# F00d_recognition
Food recognition using streamlit with inception v3 backend
#### Deployed at [Streamlit](https://share.streamlit.io/sirreajohn/f00d_recognition/f00d_recog_inception.py)
Every one likes food! This deployment recognizes 11 different classes of food using a SOTA Inception V3 Transfer Learning.\n

### Inception V3
- The paper for Inception can be found [here](https://arxiv.org/abs/1512.00567v3)

- The paper implementation using pytorch can be found [here](https://github.com/pytorch/vision/blob/6db1569c89094cf23f3bc41f79275c45e9fcb3f3/torchvision/models/inception.py#L64)

- Inception-v3 is a convolutional neural network architecture from the Inception family that makes several improvements including using 
  - Label Smoothing,
  - Factorized 7 x 7 convolutions,\n 
  and the use of an auxiliary classifer to propagate label information lower down the network (along with the use of batch normalization for layers in the sidehead).
- Training on 16,600 images yielded 90% accuracy on train and 76% accuracy on validation. over 50 epochs!
- This model is saved and used later

### Model Architecture

![architecture](https://github.com/sirreajohn/F00d_recognition/blob/master/inception_2.png)
### Dataset Details and Classes
Data consists of 1.1GB of 16,600 images of different categories of food.
the categories of food that can be classified are 

    - Bread
    - Dairy Product
    - Dessert
    - Egg
    - Fried Food
    - Meat
    - Noodles-pasta
    - Rice
    - Seafood
    - Soup
    - Vegetable-fruit
    
Dataset is obtained from [kaggle](https://www.kaggle.com/trolukovich/food11-image-dataset)
