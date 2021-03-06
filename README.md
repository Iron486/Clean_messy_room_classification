
# Clean vs messy room classification with deep learning techniques

The aim of this project was the correct classification of messy rooms from clean rooms.

There were given the train and validation datasets, both containing separate images of messy and clean rooms and the test dataset without labels. 
I fetched the data from here https://www.kaggle.com/datasets/cdawn1/messy-vs-clean-room.


In this repository there are 5 notebooks: 

- [ANN_training.ipynb](https://github.com/Iron486/Clean_messy_room_classification/blob/main/ANN_training.ipynb) that I used to fit an **Artificial Neural Network** to the train dataset and predict on the validation dataset.
- [CNN_training.ipynb](https://github.com/Iron486/Clean_messy_room_classification/blob/main/CNN_training.ipynb) in which I trained a **Convolutional Neural Network** and I predicted the model both on validation and test datasets.
- [CNN_augmented_dataset.ipynb](https://github.com/Iron486/Clean_messy_room_classification/blob/main/CNN_augmented_dataset.ipynb) in which I fit a **Convolutional Neural Network with an augmented train dataset** and I predicted the model both on validation and test datasets.
- [CNN_augmented_dataset_with_dropout.ipynb](https://github.com/Iron486/Clean_messy_room_classification/blob/main/CNN_augmented_dataset_with_dropout.ipynb) similar to the last one, but I **modified** some **parameters and hyperparameters**.
- [Bonus_exercise.ipynb](https://github.com/Iron486/Clean_messy_room_classification/blob/main/Bonus_exercise.ipynb) that is an exercise that I did only for curiosity. I calculated the average number of red,blue and green component for each pixel within the images on the train dataset. 
In addition, I calculated the standard deviation of the value of each pixel, considering all the images of the train dataset.

I tried some basic machine learning algorithms, too (such as RandomForestClassifier, SVMa and logistic regression). However, I obtained very bad results on validation dataset (less than 80% accuracy and 60% recall) with a large overfitting.


Below, I reported the training curves represented for the ANN, CNN and CNN with augmented dataset, along with a brief description of the used methods.
The first deep learning algorithm that I used was a simple **Artificial Neural Network**. 

Here, the obtained training curve can be observed:

**<p align="center"> ANN - training </p>**

<p align="center"> <img src="https://user-images.githubusercontent.com/62444785/162538666-fb66e587-a08f-452f-bd32-ff55bba4c12f.png" width="570" height="320"/>   </p>

The input data that the algorithm adopts were obtained using the `os` library and reading all the images for each folder. There were 5 folders, one for the test dataset and the others that contain images from test and train dataset of messy and clean images.
I preprocessed the data so that it was possible to train the ANN; then I scaled the data and I performed PCA to reduce the dimension of the dataset.

Next, I fit the model on train dataset using the following parameters and hyperparameters:

<p align="center">
 
| Layer (type)                | Output Shape             | Param # |  
|-----------------------------|--------------------------|---------| 
| flatten (Flatten)           | (None, 182)              | 0       |  
| batch_normalization (BatchNo| (None, 182)              | 728     |  
| dropout (Dropout)           | (None, 182)              | 0       |  
| dense (Dense)               | (None, 6000)             | 1098000 |  
| dropout_1 (Dropout)         | (None, 6000)             | 0       |  
| dense_1 (Dense)             | (None, 1000)             | 6001000 |  
| batch_normalization_1       | (None, 1000)             | 4000    |  
| dropout_2 (Dropout)         | (None, 1000)             | 0       |  
| dense_2 (Dense)             | (None, 32)               | 32032   |  
| dense_3 (Dense)             | (None, 1)                | 33      |  

- Total params: 7,135,793
- Trainable params: 7,133,429
- Non-trainable params: 2,364
- Optimizer = {'name': 'Adam',
 'learning_rate': 4e-07,
 'decay': 0.0,
 'beta_1': 0.8,
 'beta_2': 0.999,
 'epsilon': 1e-07,
 'amsgrad': False}
</p>

Note that Adam optimer was used not only for this model, but for all the training models. Below is the formula of the Adam Optimer written in LaTeX:

&nbsp;

<p align="center"> <img src="https://user-images.githubusercontent.com/62444785/170128532-dc14d2aa-f105-489c-8990-e4418baf1650.png" width="270" height="205"/>  </p>

&nbsp;

The parameters that appear both in the formula and in the Keras interface are `alpha`, called the learning rate; `beta_1` and `beta_2` that are initial decay rates used to estimate the first and second moments of the gradient; and `epsilon` that is a small number used to avoid that the second term in the right-hand side of the last equation cancels out.

`Decay` and `amsgrad` don't belong to the Adam optimizer, and they are respectively responsible for the modification of the decay of the gradient through the epochs and
the improvement of the convergence of the algorithm, trying to prevent large updates.

For further information, here is the link to the paper of the optimizer: https://arxiv.org/abs/1412.6980.

Afterwards, I tried to build a **Convolutional Neural Network** and I obtained better results:

**<p align="center"> CNN - training </p>**

<p align="center"> <img src="https://user-images.githubusercontent.com/62444785/162538984-6aeacc8a-5b42-4e15-b2cd-2dcd48d0a193.png" width="570" height="320"/>  </p>

Similarly to the ANN, I preprocessed the data in order to fit the first convolutional layer, and then I scaled the pixel values.

This time, I didn't perform PCA but I put max pooling layers after each convolutional layer. I placed a flatten layer as well, thus reducing the dimension of the image, and other 2 dense hidden layers. 

I used 120x120 pixel images and not 80x80 like in the ANN.

 <p align="center">
  
| Layer (type)                | Output Shape             |  Param #|   
|-----------------------------|--------------------------|---------|
| conv2d_43 (Conv2D)          | (None, 118, 118, 16)     | 448     |  
| max_pooling2d_43(MaxPooling)| (None, 59, 59, 16)       | 0       |  
| conv2d_44 (Conv2D)          | (None, 57, 57, 32)       | 4640    |  
| activation_11 (Activation)  | (None, 57, 57, 32)       | 0       |  
| max_pooling2d_44(MaxPooling)|(None, 28, 28, 32)        | 0       |  
| conv2d_45 (Conv2D)          | (None, 26, 26, 32)       | 9248    |  
| max_pooling2d_45(MaxPooling)| (None, 13, 13, 32)       | 0       |  
| conv2d_46 (Conv2D)          | (None, 11, 11, 64)       | 18496   |  
| max_pooling2d_46(MaxPooling)| (None, 5, 5, 64)         | 0       |  
| flatten_11 (Flatten)        | (None, 1600)             | 0       |  
| dense_33 (Dense)            | (None, 1722)             | 2756922 |  
| dense_34 (Dense)            | (None, 48)               | 82704   |  
| dense_35 (Dense)            | (None, 1)                | 49      |  


- Total params: 2,872,507
- Trainable params: 2,872,507
- Non-trainable params: 0
- Optimizer = {'name': 'Adam',
 'learning_rate': 9e-06,
 'decay': 0.0,
 'beta_1': 0.9,
 'beta_2': 0.999,
 'epsilon': 1e-07,
 'amsgrad': False}
</p>

I finally tested the **CNN with augmented dataset**, obtaining the following training curves:

**<p align="center"> CNN with augmented dataset - training </p>**


<p align="center"> <img src="https://user-images.githubusercontent.com/62444785/162535217-ebe6df02-97e2-4239-8f22-508788015d1b.png" width="570" height="320"/>  </p>



The input data were automatically fetched by Tensorflow through the function `ImageDataGenerator` in the `tensorflow.keras.preprocessing.image` module, giving as an input the training and validation datasets. 

The hyperparameters elected to augment the data were the following: 
- rescale=1/255,
- rotation_range=3,
- width_shift_range=0.1,
- height_shift_range=0.1,
- shear_range=0.1,
- zoom_range=0.1,
- horizontal_flip=True,
- fill_mode='nearest'
The model required way more time compared to the others, because the dimension of the image was bigger (150x150), the layers had more parameters than the previous CNN, and the augmentation slowed down the training time.

Below are the hyperameters and parameters that I used to train the model.

| Layer (type)                 | Output Shape             |  Param # |  
|------------------------------|--------------------------|----------|
| conv2d (Conv2D)              | (None, 148, 148, 16)     | 448      | 
| max_pooling2d (MaxPooling2D) | (None, 74, 74, 16)       | 0        | 
| conv2d_1 (Conv2D)            | (None, 72, 72, 32)       | 4640     | 
| activation (Activation)      | (None, 72, 72, 32)       | 0        |            
| max_pooling2d_1(MaxPooling2) | (None, 36, 36, 32)       | 0        | 
| conv2d_2 (Conv2D)            | (None, 34, 34, 64)       | 18496    | 
| max_pooling2d_2(MaxPooling2) | (None, 17, 17, 64)       | 0        | 
| conv2d_3 (Conv2D)            | (None, 15, 15, 64)       | 36928    |
| max_pooling2d_3(MaxPooling2) | (None, 7, 7, 64)         | 0        | 
| flatten (Flatten)            | (None, 3136)             | 0        | 
| dense (Dense)                | (None, 1522)             | 4774514  | 
| dense_1 (Dense)              | (None, 45)               | 68535    | 
| dense_2 (Dense)              | (None, 1)                | 46       | 

- Total params: 4,903,607
- Trainable params: 4,903,607
- Non-trainable params: 0
- Optimizer = {'name': 'Adam',
 'learning_rate': 4.3e-05,
 'decay': 0.0,
 'beta_1': 0.9,
 'beta_2': 0.999,
 'epsilon': 1e-07,
 'amsgrad': False}



It can be clearly noticed that **the CNN with augmented data** gives us the **best results**, with a **validation loss below 0.4** and **accuracy on validation dataset** that varies **between 0.85 and 0.95**.

On the other hand, **in the simple CNN and ANN** we have **worse results** with a **validation loss above 0.4** and **validation accuracy below 0.85**. Moreover, **overfitting** can be noticed especially in the training curve of the ANN.

The last model was a bit more unstable compared to the other two, despite using a small learning rate and a batch size of 40 images. 

**To improve stability**, I tried to put some **dropout layers** on the top neural network as well, obtaining these training curves: 

**<p align="center"> CNN with augmented dataset and dropouts - training </p>**

<p align="center"> <img src="https://user-images.githubusercontent.com/62444785/162551147-ed2b0fd6-9355-43c0-96b3-5648d97e9ca5.png" width="570" height="320"/>  </p>

I considered a dropout of 0.15 on the first hidden layer of the fully connected neural network, while 0.1 at the successive layer. 
I changed the size of the images to 180x180 pixels, and I adjusted the learning rate of the previous model without dropout from 0.000043 to 0.000004. 

I slightly increased the parameter `patience` of the model, without considering a huge number, since it would take more time to train the model considering a very high value.

Furthermore, I added an additional convolutional and max pool layers, and I decreased the number of neurons in the last two hidden layers.

The training time was even longer than the previous, taking few hours to obtain the described result.

Below is an overview of the model:



| Layer (type)                 | Output Shape             | Param # |  
|------------------------------|--------------------------|---------|
| conv2d (Conv2D)              | (None, 178, 178, 16)     | 448     |  
| max_pooling2d (MaxPooling2D) | (None, 89, 89, 16)       | 0       |  
| conv2d_1 (Conv2D)            | (None, 87, 87, 32)       | 4640    |  
| activation (Activation)      | (None, 87, 87, 32)       | 0       |  
| max_pooling2d_1 (MaxPooling2 | (None, 43, 43, 32)       | 0       |  
| conv2d_2 (Conv2D)            | (None, 41, 41, 64)       | 18496   |  
| max_pooling2d_2 (MaxPooling2 | (None, 20, 20, 64)       | 0       |  
| conv2d_3 (Conv2D)            | (None, 18, 18, 64)       | 36928   |  
| max_pooling2d_3 (MaxPooling2 | (None, 9, 9, 64)         | 0       |  
| conv2d_4 (Conv2D)            | (None, 7, 7, 64)         | 36928   |  
| max_pooling2d_4 (MaxPooling2 | (None, 3, 3, 64)         | 0       |  
| flatten (Flatten)            | (None, 576)              | 0       |  
| dense (Dense)                | (None, 1200)             | 692400  |  
| dropout (Dropout)            | (None, 1200)             | 0       |  
| dense_1 (Dense)              | (None, 38)               | 45638   |  
| dropout_1 (Dropout)          | (None, 38)               | 0       |  
| dense_2 (Dense)              | (None, 1)                | 39      |  

- Total params: 835,517
- Trainable params: 835,517
- Non-trainable params: 0
- Optimizer =  {'name': 'Adam',
 'learning_rate': 4e-06,
 'decay': 0.0,
 'beta_1': 0.9,
 'beta_2': 0.999,
 'epsilon': 1e-07,
 'amsgrad': False}

I finally evaluated on test dataset the three different CNNs. **The first two CNNs both classified correctly 8/10 of the dataset**. 
The **CNN with augmented dataset and dropout layers**, instead, **reached 90% of accuracy on test dataset**.
 
