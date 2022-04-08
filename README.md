
# Clean vs messy room classification with deep learning techniques

The aim of this problem was the correct classification of messy rooms from clean rooms.
I tried some machine learning algorithms (such as RandomForestClassifier, SVMa and logistic regression), but obtaining very bad results on validation dataset (less than 80 per cent accuracy and 60 per cent recall) and overfitting.

Hence, in this repository there are 4 notebooks: 
- [ANN_training.ipynb](https://github.com/Iron486/Clean_messy_room_classification/blob/main/ANN_training.ipynb) in which I fit an Artificial Neural Network to the train dataset and predict on the validation dataset.
- [CNN_training.ipynb](https://github.com/Iron486/Clean_messy_room_classification/blob/main/CNN_training.ipynb)  in which I trained a Convolutional Neural Network and predicted both on validation and test datasets.
- [CNN_augmented_dataset.ipynb](https://github.com/Iron486/Clean_messy_room_classification/blob/main/CNN_augmented_dataset.ipynb) in which I fit a Convolutional Neural Network to the augmented train dataset and predicted both on validation and test datasets..
- [Bonus_exercise.ipynb](https://github.com/Iron486/Clean_messy_room_classification/blob/main/Bonus_exercise.ipynb) that is an exercise that I did only for curiosity, calculating the average number of red,blue and green component for each pixel within the images on the train dataset.

Below, are
<p align="center"> CNN with augmented dataset - training </p>

![CNN_augmented](https://user-images.githubusercontent.com/62444785/162535217-ebe6df02-97e2-4239-8f22-508788015d1b.png =150x180) 
