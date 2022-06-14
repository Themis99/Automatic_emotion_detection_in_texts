# Automatic Emotion Detection In Texts
In this project we aim to detect emotion in texts. Specifically, we classify texts into six (6) emotion categories: Joy, Sadness, Anger, Fear, Surpise.

img

The task of this project is a classical multi task classification task. To train our model we used the Dataset founded here: (link) Where texts 
belong to one of the six aforementioned categories of emotions. We leveraged the power of transfer learning by training a pre-trained model to our dataset.
The model we used is the 

First we split our dataset to train set, test set and validation set. We trained our model using the train set and we evaluated used the test set. 
From the evaluation on the test set we achieved accuracy 82.5 which is pretty high for multi class classification task in texts.

