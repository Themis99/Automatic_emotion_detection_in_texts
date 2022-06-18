# Automatic Emotion Detection In Texts
In this project we aim to detect emotion in texts. Specifically, we classify texts into six (6) emotion categories: Joy, Sadness, Anger, Fear, Surpise.



## Examples
(1)

(2)

(3)

The task of this project is a classical multi-task classification task for texts. The we used to train and evaluate our modelthe Dataset founded in Kaggle: [(link)](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) Where texts belong to one of the six aforementioned categories of emotions. The model we used is the DistilBert [1]. 

First, we split our dataset into train set, test set, and validation set. We trained our model using the train set and we evaluated using the test set. 
From the evaluation , we achieved an accuracy of 82.5% which is pretty high for multi-class classification task in texts.

We also used W&B [2] library to visualize train and evaluation loss.





Finaly we plot the confusion matrix

(img)

From the confusion matrix we can clearly see that...

## References

[1]

[2]
