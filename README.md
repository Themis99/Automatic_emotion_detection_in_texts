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

![confmat](https://user-images.githubusercontent.com/46052843/174452917-cf85c478-0582-4a38-b6de-9445227ade04.png)

From the confusion matrix we can clearly see that the model fails to classify the Surprise class,
In the contrast the model successfully classifies the Joy class.

## References

[1]

[2]
