# Automatic Emotion Detection In Texts
In this project, we aim to detect emotion in texts. Specifically, we classify texts into six (6) emotion categories: Joy, Sadness, Anger, Fear, and surprise.

## Examples

![Exp1](https://user-images.githubusercontent.com/46052843/174455562-da74af9f-741b-40c4-9efb-c76f3248c748.png)
![Exp2](https://user-images.githubusercontent.com/46052843/174455564-8d5d4f84-0d8e-4c9a-9d4d-78d96a0bd315.png)
![Exp3](https://user-images.githubusercontent.com/46052843/174455566-231dbe09-2e03-4252-ba24-195829edce0f.png)

The task of this project is a classical multi-task classification task for texts. Then we used to train and evaluate our model Dataset found in Kaggle: [(link)](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) Where texts belong to one of the six aforementioned categories of emotions. The model we used is the DistilBert [1]. 

First, we split our dataset into a train set, test set, and validation set. We trained our model using the train set and we evaluated using the test set. 
From the evaluation, we achieved an accuracy of 82.5% which is pretty high for multi-class classification tasks in texts.

We also used W&B [2] library to visualize train and evaluation loss.
![W B Chart 4_20_2022, 8_14_32 PM](https://user-images.githubusercontent.com/46052843/174452980-c5a009a7-2925-48d0-9ba2-8a3459c34697.png)




Finally, we plot the confusion matrix

![confmat](https://user-images.githubusercontent.com/46052843/174452917-cf85c478-0582-4a38-b6de-9445227ade04.png)

From the confusion matrix, we can see that the model fails to classify the Surprise class,
In the contrast, the model successfully classifies the Joy class.

## References
[1] Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.

[2] https://wandb.ai/site
