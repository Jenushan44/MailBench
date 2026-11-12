# What is Logistic Regression?

Logistic Regression is a supervised machine learning algorithm used for classification. It calculates a linear score using input features, weights and bias. During training, the model adjusts its weights and bias to better predict the correct labels. The linear score is passed through a sigmoid function which converts it into a probabiiluty between 0 and 1. A classification threshold, usually 0.5, is then used to predict either class 0 or class 1. In MailBench, class 0 represents legitimate emails and class 1 represents spam/phishing emails. 

In logistic regression, the bias is a learned constant added to the weighted sum of input features that allows it to shift its decision boundary instead of forcing it through the origin helping the model fit the training data. It uses weights to determine how much each input feature contributes to the prediction. Each feature is multiplied by its corresponding weight and the results are added together with a bias to calculate a linear score. The weights and bias are learned during training. The linear score is passed through the sigmoid function which converts it into a probability between 0 and 1. Finally, a classification threshold usually 0.5, determines the predicted class. 

# How does Logistic Regression Learn? 

During training, Logistic Regression compares its predicted probabilities aginst the correct output labels and calculates the loss using binary cross-entropy (log loss). An optimization algorithm then adjusts the weights and bias to minimize the loss. 

# TF-IDF Vectorization

TF-IDF converts email text into numerical features based on word frequency and how common words are across documents. TF-IDF does not determine whether words indicate spam. Logistic Regression learns those relationships with the TF-IDF features and the correct training labels. In MailBench, fit_transform() is used on the training data to learn the vocabulary and IDF statistics and convert emails into numerical features. transform() is used on the test data to apply the same learned representation without fitting it again which prevents information from the test set from leaking into preprocessing.