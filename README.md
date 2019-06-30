# Project Fraud Detection

In this section, we'll look at a credit card fraud detection dataset, and build a binary classification model that can identify transactions as either fraudulent or valid, based on provided, historical data. In a 2016 study, it was estimated that credit card fraud was responsible for over 20 billion dollars in loss, worldwide. Accurately detecting cases of fraud is an ongoing area of research.

## General Outline

### Labeled Data

The payment fraud data set (Dal Pozzolo et al. 2015) was downloaded from Kaggle. This has features and labels for thousands of credit card transactions, each of which is labeled as fraudulent or valid. In this notebook, we'd like to train a model based on the features of these transactions so that we can predict risky or fraudulent transactions in the future.

### Binary Classification

Since we have true labels to aim for, we'll take a supervised learning approach and train a binary classifier to sort data into one of our two transaction classes: fraudulent or valid. We'll train a model on training data and see how well it generalizes on some test data.

The notebook will be broken down into a few steps:

- Loading and exploring the data
- Splitting the data into train/test sets
- Defining and training a LinearLearner, binary classifier
- Making improvements on the model
- Evaluating and comparing model test performance
- Making Improvements

A lot of this notebook will focus on making improvements, as discussed in this SageMaker blog post. Specifically, we'll address techniques for:

Tuning a model's hyperparameters and aiming for a specific metric, such as high recall or precision.
Managing class imbalance, which is when we have many more training examples in one class than another (in this case, many more valid transactions than fraudulent).

### Prerequisites

For this project, the smallest GPU instance available when using SageMaker is the ml.t2.medium instance and it is perfectly adequate for completing the project.

## Modeling

A LinearLearner has two main applications:

1. For regression tasks in which a linear line is fit to some data points, and you want to produce a predicted output value given some data point (example: predicting house prices given square area).
2. For binary classification, in which a line is separating two classes of data and effectively outputs labels; either 1 for data that falls above the line or 0 for points that fall on or below the line.

In this case, we'll be using it for case 2, and we'll train it to separate data into our two classes: valid or fraudulent.

### Model Improvements

The default LinearLearner got a high accuracy, but still classified fraudulent and valid data points incorrectly. Specifically classifying more than 30 points as false negatives (incorrectly labeled, fraudulent transactions), and a little over 30 points as false positives (incorrectly labeled, valid transactions).

**1. Model optimization**
* If we imagine that we are designing this model for use in a bank application, we know that users do *not* want any valid transactions to be categorized as fraudulent. That is, we want to have as few **false positives** (0s classified as 1s) as possible. 
* On the other hand, if our bank manager asks for an application that will catch almost *all* cases of fraud, even if it means a higher number of false positives, then we'd want as few **false negatives** as possible.
* To train according to specific product demands and goals, we do not want to optimize for accuracy only. Instead, we want to optimize for a metric that can help us decrease the number of false positives or negatives. 
     
In this notebook, we'll look at different cases for tuning a model and make an optimization decision, accordingly.

**2. Imbalanced training data**
* At the start of this notebook, we saw that only about 0.17% of the data was labeled as fraudulent. So, even if a model labels **all** of our data as valid, it will still have a high accuracy. 
* This may result in some overfitting towards valid data, which accounts for some **false negatives**; cases in which fraudulent data (1) is incorrectly characterized as valid (0).

## Built With

* [Amazon SageMaker](https://aws.amazon.com/sagemaker/) - The web framework used
* [Amazon S3](https://aws.amazon.com/s3/) - The web storage used
* [Amazon API](https://aws.amazon.com/api-gateway/) - The API used

## Authors

* **Keyvan Tajbakhsh** - [keyvantaj](https://github.com/keyvantaj)

See also the list of [contributors](https://github.com/udacity/machine-learning/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
