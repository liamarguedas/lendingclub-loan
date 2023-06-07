# Lending Club Loan Project Summarize

## Problem Description

LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California. It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform.

Solving this case study will give us an idea about how real business problems are solved using EDA and Machine Learning. In this case study, we will also develop a basic understanding of risk analytics in banking and financial services and understand how data is used to minimise the risk of losing money while lending to customers.

When the company receives a loan application, the company has to make a decision for loan approval based on the applicant’s profile. Two types of risks are associated with the bank’s decision:

If the applicant is likely to repay the loan, then not approving the loan results in a loss of business to the company If the applicant is not likely to repay the loan, i.e. he/she is likely to default, then approving the loan may lead to a financial loss for the company The data given contains the information about past loan applicants and whether they ‘defaulted’ or not. The aim is to identify patterns which indicate if a person is likely to default, which may be used for takin actions such as denying the loan, reducing the amount of loan, lending (to risky applicants) at a higher interest rate, etc.

When a person applies for a loan, there are two types of decisions that could be taken by the company:

**Loan accepted:** If the company approves the loan, there are 3 possible scenarios described below:

- **Fully paid:** Applicant has fully paid the loan (the principal and the interest rate)

- **Current:** Applicant is in the process of paying the instalments, i.e. the tenure of the loan is not yet completed. These candidates are not labelled as 'defaulted'.

- **Charged-off:** Applicant has not paid the instalments in due time for a long period of time, i.e. he/she has defaulted on the loan

**Loan rejected:** The company had rejected the loan (because the candidate does not meet their requirements etc.). Since the loan was rejected, there is no transactional history of those applicants with the company and so this data is not available with the company (and thus in this dataset)

## Project Objective

Trained a Machine Learning model capable of classifying and solve the LendingClub problem.

## Decision Making process

From the exploratory data analsis I found out that the target (loan_status) expected behavior since there is more loans fully paid than loans charged off, the target balance from the training data was:

![Target Balance](https://raw.githubusercontent.com/liamarguedas/data/main/Data-Science/Lending-Club-Loan/Summary-Charts/Target%20Balance.png)

Unfortunately the dataset had very few features that described those charged off loans with precision, so the unbalanced was unfixable so it was not worth implementing SMOTE. 

From the feature engineering I tried to create features that could describe whether that loan could get charged off or not and ended up creating the more valuable features for predicting these observations.

The original data correlation look like:

![Target Correlation](https://raw.githubusercontent.com/liamarguedas/data/main/Data-Science/Lending-Club-Loan/Summary-Charts/Target%20Correlation.png)

There was some features that could describe the data and from the feature importance It was noticeable that they were important to the model. Since it was an unbalanced target I needed to be careful with what metrics to focus on, since accuracy would not be a perfect metric for the problem. So for this problem the best metrics to focus on were Precision and Recall.

I decided to used a baseline score in order to understand whether a model is useful or no. A baseline score is the score a model would get if only predicts the bigger unbalanced class.

- ZeroR score: 

    Score: 0.803966805985839

    Since the classes are unbalanced if the model only predicts 0 (Fully Paid) it will get an score of 0.80, in order to built a useful model the algorithm need to achieve at least higher than the ZeroR score. This will tell us (Together with predictions metrics) that the model can identify and predict 1s (Charged Off).

- Random Rate Classifier (Weighted Guessing):

    Score: 0.6847918741779999

    The weightedG score was roughly 0.68, so the model has to score at least 0.68 to be useful, I decided to used the ZeroR in that case.

Knowing my baselines scores and the metrics I needed, the next thing was to understand the data in order to select a model, so the data was:

- Data is not normally distributed
- Some of the features are skewed
- Data has outliers
- Data has no missing values
- Data has labelled target
- Data has a binary output (Classification Problem)

With that on mind I decided to trained 4 different models and see the performance they have with the data, the models I chosen were:

- Naïve Bayes Classifier
- Random Forests
- Gradient Boosted Trees
- Keras NN

I trained a model using cross validation with 5 folds and this are the results I got:

![Models Performance](https://raw.githubusercontent.com/liamarguedas/data/main/Data-Science/Lending-Club-Loan/Summary-Charts/Model%20K-Fold%20Performance.png)

In order to take a decision I group the models and calculated the mean and median with the Precision and Recall metrics to analise performance. 

![Group Performance](https://raw.githubusercontent.com/liamarguedas/data/main/Data-Science/Lending-Club-Loan/Summary-Charts/Model%20Performance%20Mean-Median.png)

## Final Decision

From all the models I tried, it looks like Gradient Boosting had the better performance in all the folds in Precision and Naive Bayes in Recall (Performing horrible in Precision). But when I look at the F1-score (The optimal blend of Precision and Recall) it looks like the TensorFlow Keras performed better, so this was a hard decision and it needed to be taken thinking in the implementation of a model like this. For that, I took a look at Gradient Boosting vs. Keras NN.

**Keras NN vs. Gradient Boosting**

**- Implementation Cost**

TensorFlow is usually expensier than Gradient Boost to implement.

**- Training Time**

TensorFlow took around 10 minutes to compute each fold, Gradient Boost took around 3 minutes.

**- Overfitting & Underfitting**

TensorFlow was stopped by an early stop to avoid overfitting from the metrics it did not overffit. Gradient boosting is a greedy algorithm and can overfit a training dataset quickly. I used regularization methods that penalize various parts of the algorithm and did not overffit

**- Outliers**

Keras Can handle outliers but it affects performance if they are too many while Gradient Boost is Robust to outliers

Since the improvment was no that big from Gradient Boost to Keras NN and taking into consideration that there is actually a lot of outliers, I have choseen to use Gradient Boost. Which is better for the implementation cost and computational time.

With the model selected I performed a cross validation one more time just to capture some metrics to analise and make conclusions, the performance of Gradient Boost was:

![Gradient Boost Performance](https://raw.githubusercontent.com/liamarguedas/data/main/Data-Science/Lending-Club-Loan/Summary-Charts/Gradient%20Boost%20K-Fold%20Performance.png)

I built a classification report from the last fold in order to see how it managed to predict each target.

![Classification report](https://raw.githubusercontent.com/liamarguedas/data/main/Data-Science/Lending-Club-Loan/Summary-Charts/Classification%20Report.png)

As seen from the report, even the best performance model did not managed to get a better recall metric for the 1 (Charged Off) class. In order to see the results, I created a confusion matrix.

![Confusion Matrix](https://raw.githubusercontent.com/liamarguedas/data/main/Data-Science/Lending-Club-Loan/Summary-Charts/Confusion%20Matrix.png)

As seen from the confusion matrix the model predicted many Charged Off loans as Fully paid (Actually it predicted more wrong that it did right) so of course the data was not describing with precision the Charged Off loans. 

I trainned a model with the entire dataset in order to give a first hand solution and analise each feature importance to the model, the results were:

![Feature Importance](https://raw.githubusercontent.com/liamarguedas/data/main/Data-Science/Lending-Club-Loan/Summary-Charts/Feature%20Importance.png)

In the graph we can see that almost the entire dataset had little to none importance to the model and the few feature that had importance were the ones I feature engineered from the data. Then a questions arises, Does deleting these features are gonna improve models performance? 

Quick answer is probably not. Because Machine Learning algorithms do not work in an additative way, they work by weights and deleting those weights since they have no impact the model is gonna have pretty much the same performance.

## Did my decision solve the LendingClub problem?

With the data provided and transformations made in the project I did not solve the LendingClub problem. With an average recall of 0.44 and 0.61 average f1-score predicting 1s (Charged Off) I can not give a practical and sustainable solution for the problem. I managed to get a very high predicting value around 10% more than the baseline scores, creating a very solid and useful classification for the problem. Around 51% of the Charged Off observations are predicted correctly, which is pretty for an unbalanced good but there still more than half of the sample being wrongly predicted.

## What is the root cause of the performance?

The data provided did not have many indications that describe a Charged Off loan with precision, from the exploratory data analysis I found that the more valuable and descriptive features were the ones I created by feature engineering the address column besides that there was also a couple of descriptive feature like term or interest rate that have some value and it can be shown in the feature importance as high impact features for the model performance.

## What could improve the performance?

Creating data collection systems based in those most common reasons of why loans get charged off and gathering specific insights from that data. Some example of that data would be:

- Days past last payment
- Whether there are some legals actions against the borrower or no
- FICO Score
- Collateral value provided by the borrower
- Outstanding Debt
- Payment Behaviour
- More info about the borrower (Age, marital status, etc..)

All of this data can be gather easily from relational databases and can be implemented in the project always being carefully to not introduce data leakage. Something it can be improved is the model hyperparameters, with the right data and settings Gradient Boosting Classifier is an amazing algorithm for this project.

## Was the project useful?

The project introduces a baseline for what can be done with Machine Learning with the right tools, risk can be easily mapped with the right data and analytical tools. With the project I managed to predict around 50% of the times whether or not a loan is getting charged off and I understood what was needed in order to make a higher percentage of right predictions, the project was useful but it is not worth putting into production because there is a lot of information that is going to be lost.