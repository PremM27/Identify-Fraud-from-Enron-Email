
# P5: Intro to Machine Learning, Identify Fraud from Enron Email
By Prem Mithilesh Morampudi as a Part of the Academic Project at Udacity Data Analyst Nanodegree. 

## Short Questions
### __Question 1: *Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?*__
The goal of the project is to try to figure out if there are any patterns within the emails of people who were the persons of intrest in the fraud case using the Financial Data from the Enron Scandal which is made publicly available by US Federal Energy Regulatory Comission during its Investigation and to create a Model which can efficiently Identify the Person of Intrest in the Scandal using the Data. 
About the Dataset:
Enron is an 63.4 Billion Dollar Energy trading Company, Which was one of the top 10 Companies in America with a share value of around 90.75 in the Mid of 2000 and filed Bankruptcy in November of 2001, With a share value collapsed to less than a Dollar by the end of 2011. This is the Largest corporate Scam and Bankruptcy filled till then with dozens of People in Jail, Thousands of people who lost their Jobs. This datasets is one of the most intresting Datasets to investigate its features to learn and apply Machine Learning or Analytical skills. The Dataset contained 146 records of financial data, Most of which are the senior management of Enron, which is being used to develop a predictive model to identify the Person of Intrest from its features. 

__*The dataset Contained 146 Records, 14 Financial Features and 6 email Features with one feature labeled "POI". 18 of these Records were Labled as "Person of Intrest".*__

Outliers have been observed in the dataset by looking at the PDF given in the dataset that contains information about the financial data and found that there are __*Two Outliers*__, i.e: __*TOTAL*__ and __*TRAVEL AGENCY IN THE PARK*__. The total was the Sum of the Financial data so, This record is being Executed and The Travel Agency in the Park is a firm and Not a Person, So these two records were considered to outliers. Also there are many records with missing values reported as NaN, which are the following

* Deferred Income
* Deferral Payments 
* Loan Advances 
* Director Fees
* Restricted Stock Deferred

All of the missing values have been converted to **0** for the analysis purposes use featureFormat utility. 


#### __Question 2: *What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it.*__

##### Feature Selection
There were 18 features in the dataset and not all of the were of equal weightage. Earlier I tried to work with all the
features but later I decided to use an automated algorithm for using the important features for training. 

I used the **SelectKBest** algorithm to decide on the best features based on the score returned. I selected the 
number of features to be used to 10 which were empirically determined. 
Here is the list of features with the scores returned by the algorithm. 
| Feature  | Score |
|----------|-------|
|exercised_stock_options | 25.097541528735491
 |bonus|21.060001707536571
 |salary |18.575703268041785
 |deferred_income| 11.595547659730601
 |long_term_incentive |10.072454529369441
 |restricted_stock| 9.3467007910514877
 |shared_receipt_with_poi| 8.7464855321290802
 |loan_advances| 7.2427303965360181
 |expenses| 6.2342011405067401
 |from_poi_to_this_person| 5.3449415231473374

##### Scaling
Scaling was used in pre-processing the data before passing it to the model for training. Scaling is necssary for machine learning algorithms as if some of the features have very large / small values then the model may bias/ignore these features. 
The data was transformed to be centered around mean and scaling was done to the unit variance of data. 
I have added a scaling operation in my pipeline. 

##### New Feature Creation
__*New Feature:*__ *(poi_msg_fraction)* I have created a new feature to quantify the fraction of communication between all the emails Sent & Received by a Person to the total messages Sent & Received with the Person of Intrest as It is very likely that the person of interest might communicate more often with each other. 

The model perfornamce was tested with this feature and here are the results 

|Status | Precision | Recall  | F-Score|
|-------|-----------|---------|--------|
|**With New Feature** |0.375|0.6|0.4615
|**Without New Feature** |0.4|0.3333| 0.3636

From the results we see that our overall results improved when using the new feature and hence it's successful. 

#### __Question 3: *What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?*__

I have tried four algorthims: 
1. Gaussian Naive Bayes
2. Decision Tree Classifier
3. Logistic Regression and 
4. Random Forest Classifier
to create a model and the best performance was observed from the Gaussian Naive Bayes Algorithm .
Parameter tuning is done for all the algorithms other than Guassian NB. Even after parameter tuning the alogorithm 
closest to GuassianNB is Random Forrest. I think with more work it can be made to improve the score more. 

Performance of various classifiers tested without the new feature added
|Status | Precision | Recall  | F-Score|
|-------|-----------|---------|--------|
|**Guassian Naive Bayes**| **0.4**|**0.333**|**0.3636**|
|Decision Trees|0.16|0.16|0.16
|Logistic Regression|0.4|0.333|0.3636|
|Random Forrest|0.2|0.16|0.18|

Performance of various classifiers tested with  the new feature added
|Status | Precision | Recall  | F-Score|
|-------|-----------|---------|--------|
|**Guassian Naive Bayes**| **0.375**|**0.61**|**0.461**|
|Decision Trees|0.3|0.4|0.3636|
|Logistic Regression|0.142|0.2|0.16|
|Random Forrest|0.4|0.4|0.4|

We see that in both the cases with or without the new feature Guassian Naive Bayes performs best based on **f1-score**. 
### __Question 4: *What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm?*__ 

## TODO 
## Discuss about parameter tuning and not validation

Parameter tuning is the process of tuning the hyper-parameters of the algorithm to get the best classifier perfornamce. 
If the parameters of an algorithm are not tuned then even though an algorithm is suitable for the data it may
give sub-par performance. With the tuned hyper parameters we can increase the performance and get the best classifier.
Tuning the hyperpameters also results in how much complex our algorithm is in creating a decision boundary.

For the various algorithms tested I used the GridSearchCV method in sklearn to auto-tune the parameters over the 
common hyper parameters. The values for the parameters to search the best algorithm are chosen based on the 
available options and also the number of features we have used to train upon. All the performances reported are for 
the best tuned classifier given the parameters provided. 

Here is a table of parameters tuned for each of the algorithms 

**Guassian Naive Bayes** :
Guassian Naive Bayes only has one parameter to tune which is the **priors**. The default values of the priors
is None and that is used. So no paramter tuning for Guassian NB is done. 

**Decision Tree** : 
```clf_tree_params = {
	'clf__min_samples_split' : range(1,6),
	'clf__max_depth' : [2,3,4,None],
	'clf__max_features' : range(1,5),
	'clf__presort' : [False,True]
}
```

**Logistic Regression** :
```
clf_log_params = {
	'clf__C' : [1e-5,1e-3,1],
	'clf__penalty' : ['l1','l2'],
	'clf__tol' : [1e-4, 1e-6]
}
```

**Random Forrest** :
```
clf_rf_params = {
	'clf__n_estimators' : [3,6,10],
	'clf__min_samples_split' : [2,3],
	'clf__max_depth' : [4,None],
	'clf__max_features' : [4] #sqrt(n_features),
}
```



### __Question 5:*What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?*__

Validation is the process of making sure that the Model does not get biased by dataset used to train it. If validation is not done properly, The model gets biased to the training Data used by the model and the Models fails to perform efficiently across the Real Data. Validation is a Method to prevent the Overfitting or Underfitting of the Model. 

For the validation of the models, the data is divided in two set of training and testing. The models were trained 
only on the training dataset and then the results are provided for the testing dataset to make sure that we are not overfitting on the data. 

The tester.py uses a 1000-fold cross validation to test the model which is better approach as the data is small 
and k-fold cross validation is more robust. But the results reported are on StratifiedShuffleSplit approach with 
100 iteration. . 


### __Question 6:*Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.*__


For the Evaluation Metrics I have used the Precesion and Recall metrics as intended by the rubric.
Accuracy will not be a good metric to work with this as the the data is highly imabalanced and if a model just 
predicts all of the data points as non-POI then also the accuracy will be very high. Precision and recall take
into consideration the correctness of the prediction based on the true values and are defined as follows in  context
of the project 

**Precision** : It's the ratio of correctly predicted poi's to all predicted poi's. In simple terms out of all the persons
predicted as poi how many of them are true poi's. The score of **0.375** says that if we out of 100 persons predicted as 
poi 37.5 are true person of interests. 


**Recall** : It's the ratio of correctly predicted poi's to all onservation in the poi-class. In simple terms out of 
all the persons that are truly poi how many of them did we label correctly. The score of **0.61** tells that out of 100 persons we labelled 61 correctly.

Based on the precision and recall values, we can say that our model is better at not classiying a poi as non-poi than classifying a  non-poi as poi. . 


### References:
- https://en.wikipedia.org/wiki/Enron
- https://www.cs.cmu.edu/~./enron/
- http://scikit-learn.org/stable/documentation.html
- http://scikit-learn.org/stable/
- http://machinelearningmastery.com/an-introduction-to-feature-selection/
- http://rushdishams.blogspot.com/2011/03/precision-and-recall.html
- http://stackoverflow.com/questions/7142227/how-do-i-sort-a-zipped-list-in-python




```python

```
