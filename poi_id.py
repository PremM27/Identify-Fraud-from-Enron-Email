#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from helper import *

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#Using all the features and then selecting the best of them later on
features_list = ['poi','salary', 'deferral_payments', 'loan_advances', \
                 'bonus', 'restricted_stock_deferred', 'deferred_income', \
                 'expenses', 'exercised_stock_options', \
                 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',\
                 'to_messages', 'from_poi_to_this_person', \
                 'from_messages', 'from_this_person_to_poi', \
                  'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
#There are two outliers, 
# 1. TOTAL : which is the total of all the fields
# 2. TRAVEL AGENCY IN THE PARK : This is a separate company to which payments are made 
outlier_list = ["TOTAL","THE TRAVEL AGENCY IN THE PARK"]
for outlier in outlier_list:
	data_dict.pop(outlier)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
### A new feature is created based on the number of messages sent and received between 
### the poi's. This was discussed in the classroom. 

for name in data_dict:
	all_messages = data_dict[name]["from_messages"] + data_dict[name]["to_messages"]
	poi_messages = data_dict[name]["from_poi_to_this_person"] + data_dict[name]["from_this_person_to_poi"]
	if all_messages == "NaNNaN" or poi_messages == "NaNNaN":
		data_dict[name]["poi_msg_fraction"] = poi_message_fraction
	else:	
		# prit type(all_messages)	
		try:
			poi_message_fraction = (float(poi_messages)) / float(all_messages)
			data_dict[name]["poi_msg_fraction"] = poi_message_fraction	
		except:
			print all_messages,poi_messages

### Adding the new feature to features list
features_list = features_list + ["poi_msg_fraction"]



my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)


print data.shape

# get the missing values in each feature
labels, features = targetFeatureSplit(data)


# Out of the 19 features we have let's find the k -best features 
from sklearn.feature_selection import SelectKBest
k = 10
select_k_best = SelectKBest(k = k)
select_k_best.fit(features,labels)

scores = select_k_best.scores_

# Sort the feature names with best scores 
data_with_features = zip(features_list[1:],scores)
data_with_features =  sorted(data_with_features,key = lambda t: -t[1])
from pprint import pprint
pprint( data_with_features[:10])

# now updating the feature list to work with only these features 
features_list = ['poi','salary', 'loan_advances', \
                 'bonus', 'deferred_income', \
                 'expenses', 'exercised_stock_options', \
                 'shared_receipt_with_poi', 'long_term_incentive', 'restricted_stock',\
                 'from_poi_to_this_person'] 

features_list = features_list + ["poi_msg_fraction"]

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



 ### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html






### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)



### Preprocessing Data : Normalize and apply PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


estimators = [
		('std',StandardScaler()),
		('pca',PCA())
]

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
# clf_guassian = GaussianNB()
# clf_guassian.fit(features_train,labels_train)

# pred = clf_guassian.predict(features_test)


#Decision tree classifer 
from sklearn import tree 
clf_tree = tree.DecisionTreeClassifier(random_state=44,max_features=4,min_samples_split=2)

clf_tree_params = {
	'clf__min_samples_split' : range(1,6),
	'clf__max_depth' : [2,3,4,None],
	'clf__max_features' : range(1,5),
	'clf__presort' : [False,True]
}

# optimize_classifier(clf_tree, clf_tree_params,features_train,labels_train, 
			# features_test, labels_test)

# clf_tree.fit(features_train,labels_train)

# pred = clf_tree.predict(features_test)
# print_metrics(pred,labels_test)

#Logistic Regression Classifier 
from sklearn.linear_model import LogisticRegression

clf_logistic = LogisticRegression()
clf_log_params = {
	'clf__C' : [1e-5,1e-3,1],
	'clf__penalty' : ['l1','l2'],
	'clf__tol' : [1e-4, 1e-6]
}

# optimize_classifier(clf_logistic, clf_log_params,features_train,labels_train, 
			# features_test, labels_test)

#Random boost classifier 
from sklearn.ensemble import RandomForestClassifier
clf_randomForrest = RandomForestClassifier(random_state=42)
clf_rf_params = {
	'clf__n_estimators' : [3,6,10],
	'clf__min_samples_split' : [2,3],
	'clf__max_depth' : [4,None],
	'clf__max_features' : [4] #sqrt(n_features),
}
# optimize_classifier(clf_randomForrest, clf_rf_params,features_train,labels_train, 
			# features_test, labels_test)

# clf_randomForrest.fit(features_train,labels_train)

# pred = clf_randomForrest.predict(features_test)
# print_metrics(pred,labels_test)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# estimators.append(("clf",GaussianNB()))
# estimators.append(("clf",LogisticRegression()))



# ## Random Forrest
# estimators.append(("clf",RandomForestClassifier(random_state=44)))
# model = Pipeline(estimators)
# clf = optimize_classifier(model,clf_rf_params,features_train,labels_train,features_test,labels_test)

## Decision Tree
# clf_tree = tree.DecisionTreeClassifier(random_state=42,max_features=4,min_samples_split=2)
# estimators.append(("clf",clf_tree))
# model = Pipeline(estimators)
# clf = optimize_classifier(model,clf_tree_params,features_train,labels_train,features_test,labels_test)
## Logistic Regression
# estimators.append(("clf",clf_logistic))
# model = Pipeline(estimators)
# clf = optimize_classifier(model,clf_log_params,features_train,labels_train,features_test,labels_test)


## GuassianNB - Final classifier to be submitted. 
estimators.append(("clf",GaussianNB()))
model = Pipeline(estimators)
clf = optimize_classifier(model,{},features_train,labels_train,features_test,labels_test)


# from tester import test_classifier
# test_classifier(clf ,my_dataset,features_list)
# pipeline = model.fit(features_train,labels_train)
# pred =  pipeline.predict(features_test)

# print_metrics(pred,labels_test)

# clf = pipeline
dump_classifier_and_data(clf, my_dataset, features_list)
