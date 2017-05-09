from sklearn.metrics import f1_score,recall_score,precision_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV


def print_metrics(pred,labels):
	r = recall_score(labels,pred)
	f = f1_score(labels,pred)
	p = precision_score(labels,pred)

	print "precision : {}\nrecall: {}\nf-score: {}".format(p,r,f)


#function to optimize a classifier given a parameter list 
def optimize_classifier(clf, params, features_train, labels_train, features_test, labels_test):
	#using the startified shuffle split with 100 iteratioms

	cv = StratifiedShuffleSplit(labels_train, n_iter=100, test_size=0.3, random_state=42)

	grid_cv = GridSearchCV(clf,params, scoring="f1",cv=cv)
	optimized_clf = grid_cv.fit(features_train,labels_train)
	optimized_clf = optimized_clf.best_estimator_

	print optimized_clf
	#print the result 
	pred = optimized_clf.predict(features_test)
	print_metrics(pred,labels_test)

	return optimized_clf