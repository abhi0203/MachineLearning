#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary','total_stock_value','total_payments','ContactWithPOI','bonus','loan_advances','expenses','exercised_stock_options','long_term_incentive','shared_receipt_with_poi','restricted_stock']
features_list = ['poi','salary','total_stock_value','ContactWithPOI','bonus','expenses','exercised_stock_options','long_term_incentive','shared_receipt_with_poi','restricted_stock']
# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

print("Length of dataset", len(data_dict))

nanSalaryCount=0
nanBonusCount=0
nanTotalPaymentCount=0
nanTotalStockValue=0
nanExpenses= 0
nanExercisedStockValue= 0
nanLongTermIncentive=0
nanRestrictedStock=0

poiCount=0
nonPoiCount=0

for key in data_dict.keys():
	if data_dict[key]['from_poi_to_this_person']=='NaN':
		data_dict[key]['from_poi_to_this_person']=0
	
	if data_dict[key]['from_this_person_to_poi']=='NaN':
		data_dict[key]['from_this_person_to_poi']=0

	data_dict[key]['ContactWithPOI']= data_dict[key]['from_poi_to_this_person']+ data_dict[key]['from_this_person_to_poi']

	if data_dict[key]['salary']=='NaN':
		data_dict[key]['salary']=0

	if data_dict[key]['total_stock_value']=='NaN':
		data_dict[key]['total_stock_value']=0

	if data_dict[key]['total_payments']=='NaN':
		data_dict[key]['total_payments']=0

	if data_dict[key]['bonus']=='NaN':
		data_dict[key]['bonus']=0

	if data_dict[key]['loan_advances']=='NaN':
		data_dict[key]['loan_advances']=0

	if data_dict[key]['expenses']=='NaN':
		data_dict[key]['expenses']=0

	if data_dict[key]['exercised_stock_options']=='NaN':
		data_dict[key]['exercised_stock_options']=0

	if data_dict[key]['long_term_incentive']=='NaN':
		data_dict[key]['long_term_incentive']=0

	if data_dict[key]['shared_receipt_with_poi']=='NaN':
		data_dict[key]['shared_receipt_with_poi']=0

	if data_dict[key]['restricted_stock']=='NaN':
		data_dict[key]['restricted_stock']=0
	
	if data_dict[key]['poi']==True:
		poiCount+=1
	else:
		nonPoiCount+=1

### Task 2: Remove outliers
### For now we know that the only outlier we are aware of is TOTAL. So we are deleting the TOTAL key from data_dict.

del data_dict['TOTAL']



### Task 3: Create new feature(s)
### Here I would like to add a new feature to the dataset known as ContactWithPOI.
### This feature is nothing but the sum of emails received from and sent to POI by a person. 
### The thought is, if a person sends or receives too many emails from a POI, then he or she can be a POI.

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)





### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
#clf = GaussianNB(priors=None)
#clf= SVC(kernel='linear')
clf= DecisionTreeClassifier(max_depth=5,splitter='best', criterion='entropy')


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

#Checking how useful the features are.

from sklearn.feature_selection import SelectKBest
featureSelector= SelectKBest(k=7)
featureSelector.fit(features_train,labels_train)
chosenTrainFeatures= featureSelector.transform(features_train)
#chosenTrainLabels= featureSelector.transform(labels_train)
chosenTestFeatures= featureSelector.transform(features_test)
#chosenTestLabels= featureSelector.transform(labels_test)





### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

clf.fit(chosenTrainFeatures,labels_train)
pred_train= clf.predict(chosenTrainFeatures)
pred= clf.predict(chosenTestFeatures)
print("Prediction is ", pred)

from sklearn.metrics import accuracy_score, recall_score, precision_score

accuracy_train= accuracy_score(labels_train, pred_train)
print("Trainig Accuracy is ", accuracy_train)
accuracy= accuracy_score(labels_test, pred)
print("Accuracy is ", accuracy)
precision= precision_score(labels_test, pred)
print("Precision is ", precision)
recall= recall_score(labels_test, pred)
print("Recall is ", recall)


dump_classifier_and_data(clf, my_dataset, features_list)