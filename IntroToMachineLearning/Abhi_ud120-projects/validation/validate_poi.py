#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]


data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson13_keys.pkl')
labels, features = targetFeatureSplit(data)




### it's all yours from here forward! 
#Decision Tree without validation

from sklearn.tree import DecisionTreeClassifier
clf= DecisionTreeClassifier()
clf.fit(features,labels)
pred= clf.predict(features)

from sklearn.metrics import accuracy_score
print(accuracy_score(pred, labels))




# Implementing cross validation

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(features, labels,test_size= 0.3, random_state= 42)
print(X_train.shape)
print(X_test.shape)
print(len(Y_train))
print(len(Y_test))



# Implementing Decision Tree Classifier using Grid Search
from sklearn.tree import DecisionTreeClassifier
clf= DecisionTreeClassifier()
clf.fit(X_train, Y_train)
pred= clf.predict(X_test)


# Calculating accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(pred, Y_test))








