#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


def ten(elm):
	return elm[1]



### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )
data_dict.pop('TOTAL')
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

for point in data:
	salary= point[0]
	bonus= point[1]
	plt.scatter(salary, bonus)

testdata=data.tolist()
#testdata= testdata.sort(key=ten, reverse= True)
testdata.sort(key=ten, reverse= True)
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

### your code below

outlier= testdata[0][1]
print(testdata)

#print(data_dict)





