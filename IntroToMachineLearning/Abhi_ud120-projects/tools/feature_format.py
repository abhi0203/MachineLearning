#!/usr/bin/python

""" 
    A general tool for converting data from the
    dictionary format to an (n x k) python list that's 
    ready for training an sklearn algorithm

    n--no. of key-value pairs in dictonary
    k--no. of features being extracted

    dictionary keys are names of persons in dataset
    dictionary values are dictionaries, where each
        key-value pair in the dict is the name
        of a feature, and its value for that person

    In addition to converting a dictionary to a numpy 
    array, you may want to separate the labels from the
    features--this is what targetFeatureSplit is for

    so, if you want to have the poi label as the target,
    and the features you want to use are the person's
    salary and bonus, here's what you would do:

    feature_list = ["poi", "salary", "bonus"] 
    data_array = featureFormat( data_dictionary, feature_list )
    label, features = targetFeatureSplit(data_array)

    the line above (targetFeatureSplit) assumes that the
    label is the _first_ item in feature_list--very important
    that poi is listed first!
"""


import numpy as np
from sklearn.preprocessing import MinMaxScaler

def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    """ convert dictionary to numpy array of features
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True will omit any data points for which
            all the features you seek are 0.0
        remove_any_zeroes = True will omit any data points for which
            any of the features you seek are 0.0
        sort_keys = True sorts keys by alphabetical order. Setting the value as
            a string opens the corresponding pickle file with a preset key
            order (this is used for Python 3 compatibility, and sort_keys
            should be left as False for the course mini-projects).
        NOTE: first feature is assumed to be 'poi' and is not checked for
            removal for zero or missing values.
    """


    return_list = []

    # Key order - first branch is for Python 3 compatibility on mini-projects,
    # second branch is for compatibility on final project.
    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print("error: key ", feature, " not present")
                return
            value = dictionary[key][feature]
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append( float(value) )

        # Logic for deciding whether or not to add the data point.
        append = True
        # exclude 'poi' class as criteria.
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        ### if all features are zero and you want to remove
        ### data points that are all zero, do that here
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        ### if any features for a given data point are zero
        ### and you want to remove data points with any zeroes,
        ### handle that here
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        ### Append the data point if flagged for addition.
        if append:
            return_list.append( np.array(tmp_list) )

    return np.array(return_list)


def targetFeatureSplit( data ):
    """ 
        given a numpy array like the one returned from
        featureFormat, separate out the first feature
        and put it into its own list (this should be the 
        quantity you want to predict)

        return targets and features as separate lists

        (sklearn can generally handle both lists and numpy arrays as 
        input formats when training/predicting)
    """

    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )

    scaledFeatures= featureScaling(features)

    return target, scaledFeatures

def featureScaling(features):
    salaryFeature=[]
    total_stock_valueFeature=[]
    #total_paymentsFeature=[]
    ContactWithPOIFeature=[]
    bonusFeature=[]
    #loanAdvancesFeature= []
    expensesFeature=[]
    exercisedStockOptionsFeature=[]
    longTermIncentiveFeature=[]
    sharedReceiptWithPoiFeature= []
    restrictedStockFeature=[]

    print("This is Abhiram")

#total_paymentsFeature.append(item[2])
#loanAdvancesFeature.append(item[5])
    for item in features:
        salaryFeature.append(item[0])
        total_stock_valueFeature.append(item[1])
        ContactWithPOIFeature.append(item[2])
        bonusFeature.append(item[3])
        expensesFeature.append(item[4])
        exercisedStockOptionsFeature.append(item[5])
        longTermIncentiveFeature.append(item[6])
        sharedReceiptWithPoiFeature.append(item[7])
        restrictedStockFeature.append(item[8])


    
    
    salaryFeatureArr= np.array(salaryFeature)
    total_stock_valueFeatureArr= np.array(total_stock_valueFeature)
    #total_paymentsFeatureArr= np.array(total_paymentsFeature)
    ContactWithPOIFeatureArr= np.array(ContactWithPOIFeature)
    bonusArr= np.array(bonusFeature)
    #loanAdvanceArr= np.array(loanAdvancesFeature)
    expensesArr= np.array(expensesFeature)
    exercisedStockOptionsArr= np.array(exercisedStockOptionsFeature)
    longTermIncentiveArr= np.array(longTermIncentiveFeature)
    sharedReceiptWithPoiArr= np.array(sharedReceiptWithPoiFeature)
    restrictedStockArr= np.array(restrictedStockFeature)

    #exercised_stock_optionsFeatureArr= np.array(exercised_stock_optionsFeature)
    '''
    Features under Consideration
    'salary','total_stock_value','total_payments','ContactWithPOI','bonus','loan_advances',
    'expenses','exercised_stock_options','long_term_incentive','shared_receipt_with_poi','restricted_stock']
    '''


    salaryScaler= MinMaxScaler()
    scaledSalary = salaryScaler.fit_transform(salaryFeatureArr)

    totalStockValueScaler= MinMaxScaler()
    scaledTotalStockValue= totalStockValueScaler.fit_transform(total_stock_valueFeatureArr)

    #totalPaymentScaler= MinMaxScaler()
    #scaledTotalPayment= totalPaymentScaler.fit_transform(total_paymentsFeatureArr)

    contactWithPOIScaler= MinMaxScaler()
    scaledContactWithPOI = contactWithPOIScaler.fit_transform(ContactWithPOIFeatureArr)

    bonusScaler= MinMaxScaler()
    scaledBonus= bonusScaler.fit_transform(bonusArr)

    #loanAdvanceScaler= MinMaxScaler()
    #scaledLoanAdvance= loanAdvanceScaler.fit_transform(loanAdvanceArr)

    expenseScaler= MinMaxScaler()
    scaledExpense= expenseScaler.fit_transform(expensesArr)

    exercisedStockOptionScaler= MinMaxScaler()
    scaledExercisedStockOption = exercisedStockOptionScaler.fit_transform(exercisedStockOptionsArr)

    longTermIncentiveScaler= MinMaxScaler()
    scaledLongTermIncentive= longTermIncentiveScaler.fit_transform(longTermIncentiveArr)

    sharedReceiptWithPoiScaler= MinMaxScaler()
    scaledSharedreceiptWithPoi= sharedReceiptWithPoiScaler.fit_transform(sharedReceiptWithPoiArr)

    restrictedStockScaler= MinMaxScaler()
    scaledRestrictedStock= restrictedStockScaler.fit_transform(restrictedStockArr)



    #stockScaler= MinMaxScaler()
    #scaledStock= stockScaler.fit_transform(exercised_stock_optionsFeatureArr)

    #print(salaryScaler.transform(np.array([[200000.0]])))
    #print(stockScaler.transform(np.array([[1000000.0]])))

    retFeatures= np.column_stack((scaledSalary, scaledTotalStockValue,scaledContactWithPOI,scaledBonus,scaledExpense,scaledExercisedStockOption,scaledLongTermIncentive,scaledSharedreceiptWithPoi,scaledRestrictedStock))

    #print(retFeatures)

    ###If the feature is single feture then you have to reshape the feature.
    '''
    print(type(scaledSalary))
    retFeatures= np.array(scaledSalary).reshape(len(scaledSalary),1)
    print(type(retFeatures))
    '''
    print("This is Abhiram 2")
    return retFeatures








