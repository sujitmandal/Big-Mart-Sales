import os
import json
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
rcParams['figure.figsize'] = 12, 8

# Github: https://github.com/sujitmandal
# Pypi : https://pypi.org/user/sujitmandal/
# LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/ 

if not os.path.exists('plot/RandomForest'):
    os.mkdir('plot/RandomForest')

def RandomForest(train, test):
    train_identifier = []
    for i in train['Item_Identifier']:
        train_identifier.append(i)

    train_keys = []
    train_values = []

    for i in range(len(train_identifier)):
        train_keys.append(i + 1)
        train_values.append(train_identifier[i])

    train_item_identifier = {}
    train_item_identifier = dict(zip(train_keys, train_values))
    #print(item_identifier)

    test_identifier = []
    for i in test['Item_Identifier']:
        test_identifier.append(i)

    test_keys = []
    test_values = []

    for i in range(len(test_identifier)):
        test_keys.append(i + 1)
        test_values.append(test_identifier[i])

    test_item_identifier = {}
    test_item_identifier = dict(zip(test_keys, test_values))
    #print(item_identifier)

    target = 'Item_Outlet_Sales'
    IDcol = ['Item_Identifier','Outlet_Identifier']

    predictors = [x for x in train.columns if x not in [target]+IDcol]
    #print(predictors)

    dataset = train[predictors]
    print(dataset.shape)
    train_dataset = dataset[:5000]
    test_dataset = dataset[5000:]
    print(train_dataset.shape)
    print(test_dataset.shape)

    target_data = train[target]
    print(target_data.shape)
    train_labels = target_data[:5000]
    test_labels = target_data[5000:]
    print(train_labels.shape)
    print(test_labels.shape)

    a = train_item_identifier.keys()
    b = list(a)
    item_identifier = b[5000:]
    print(len(item_identifier))

    random_forest = RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=4)
    random_forest.fit(train_dataset, train_labels)
    predictions = random_forest.predict(test_dataset)
    score = random_forest.score(train_dataset, train_labels)
    print('Model Score : ', score)

    print(predictions)
    print(predictions.shape)

    #cross-validation (cv):
    cv_score = cross_val_score(random_forest, train_dataset, train_labels, cv=20)
    #print(cv_score)

    mean_results = {}
    std_results = {}

    lr_mean = np.mean(cv_score)
    lr_std = np.std(cv_score)
    mean_results['Random Forest'] = lr_mean
    std_results['Random Forest'] = lr_std

    print('Mean')
    print(mean_results)

    print('Standard Deviation')
    print(std_results)

    #Plot Actuall vs. Prediction
    plt.title('Actual vs. Predicted')
    plt.plot(item_identifier, test_labels)
    plt.plot(item_identifier, predictions)
    plt.xlabel('Item Identifier')
    plt.ylabel('Item Outlet Sales')
    plt.grid(True)
    plt.savefig('plot/RandomForest/plot1.pdf')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    #Plot True Values vs. Predictions
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, predictions)
    plt.xlabel('Actual Item Outlet Sales')
    plt.ylabel('Predicted Item Outlet Sales')
    lims = [-1000,10000]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.grid(True)
    plt.plot(lims, lims)
    plt.savefig('plot/RandomForest/plot1.pdf')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    #Plot Prediction Error vs.Count
    error = predictions - test_labels
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error [Item Outlet Sales]")
    _ = plt.ylabel("Count")
    plt.savefig('plot/RandomForest/plot1.pdf')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    unknow_data = test[predictors]
    unknow_predictions = random_forest.predict(unknow_data)
    print(unknow_predictions)
    print(len(unknow_predictions))

    test_a = test_item_identifier.keys()
    test_b = list(test_a)
    un_item_identifier = test_b
    print(len(un_item_identifier))

    plt.plot(un_item_identifier, unknow_predictions)
    plt.xlabel('Item Identifier')
    plt.ylabel('Item Outlet Sales')
    plt.grid(True)
    plt.savefig('plot/RandomForest/plot1.pdf')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    return(mean_results, std_results)