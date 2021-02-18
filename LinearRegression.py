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


train = pd.read_csv('dataset/train_modified.csv')
test = pd.read_csv('dataset/test_modified.csv')

#print(train.shape)
#print(test.shape)

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

linearRegression = LinearRegression(normalize=True)
linearRegression.fit(train_dataset, train_labels)
predictions = linearRegression.predict(test_dataset)
score = linearRegression.score(train_dataset, train_labels)
print('Model Score : ', score)

print(predictions)
print(predictions.shape)

#cross-validation (cv):
cv_score = cross_val_score(linearRegression, train_dataset, train_labels, cv=20)
#print(cv_score)

mean_results = {}
std_results = {}

lr_mean = np.mean(cv_score)
lr_std = np.std(cv_score)
mean_results['Linear Regression'] = lr_mean
std_results['Linear Regression'] = lr_std

print('Mean')
print(mean_results)
with open('dataset/mean_LinearRegression.json', 'w') as i:
    json.dump(mean_results, i)

print('Standard Deviation')
print(std_results)
with open('dataset/standard_deviation_LinearRegression.json', 'w') as j:
    json.dump(std_results, j)

#Plot Actuall vs. Prediction
plt.title('Actual vs. Predicted')
plt.plot(item_identifier, test_labels)
plt.plot(item_identifier, predictions)
plt.xlabel('Item Identifier')
plt.ylabel('Item Outlet Sales')
plt.grid(True)
plt.show()

'''#Plot True Values vs. Predictions
a = plt.axes(aspect='equal')
plt.scatter(test_labels, predictions)
plt.xlabel('Actual Item Outlet Sales')
plt.ylabel('Predicted Item Outlet Sales')
lims = [-1000,10000]
plt.xlim(lims)
plt.ylim(lims)
plt.grid(True)
plt.plot(lims, lims)
plt.show()

#Plot Prediction Error vs.Count
error = predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [Item Outlet Sales]")
_ = plt.ylabel("Count")
plt.show()'''

unknow_data = test[predictors]
unknow_predictions = linearRegression.predict(unknow_data)
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
plt.show()