import json 
import matplotlib.pyplot as plt


with open('dataset/mean_LinearRegression.json') as i:
    lr_mean = json.load(i)
with open('dataset/mean_RandomForest.json') as j:
    rf_mean= json.load(j)
with open('dataset/mean_RidgeRegression.json') as i:
    rr_mean = json.load(i)
with open('dataset/mean_decisionTree.json') as j:
    dt_mean = json.load(j)

mean_result = {}
mean_result.update(lr_mean)
mean_result.update(rf_mean)
mean_result.update(rr_mean)
mean_result.update(dt_mean)

with open('dataset/standard_deviation_LinearRegression.json') as i:
    lr_std = json.load(i)
with open('dataset/standard_deviation_RandomForest.json') as j:
    rf_std= json.load(j)
with open('dataset/standard_deviation_RidgeRegression.json') as i:
    rr_std = json.load(i)
with open('dataset/standard_deviation_decisionTree.json') as j:
    dt_std = json.load(j)

standard_deviation_result = {}
standard_deviation_result.update(lr_std)
standard_deviation_result.update(rf_std)
standard_deviation_result.update(rr_std)
standard_deviation_result.update(dt_std)


#print(mean_result)
#print(standard_deviation_result)


print('\nMean :')
print(mean_result.values())
print('\nStandard Deviation :')
print(standard_deviation_result.values())


plt.figure(figsize=(15,8))
plt.bar(mean_result.keys(), mean_result.values())
plt.xlabel('Model Name')
plt.ylabel('Model Mean Values')
plt.grid(True)
plt.show()


plt.figure(figsize=(15,8))
plt.bar(standard_deviation_result.keys(), standard_deviation_result.values())
plt.xlabel('Model Name')
plt.ylabel('Model Standard Deviation Values')
plt.grid(True)
plt.show()

plt.figure(figsize=(15,8))
plt.bar(mean_result.keys(), mean_result.values())
plt.bar(mean_result.keys(), standard_deviation_result.values())
plt.xlabel('Model Name')
plt.ylabel('Model Standard Deviation Values')
plt.grid(True)
plt.show()