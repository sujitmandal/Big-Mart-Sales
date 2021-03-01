import os
import json
import pandas as pd
from plot_model import ploModel
from model.RandomForest import RandomForest
from model.DecisionTree import DecisionTree
from model.RdgeRegression import RidgeRegression
from DataPreProcessing import FeatureEngineering
from model.LinearRegression import linearRegression

# Github: https://github.com/sujitmandal
# Pypi : https://pypi.org/user/sujitmandal/
# LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/ 

if os.path.exists('dataset'):
    train = pd.read_csv('dataset/Train.csv')
    test = pd.read_csv('dataset/Test.csv')
else:
    print('\nDataset Not Exists!')

if not os.path.exists('dataset/inputTrain.csv') and not os.path.exists('dataset/inputTest.csv') == True:
    FeatureEngineering(train, test)

if not os.path.exists('dataset/modelPerformance.json'):
    inputTrain = pd.read_csv('dataset/inputTrain.csv')
    inputTest = pd.read_csv('dataset/inputTest.csv')


    RFmean, RFstd = RandomForest(inputTrain, inputTest)
    DTmean, DTstd = DecisionTree(inputTrain, inputTest)
    RRmean, RRstd = RidgeRegression(inputTrain, inputTest)
    LLmean, LLstd = linearRegression(inputTrain, inputTest)

    meanResult = {}
    stdResult = {}
    meanResult['mean'] = RFmean
    meanResult.get('mean').update(DTmean)
    meanResult.get('mean').update(RRmean)
    meanResult.get('mean').update(LLmean)

    stdResult['std'] = RFstd
    stdResult.get('std').update(DTstd)
    stdResult.get('std').update(RRstd)
    stdResult.get('std').update(LLstd)

    modelPerformance = {}
    modelPerformance.update(meanResult)
    modelPerformance.update(stdResult)

    print('Model Performance : ')
    print(json.dumps(modelPerformance, indent=3))

    with open('dataset/modelPerformance.json', 'w') as i:
        json.dump(modelPerformance, i)

    print('modelPerformance.json is created on {}'.format(os.getcwd()))
    ploModel(modelPerformance)

else:
    with open('dataset/modelPerformance.json') as i:
        modelPerformance = json.load(i)

    print('Model is already train and the model data is exist on {}'.format(os.getcwd()))
    print(json.dumps(modelPerformance, indent=3))

    ploModel(modelPerformance)