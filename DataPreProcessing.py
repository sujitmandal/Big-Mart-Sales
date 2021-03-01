import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder

def FeatureEngineering(train, test):
    #Combine test and train into one file
    train['source']='train'
    test['source']='test'
    data = pd.concat([train, test],ignore_index=True)
    print('\nTrain Data Shape : {}'.format(train.shape))
    print('\nTest Data Shape : {}'.format(test.shape))
    print('\nTotal Data Shape : {}'.format(data.shape))

    #Check missing values:
    check_missing_values = data.apply(lambda x: sum(x.isnull()))
    print('\nCheck missing values : ')
    print(check_missing_values)

    #Numerical data summary:
    numerical_data_summary = data.describe()
    print('\nNumerical data summary : ')
    print(numerical_data_summary)

    #Number of unique values in each:
    find_unique_values = data.apply(lambda x: len(x.unique()))
    print('\nUnique values :')
    print(find_unique_values)


    #Filter categorical variables
    categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
    #print(categorical_columns)
    #Exclude ID cols and source:
    categorical_column = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
    #print(categorical_column)

    #Print frequency of categories
    for column in categorical_column:
        print('\nFrequency of Categories for varible %s'%column)
        print(data[column].value_counts())

    #Data Cleaning
    #Determine the average weight per item:
    item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')
    #Get a boolean variable specifying missing Item_Weight values
    miss_bool = data['Item_Weight'].isnull() 

    #Impute data and check #missing values before and after imputation to confirm
    print('\nOrignal #missing: %d'% sum(miss_bool))

    data.loc[miss_bool,'Item_Weight'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight.mean())
    print('\nFinal #missing: %d'% sum(data['Item_Weight'].isnull()))

    print('\nOrignal #missing: %d'% sum(miss_bool))
    print('\nFinal #missing: %d'% sum(data['Item_Weight'].isnull()))
    #data.to_csv('dataset/t.csv')

    #Determing the mode for each
    outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x:mode(x).mode[0]) )
    print('\nMode for each Outlet_Type:')
    print(outlet_size_mode)

    #Get a boolean variable specifying missing Item_Weight values
    miss_bool = data['Outlet_Size'].isnull() 
    #Impute data and check #missing values before and after imputation to confirm
    print('\nOrignal #missing: %d'% sum(miss_bool))

    #Impute data and check #missing values before and after imputation to confirm
    print('\nOrignal #missing: %d'% sum(miss_bool))
    data.loc[miss_bool,'Outlet_Size'] = data.loc[miss_bool,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
    print('\nFinal #missing: %d'% sum(data['Outlet_Size'].isnull()))

    #Feature Engineering:
    #Step1: Consider combining categories in Outlet_Type

    #Check the mean sales by type:
    data.pivot_table(values='Item_Outlet_Sales',index='Outlet_Type')

    #Step2: Modify Item_Visibility
    #Determine average visibility of a product
    visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')

    #Impute 0 values with mean visibility of that product:
    miss_bool = (data['Item_Visibility'] == 0)
    print('\nNumber of 0 values initially: %d'%sum(miss_bool))

    #Step2: Modify Item_Visibility
    #Determine average visibility of a product
    visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')
    #Impute 0 values with mean visibility of that product:
    miss_bool = (data['Item_Visibility'] == 0)
    print('\nNumber of 0 values initially: %d'%sum(miss_bool))

    data.loc[miss_bool,'Item_Visibility'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: visibility_avg.mean())
    print('\nNumber of 0 values after modification: %d'%sum(data['Item_Visibility'] == 0))

    print('\nNumber of 0 values initially: %d'%sum(miss_bool))
    print('\nNumber of 0 values after modification: %d'%sum(data['Item_Visibility'] == 0))

    #data.to_csv('dataset/tt.csv')

    Item_Visibility_MeanRatio =  data.apply(lambda x: x['Item_Visibility']/visibility_avg['Item_Visibility'], axis=1)
    print('\nItem Visibility MeanRatio')
    print(Item_Visibility_MeanRatio[:3])

    item_visibility_statistics = Item_Visibility_MeanRatio.describe()
    item_visibility_statistics = item_visibility_statistics.transpose()[['mean', 'std', 'min', 'max']]
    print('\nitem visibility mean, std, min, max, values :')
    print(item_visibility_statistics)

    #Step 3: Create a broad category of Type of Item

    #Item type combine:
    data['Item_Identifier'].value_counts()
    data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
    data['Item_Type_Combined'] = data['Item_Type_Combined'].map({
                                                                'FD':'Food',
                                                                'NC':'Non-Consumable',
                                                                'DR':'Drinks'})

    print('\nItem_Type_Combined')
    print(data['Item_Type_Combined'].value_counts())

    #Step 4: Determine the years of operation of a store
    #Years:
    data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
    data['Outlet_Years'].describe().transpose()[['mean', 'std', 'min', 'max']]

    #Step 5: Modify categories of Item_Fat_Content
    #Change categories of low fat:
    print('\nOriginal Categories:')
    print(data['Item_Fat_Content'].value_counts())

    print('\nModified Categories:')
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                                'reg':'Regular',
                                                                'low fat':'Low Fat'})
    print(data['Item_Fat_Content'].value_counts())


    #Mark non-consumables as separate category in low_fat:
    data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
    data['Item_Fat_Content'].value_counts()

    #Step 6: Numerical and One-Hot Coding of Categorical variables
    #Import library:
    le = LabelEncoder()
    #New variable for outlet
    data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
    var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
    le = LabelEncoder()
    for i in var_mod:
        data[i] = le.fit_transform(data[i])

    #One Hot Coding:
    data = pd.get_dummies(data, columns=[
                                        'Item_Fat_Content',
                                        'Outlet_Location_Type',
                                        'Outlet_Size',
                                        'Outlet_Type',
                                        'Item_Type_Combined',
                                        'Outlet'])

    print('\n pre_process data')
    print(data.dtypes)
    print(data[['Item_Fat_Content_0','Item_Fat_Content_1','Item_Fat_Content_2']].head(10))
    #data.to_csv('dataset/pre_pro_data.csv')


    #Drop the columns which have been converted to different types:
    data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

    #Divide into test and train:
    train = data.loc[data['source']=="train"]
    test = data.loc[data['source']=="test"]

    #Drop unnecessary columns:
    test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
    train.drop(['source'],axis=1,inplace=True)

    #Export files as modified versions:
    train.to_csv('dataset/inputTrain.csv',index=False)
    test.to_csv('dataset/inputTest.csv',index=False)