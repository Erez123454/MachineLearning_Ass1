import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold, cross_validate
from SoftSplitDecisionTrees import SoftSplitDecisionTreeRegressor
import seaborn as sns

def preprocess(dataset):
    le = preprocessing.OneHotEncoder()
    for column_name in dataset.columns:
        if dataset[column_name].dtype == object:
            dataset[column_name] = le.fit_transform(dataset[column_name])
        else:
            pass
    dataset.fillna(dataset.mean(), inplace=True)
    return dataset

def evaluate(X, y):
    def getScore(n, alphaProbability):
        # Change model type for regression
        model = SoftSplitDecisionTreeRegressor(n=n, alphaProbability=alphaProbability)
        rkfcv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)
        scores = cross_validate(estimator=model, scoring=['neg_mean_squared_error'], X=X, y=y, cv=rkfcv, n_jobs=-1)
        return scores['test_neg_mean_squared_error'].mean(), scores['test_neg_mean_squared_error'].mean()
    return getScore

# At the following section paste all the preprocessing which you have made in the Jupyter notebook and append it via a tuple
Datasets = []
#wine alcohol dataset
wineAlcoholDataSet = pd.read_csv('datasets/regression/wineQuality.csv')
# endregion
Datasets.append(('Wine Alcohol dataset',wineAlcoholDataSet.loc[:, (wineAlcoholDataSet.columns!='id')&(wineAlcoholDataSet.columns!='alcohol')],wineAlcoholDataSet['alcohol']))

# Life expectancy dataset
dataLifeExpectancyDataSet = pd.read_csv('datasets/regression/data_life expectancy.csv')

dataLifeExpectancyDataSet['Status'].replace({'Developing':0,'Developed':1},inplace=True)
dataLifeExpectancyDataSet.fillna(dataLifeExpectancyDataSet.mean(), inplace=True)
#endregion
Datasets.append(('Life expectancy dataset',dataLifeExpectancyDataSet.loc[:, (dataLifeExpectancyDataSet.columns!='Country')&(dataLifeExpectancyDataSet.columns!='Life expectancy ')],dataLifeExpectancyDataSet['Life expectancy ']))

#Stroke glucose dataset
strokeDataset = pd.read_csv('datasets/regression/healthcare-dataset-stroke-data.csv')

strokeDataset=strokeDataset[(strokeDataset['bmi'] > strokeDataset['bmi'].quantile(0.01)) & (strokeDataset['bmi'] < strokeDataset['bmi'].quantile(0.99))]
strokeDataset = strokeDataset[strokeDataset['gender']!='Other']
strokeDataset=pd.get_dummies(strokeDataset,columns=['gender','work_type'],drop_first=False)
strokeDataset['ever_married'].replace({'Yes':1,'No':0},inplace=True)
strokeDataset['Residence_type'].replace({'Urban':1,'Rural':0},inplace=True)
strokeDataset['smoking_status'].replace({'never smoked':0,'formerly smoked':1,'smokes':2,'Unknown':3},inplace=True)
strokeDataset.fillna(strokeDataset.mean(), inplace=True)

#endregion
Datasets.append(('Stroke glucose dataset',strokeDataset.loc[:, strokeDataset.columns!='avg_glucose_level'],strokeDataset['avg_glucose_level']))

#House price dataset
housePriceDataSet = pd.read_csv('datasets/regression/house_data_price.csv')
housePriceDataSet.fillna(housePriceDataSet.mean(), inplace=True)

#endregion
Datasets.append(('House Price dataset',housePriceDataSet.loc[:, (housePriceDataSet.columns!='price')&(housePriceDataSet.columns!='date')],housePriceDataSet['price']))

#water hardness dataset
waterHardnessDataSet = pd.read_csv('datasets/regression/waterQuality.csv')
waterHardnessDataSet.fillna(waterHardnessDataSet.mean(), inplace=True)
# endregion
Datasets.append(('Water Hardness dataset',waterHardnessDataSet.loc[:, waterHardnessDataSet.columns!='Hardness'],waterHardnessDataSet['Hardness']))

#region Alpha graphs
MSE= {}

for dataset in Datasets:
    name, X, y = dataset
    mseList=[]
    for alpha in np.arange(0.1, 1, 0.05):
        n = 100
        evalFunction = evaluate(X, y)
        print(f'dataset:{name} n:{n} alpha:{alpha}')
        mse = evalFunction(n, alpha)
        mseList.append((alpha,mse))
    MSE[name]=mseList


colors = sns.color_palette(n_colors=5)

for index,dataset in enumerate(Datasets):
    name, X, y = dataset
    xs,ys = zip(*MSE[name])
    plt.plot(xs,ys,label=f'{name} MSE', color=colors[index], marker="o")

plt.legend()
plt.title(f'Sensitivity Analysis - MSE by Alpha [n=100]')
plt.xlabel('Alpha')
plt.xticks(np.arange(0, 1, 0.05))
plt.ylabel(f'MSE')
plt.savefig(f'Sensitivity Analysis - MSE by Alpha.png')
#endregion

#region N graphs
MSE= {}

for dataset in Datasets:
    name, X, y = dataset
    mseList=[]
    for n in range(1,100,10):
        alpha = 0.1
        evalFunction = evaluate(X, y)
        print(f'dataset:{name} n:{n} alpha:{alpha}')
        mse = evalFunction(n, alpha)
        mseList.append((n,mse))
    MSE[name]=mseList


for index,dataset in enumerate(Datasets):
    name, X, y = dataset
    xs,ys = zip(*MSE[name])
    plt.plot(xs,ys,label=f'{name} MSE', color=colors[index], marker="o")

plt.legend()
plt.title(f'Sensitivity Analysis - MSE by n [alpha=0.1]')
plt.xlabel('n')
plt.xticks(range(1,100,10))
plt.ylabel(f'MSE')
plt.savefig(f'Sensitivity Analysis - MSE by n.png')
#endregion