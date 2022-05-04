import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold, cross_validate
from SoftSplitDecisionTrees import SoftSplitDecisionTreeClassifier
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
        model = SoftSplitDecisionTreeClassifier(n=n, alphaProbability=alphaProbability)
        rkfcv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)
        scores = cross_validate(estimator=model, scoring=['accuracy', 'roc_auc'], X=X, y=y, cv=rkfcv, n_jobs=-1)
        return scores['test_accuracy'].mean(), scores['test_roc_auc'].mean()
    return getScore

# At the following section paste all the preprocessing which you have made in the Jupyter notebook and append it via a tuple
Datasets = []
#region Stroke dataset
strokeDataset = pd.read_csv('datasets/classification/healthcare-dataset-stroke-data.csv')
strokeDataset=strokeDataset[(strokeDataset['bmi'] > strokeDataset['bmi'].quantile(0.01)) & (strokeDataset['bmi'] < strokeDataset['bmi'].quantile(0.99))]
strokeDataset = strokeDataset[strokeDataset['gender']!='Other']
strokeDataset=pd.get_dummies(strokeDataset,columns=['gender','work_type'],drop_first=False)
strokeDataset['ever_married'].replace({'Yes':1,'No':0},inplace=True)
strokeDataset['Residence_type'].replace({'Urban':1,'Rural':0},inplace=True)
strokeDataset['smoking_status'].replace({'never smoked':0,'formerly smoked':1,'smokes':2,'Unknown':3},inplace=True)
strokeDataset.fillna(strokeDataset.mean(), inplace=True)
# endregion
Datasets.append(('Stroke dataset',strokeDataset.loc[:, strokeDataset.columns != 'stroke'],strokeDataset['stroke']))

# region Water Quality dataset
waterQualityDataset = pd.read_csv('datasets/classification/waterQuality.csv')
waterQualityDataset=preprocess(waterQualityDataset)
#endregion
Datasets.append(('Water Quality dataset',waterQualityDataset.loc[:, waterQualityDataset.columns!='Potability'],waterQualityDataset['Potability']))

#region Crystal dataset
crystalDataset = pd.read_csv('datasets/classification/crystal.csv')
crystalDataset=crystalDataset[(crystalDataset['τ'] > crystalDataset['τ'].quantile(0.15)) & (crystalDataset['τ'] < crystalDataset['τ'].quantile(0.95))]
crystalDataset['Lowest distortion'].mask(crystalDataset['Lowest distortion'] != 'cubic', 'no cubic', inplace=True)
#endregion
Datasets.append(('Crystals dataset',crystalDataset.loc[:, crystalDataset.columns!='Lowest distortion'],crystalDataset['Lowest distortion']))

#region Wine Quality dataset
wineQualityDataset = pd.read_csv('datasets/classification/wineQuality.csv')
wineQualityDataset=wineQualityDataset[wineQualityDataset['chlorides'] < wineQualityDataset['chlorides'].quantile(0.95)]
wineQualityDataset=wineQualityDataset[wineQualityDataset['free sulfur dioxide'] < wineQualityDataset['free sulfur dioxide'].quantile(0.95)]
wineQualityDataset=wineQualityDataset[wineQualityDataset['total sulfur dioxide'] < wineQualityDataset['total sulfur dioxide'].quantile(0.95)]
wineQualityDataset=wineQualityDataset[wineQualityDataset['residual sugar'] < wineQualityDataset['residual sugar'].quantile(0.95)]
wineQualityDataset=wineQualityDataset[wineQualityDataset['sulphates'] < wineQualityDataset['sulphates'].quantile(0.95)]
wineQualityDataset['quality'].mask(wineQualityDataset['quality'] <= 5, 0, inplace=True)
wineQualityDataset['quality'].mask(wineQualityDataset['quality'] > 5, 1, inplace=True)
wineQualityDataset['quality'].mask(wineQualityDataset['quality'] == 0, 'Low', inplace=True)
wineQualityDataset['quality'].mask(wineQualityDataset['quality'] == 1, 'High', inplace=True)

#endregion
Datasets.append(('Wine Quality dataset',wineQualityDataset.loc[:, wineQualityDataset.columns!='quality'],wineQualityDataset['quality']))

#region Adults Income dataset
adultIncomeDataset = pd.read_csv('datasets/classification/adult.csv')
adultIncomeDataset.drop(columns='education',inplace=True)
adultIncomeDataset=pd.get_dummies(adultIncomeDataset,columns=['race','marital-status','relationship','native-country','gender','workclass','occupation'],drop_first=False)
# endregion
Datasets.append(('Adults Income dataset',adultIncomeDataset.loc[:, adultIncomeDataset.columns!='income'],adultIncomeDataset['income']))

#region Alpha graphs
Accuracies= {}
Aucs={}

for dataset in Datasets:
    name, X, y = dataset
    accuracyList=[]
    aucList=[]
    for alpha in np.arange(0.1, 1, 0.05):
        n = 100
        evalFunction = evaluate(X, y)
        print(f'dataset:{name} n:{n} alpha:{alpha}')
        acc,auc = evalFunction(n, alpha)
        accuracyList.append((alpha,acc))
        aucList.append((alpha,auc))
    Accuracies[name]=accuracyList
    Aucs[name]=aucList

colors = sns.color_palette(n_colors=5)

for index,dataset in enumerate(Datasets):
    name, X, y = dataset
    xs,ys = zip(*Accuracies[name])
    plt.plot(xs,ys,label=f'{name} accuracy', color=colors[index], marker="o")

plt.legend()
plt.title(f'Sensitivity Analysis - Accuracy by Alpha [n=100]')
plt.xlabel('Alpha')
plt.xticks(np.arange(0, 1, 0.05))
plt.ylabel(f'Accuracy')
plt.savefig(f'Sensitivity Analysis - Accuracy by Alpha.png')

for index,dataset in enumerate(Datasets):
    name, X, y = dataset
    xs, ys = zip(*Aucs[name])
    plt.plot(xs, ys,label=f'{name} auc', color=colors[index],marker="o")

plt.legend()
plt.title(f'Sensitivity Analysis - AUC by Alpha [n=100]')
plt.xlabel('Alpha')
plt.xticks(np.arange(0, 1, 0.05))
plt.ylabel(f'AUC')
plt.savefig(f'Sensitivity Analysis - AUC by Alpha.png')

#endregion

#region N graphs
Accuracies= {}
Aucs={}

for dataset in Datasets:
    name, X, y = dataset
    accuracyList=[]
    aucList=[]
    for n in range(1,100,10):
        alpha = 0.1
        evalFunction = evaluate(X, y)
        print(f'dataset:{name} n:{n} alpha:{alpha}')
        acc,auc = evalFunction(n, alpha)
        accuracyList.append((n,acc))
        aucList.append((n,auc))
    Accuracies[name]=accuracyList
    Aucs[name]=aucList


for index,dataset in enumerate(Datasets):
    name, X, y = dataset
    xs,ys = zip(*Accuracies[name])
    plt.plot(xs,ys,label=f'{name} accuracy', color=colors[index], marker="o")

plt.legend()
plt.title(f'Sensitivity Analysis - Accuracy by n [alpha=0.1]')
plt.xlabel('n')
plt.xticks(range(1,100,10))
plt.ylabel(f'Accuracy')
plt.savefig(f'Sensitivity Analysis - Accuracy by n.png')

for index,dataset in enumerate(Datasets):
    name, X, y = dataset
    xs, ys = zip(*Aucs[name])
    plt.plot(xs, ys,label=f'{name} auc', color=colors[index],marker="o")

plt.legend()
plt.title(f'Sensitivity Analysis - AUC by n [alpha=0.1]')
plt.xlabel('n')
plt.xticks(range(1,100,10))
plt.ylabel(f'AUC')
plt.savefig(f'Sensitivity Analysis - AUC by n.png')

#endregion