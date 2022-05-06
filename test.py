import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

#region Utils
from SoftSplitDecisionTrees import SoftSplitDecisionTreeClassifier
from SoftSplitOptimization import SoftSplitOptimizationDecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler


def preprocess(dataset):
    le = preprocessing.LabelEncoder()
    for column_name in dataset.columns:
        if dataset[column_name].dtype == object:
            dataset[column_name] = le.fit_transform(dataset[column_name])
        else:
            pass
    dataset.fillna(dataset.mean(), inplace=True)
    return dataset

def evaluateModel(model, X, y, k=5, repeats=2):
    rkfcv = RepeatedKFold(n_splits=k, n_repeats=repeats, random_state=1)
    return cross_validate(estimator=model, scoring=['accuracy', 'roc_auc'], X=X, y=y, cv=rkfcv, n_jobs=-1)

def evaluateModel2(model, X, y, k=5, repeats=2):
    rkfcv = RepeatedKFold(n_splits=k, n_repeats=repeats, random_state=1)
    return cross_validate(estimator=model, scoring=['neg_mean_squared_error'], X=X, y=y, cv=rkfcv, n_jobs=-1)

def plotModelScore(scoresRegular, scoresSoftSplit,dataset_name ,title, metric):
    colors = sns.color_palette("Paired")
    plt.plot(scoresRegular[metric], label=f'regular classifier {title}', color=colors[0])
    plt.plot([scoresRegular[metric].mean() for x in scoresRegular[metric]], label=f'regular classifier {title} mean',
             color=colors[1], linewidth=0.5, marker="_")
    plt.plot(scoresSoftSplit[metric], label=f'soft split classifier {title}', color=colors[2])
    plt.plot([scoresSoftSplit[metric].mean() for x in scoresSoftSplit[metric]], label=f'soft split classifier {title} mean',
             color=colors[3], linewidth=0.5, marker="_")
    plt.legend()
    plt.title(f'{title.capitalize()} of models on {dataset_name.capitalize()}')
    plt.xlabel('Iteration')
    plt.ylabel(f'{title.capitalize()}')
    plt.show()
#endregion

#region Healthcare
# strokeDataset = pd.read_csv('datasets/classification/healthcare-dataset-stroke-data.csv')
#
# strokeDataset=strokeDataset[(strokeDataset['bmi'] > strokeDataset['bmi'].quantile(0.01)) & (strokeDataset['bmi'] < strokeDataset['bmi'].quantile(0.99))]
# strokeDataset = strokeDataset[strokeDataset['gender']!='Other']
# strokeDataset=pd.get_dummies(strokeDataset,columns=['gender','work_type'],drop_first=False)
# strokeDataset['ever_married'].replace({'Yes':1,'No':0},inplace=True)
# strokeDataset['Residence_type'].replace({'Urban':1,'Rural':0},inplace=True)
# strokeDataset['smoking_status'].replace({'never smoked':0,'formerly smoked':1,'smokes':2,'Unknown':3},inplace=True)
# strokeDataset.fillna(strokeDataset.mean(), inplace=True)
#
# X,y = strokeDataset.loc[:, strokeDataset.columns!='stroke'],strokeDataset['stroke']
# scaler = MinMaxScaler()
# # To scale data
# X= scaler.fit_transform(X)
#
# softsplitOptimizedClassifier = SoftSplitOptimizationDecisionTreeClassifier()
# treeSoftSplitClassifier = SoftSplitDecisionTreeClassifier(n=100,alphaProbability=0.1)
#
# scoresRegular =evaluateModel(softsplitOptimizedClassifier,X,y)
# scoresSoftSplit =evaluateModel(treeSoftSplitClassifier,X,y)
# plotModelScore(scoresRegular, scoresSoftSplit,'Stroke dataset','accuracy','test_accuracy')
# plotModelScore(scoresRegular, scoresSoftSplit,'Stroke dataset','auc','test_roc_auc')
#endregion

#region Water Quality

# waterQualityDataset = pd.read_csv('datasets/classification/waterQuality.csv')
# waterQualityDataset=preprocess(waterQualityDataset)
#
# X,y = waterQualityDataset.loc[:, waterQualityDataset.columns!='Potability'],waterQualityDataset['Potability']
# # Initialise the Scaler
# scaler = MinMaxScaler()
# # To scale data
# X= scaler.fit_transform(X)
#
# softsplitOptimizedClassifier = SoftSplitOptimizationDecisionTreeClassifier()
# treeSoftSplitClassifier = SoftSplitDecisionTreeClassifier(n=100,alphaProbability=0.1)
#
# scoresRegular =evaluateModel(softsplitOptimizedClassifier,X,y)
# scoresSoftSplit =evaluateModel(treeSoftSplitClassifier,X,y)
# plotModelScore(scoresRegular, scoresSoftSplit,'Water Quality dataset','accuracy','test_accuracy')
# plotModelScore(scoresRegular, scoresSoftSplit,'Water Quality dataset','auc','test_roc_auc')
#endregion

#region Crystals
# crystalDataset = pd.read_csv('datasets/classification/crystal.csv')
# # We will convert this multi-class problem to binary problem. We will try to classify a samples to 'Cubic' or 'No cubic'
# # By converting this to binary classification problem get better balanced dataset
# # and also we could measure our model performance using the AUC metric
# # which can be apply only on binary classification problems.
# crystalDataset=crystalDataset[(crystalDataset['τ'] > crystalDataset['τ'].quantile(0.15)) & (crystalDataset['τ'] < crystalDataset['τ'].quantile(0.95))]
# crystalDataset['Lowest distortion'].mask(crystalDataset['Lowest distortion'] != 'cubic', 'no cubic', inplace=True)
# X,y = crystalDataset.loc[:, crystalDataset.columns!='Lowest distortion'],crystalDataset['Lowest distortion']
# # Initialise the Scaler
# scaler = MinMaxScaler()
# # To scale data
# X= scaler.fit_transform(X)
# softsplitOptimizedClassifier = SoftSplitOptimizationDecisionTreeClassifier(alphaProbability=0.1,nearSensitivity=10**7)
# treeSoftSplitClassifier = SoftSplitDecisionTreeClassifier(n=100,alphaProbability=0.1)
#
# scoresRegular =evaluateModel(softsplitOptimizedClassifier,X,y)
# scoresSoftSplit =evaluateModel(treeSoftSplitClassifier,X,y)
# plotModelScore(scoresRegular, scoresSoftSplit,'Crystal dataset','accuracy','test_accuracy')
# plotModelScore(scoresRegular, scoresSoftSplit,'Crystal dataset','auc','test_roc_auc')
#endregion

#region Wine Qaulity
# wineQualityDataset = pd.read_csv('datasets/classification/wineQuality.csv')
# # We will convert this multi-class problem to binary problem. We will try to classify a samples to 'Low' or 'High'
# # By converting this to binary classification problem get better balanced dataset
# # and also we could measure our model performance using the AUC metric
# wineQualityDataset=wineQualityDataset[wineQualityDataset['chlorides'] < wineQualityDataset['chlorides'].quantile(0.95)]
# wineQualityDataset=wineQualityDataset[wineQualityDataset['free sulfur dioxide'] < wineQualityDataset['free sulfur dioxide'].quantile(0.95)]
# wineQualityDataset=wineQualityDataset[wineQualityDataset['total sulfur dioxide'] < wineQualityDataset['total sulfur dioxide'].quantile(0.95)]
# wineQualityDataset=wineQualityDataset[wineQualityDataset['residual sugar'] < wineQualityDataset['residual sugar'].quantile(0.95)]
# wineQualityDataset=wineQualityDataset[wineQualityDataset['sulphates'] < wineQualityDataset['sulphates'].quantile(0.95)]
# wineQualityDataset['quality'].mask(wineQualityDataset['quality'] <= 5, 0, inplace=True)
# wineQualityDataset['quality'].mask(wineQualityDataset['quality'] > 5, 1, inplace=True)
# wineQualityDataset['quality'].mask(wineQualityDataset['quality'] == 0, 'Low', inplace=True)
# wineQualityDataset['quality'].mask(wineQualityDataset['quality'] == 1, 'High', inplace=True)
#
# X,y = wineQualityDataset.loc[:, wineQualityDataset.columns!='quality'],wineQualityDataset['quality']
# # Initialise the Scaler
# scaler = MinMaxScaler()
# # To scale data
# X= scaler.fit_transform(X)
# softsplitOptimizedClassifier = SoftSplitOptimizationDecisionTreeClassifier(alphaProbability=0.1,nearSensitivity=1399)
# treeSoftSplitClassifier = SoftSplitDecisionTreeClassifier(n=100,alphaProbability=0.1)
#
# scoresRegular =evaluateModel(softsplitOptimizedClassifier,X,y)
# scoresSoftSplit =evaluateModel(treeSoftSplitClassifier,X,y)
# plotModelScore(scoresRegular, scoresSoftSplit,'Wine Quality dataset','accuracy','test_accuracy')
# plotModelScore(scoresRegular, scoresSoftSplit,'Wine Quality dataset','auc','test_roc_auc')
#endregion

#region Adult

adultIncomeDataset = pd.read_csv('datasets/classification/adult.csv')
adultIncomeDataset.drop(columns='education',inplace=True)

adultIncomeDataset=pd.get_dummies(adultIncomeDataset,columns=['race','marital-status','relationship','native-country','gender','workclass','occupation'],drop_first=False)

X,y = adultIncomeDataset.loc[:, adultIncomeDataset.columns!='income'],adultIncomeDataset['income']
# Initialise the Scaler
scaler = MinMaxScaler()
# To scale data
X= scaler.fit_transform(X)
softsplitOptimizedClassifier = SoftSplitOptimizationDecisionTreeClassifier(alphaProbability=0.1,nearSensitivity=1399)
treeSoftSplitClassifier = SoftSplitDecisionTreeClassifier(n=100,alphaProbability=0.1)

scoresRegular =evaluateModel(softsplitOptimizedClassifier,X,y)
scoresSoftSplit =evaluateModel(treeSoftSplitClassifier,X,y)
plotModelScore(scoresRegular, scoresSoftSplit,'Adult Income dataset','accuracy','test_accuracy')
plotModelScore(scoresRegular, scoresSoftSplit,'Adult Income dataset','auc','test_roc_auc')
#endregion