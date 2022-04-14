import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

strokeDataset = pd.read_csv('datasets/classification/healthcare-dataset-stroke-data.csv')
waterQualityDataset = pd.read_csv('datasets/classification/crystal.csv')

from sklearn.tree import DecisionTreeClassifier
from SoftSplitDecisionTrees import SoftSplitDecisionTreeClassifier
from sklearn.model_selection import RepeatedKFold, cross_validate

from sklearn import preprocessing


# Utils
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


# strokeDataset = preprocess(strokeDataset)
# X, y = strokeDataset.loc[:, strokeDataset.columns != 'stroke'], strokeDataset['stroke']
#
# treeClassifier = DecisionTreeClassifier()
# treeSoftSplitClassifier = SoftSplitDecisionTreeClassifier(n=100, alphaProbability=0.1)
#
# scoresRegular = evaluateModel(treeClassifier, X, y)
# scoresSoftSplit = evaluateModel(treeSoftSplitClassifier, X, y)
# plotModelScore(scoresRegular, scoresSoftSplit,'accuracy','test_accuracy')
# plotModelScore(scoresRegular, scoresSoftSplit,'auc','test_roc_auc')
# print(f'Regular Model accuracy {scoresRegular.mean()} SoftSplit Model accuracy {scoresSoftSplit.mean()}')


# waterQualityDataset = waterQualityDataset[waterQualityDataset['Lowest distortion'] == 'cubic' or waterQualityDataset['Lowest distortion']== 'orthorhombic']
# waterQualityDataset=waterQualityDataset.loc[(waterQualityDataset['Lowest distortion'] == 'cubic') | (waterQualityDataset['Lowest distortion'] == 'orthorhombic')]
waterQualityDataset['Lowest distortion'].mask(waterQualityDataset['Lowest distortion'] != 'cubic', 'no cubic', inplace=True)
waterQualityDataset=preprocess(waterQualityDataset)

X,y = waterQualityDataset.loc[:, waterQualityDataset.columns!='Lowest distortion'],waterQualityDataset['Lowest distortion']

treeClassifier = DecisionTreeClassifier()
treeSoftSplitClassifier = SoftSplitDecisionTreeClassifier(n=100,alphaProbability=0.1)

scoresRegular =evaluateModel(treeClassifier,X,y)
scoresSoftSplit =evaluateModel(treeSoftSplitClassifier,X,y)
plotModelScore(scoresRegular, scoresSoftSplit,'WaterQuality','accuracy','test_accuracy')
plotModelScore(scoresRegular, scoresSoftSplit,'WaterQuality','auc','test_roc_auc')