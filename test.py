import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

strokeDataset = pd.read_csv('datasets/classification/healthcare-dataset-stroke-data.csv')


from sklearn.tree import DecisionTreeClassifier
from SoftSplitDecisionTrees import SoftSplitDecisionTreeClassifier
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn import preprocessing


# Utils
def preprocess(dataset):
    le = preprocessing.LabelEncoder()
    for column_name in dataset.columns:
        if dataset[column_name].dtype == object:
            dataset[column_name] = le.fit_transform(dataset[column_name])
        else:
            pass
    dataset.fillna(strokeDataset.mean(), inplace=True)
    return dataset

def evaluateModel(model,X,y,k=5,repeats=2):
    rkfcv = RepeatedKFold(n_splits=k, n_repeats=repeats, random_state=1)
    return cross_val_score(estimator=model, X=X, y=y, cv=rkfcv)


strokeDataset=preprocess(strokeDataset)
X,y = strokeDataset.loc[:, strokeDataset.columns!='stroke'],strokeDataset['stroke']

treeClassifier = DecisionTreeClassifier()
treeSoftSplitClassifier = SoftSplitDecisionTreeClassifier(n=100,alphaProbability=0.1)

scoresRegular =evaluateModel(treeClassifier,X,y)
scoresSoftSplit =evaluateModel(treeSoftSplitClassifier,X,y)
print(f'Regular Model accuracy {scoresRegular.mean()} SoftSplit Model accuracy {scoresSoftSplit.mean()}')
