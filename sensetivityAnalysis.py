from random import random

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold, cross_validate

from SoftSplitDecisionTrees import SoftSplitDecisionTreeClassifier

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
        model = SoftSplitDecisionTreeClassifier(n=n, alphaProbability=alphaProbability)
        rkfcv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)
        scores = cross_validate(estimator=model, scoring=['accuracy', 'roc_auc'], X=X, y=y, cv=rkfcv, n_jobs=-1)
        return scores['test_accuracy'].mean()

    return getScore


Datasets = []
# region Stroke dataset
strokeDataset = pd.read_csv('datasets/classification/healthcare-dataset-stroke-data.csv')
strokeDataset = strokeDataset[(strokeDataset['bmi'] > strokeDataset['bmi'].quantile(0.01)) & (
        strokeDataset['bmi'] < strokeDataset['bmi'].quantile(0.99))]
strokeDataset = strokeDataset[strokeDataset['gender'] != 'Other']
strokeDataset = pd.get_dummies(strokeDataset, columns=['gender', 'work_type'], drop_first=False)
strokeDataset['ever_married'].replace({'Yes': 1, 'No': 0}, inplace=True)
strokeDataset['Residence_type'].replace({'Urban': 1, 'Rural': 0}, inplace=True)
strokeDataset['smoking_status'].replace({'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3},
                                        inplace=True)
strokeDataset.fillna(strokeDataset.mean(), inplace=True)
X, y = strokeDataset.loc[:, strokeDataset.columns != 'stroke'], strokeDataset['stroke']
# # endregion
Datasets.append(('Stroke dataset',X,y))

# region Water Quality dataset
waterQualityDataset = pd.read_csv('datasets/classification/waterQuality.csv')
waterQualityDataset=preprocess(waterQualityDataset)
X,y = waterQualityDataset.loc[:, waterQualityDataset.columns!='Potability'],waterQualityDataset['Potability']
#endregion
Datasets.append(('Water Quality dataset',X,y))

#region Crystal dataset
crystalDataset = pd.read_csv('datasets/classification/crystal.csv')
crystalDataset['Lowest distortion'].mask(crystalDataset['Lowest distortion'] != 'cubic', 'no cubic', inplace=True)
crystalDataset=crystalDataset[(crystalDataset['τ'] > crystalDataset['τ'].quantile(0.01)) & (crystalDataset['τ'] < crystalDataset['τ'].quantile(0.99))]
crystalDataset=preprocess(crystalDataset)
X,y = crystalDataset.loc[:, crystalDataset.columns!='Lowest distortion'],crystalDataset['Lowest distortion']
#endregion
Datasets.append(('Crystals dataset',X,y))

points = []
evalFunction = evaluate(X, y)
for n in range(1, 200, 20):
    for alpha in np.linspace(start=0, stop=1, num=20):
         # points.append((n, alpha, evalFunction(n, alpha)))
         points.append((n, alpha, 1))


N, Alpha, Accuracy = zip(*points)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(N, Alpha, Accuracy, c=Accuracy, cmap='Greens')
ax.set_xlabel('n')
ax.set_ylabel('alpha')
ax.set_zlabel('accuracy')
ax.set_title('title')
plt.savefig('books_read.png')
