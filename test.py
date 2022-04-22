import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#
# strokeDataset = pd.read_csv('datasets/classification/healthcare-dataset-stroke-data.csv')
# waterQualityDataset = pd.read_csv('datasets/classification/crystal.csv')
#
# from sklearn.tree import DecisionTreeClassifier
# from SoftSplitDecisionTrees import SoftSplitDecisionTreeClassifier
# from sklearn.model_selection import RepeatedKFold, cross_validate
#
# from sklearn import preprocessing
#
#
# # Utils
# def preprocess(dataset):
#     le = preprocessing.LabelEncoder()
#     for column_name in dataset.columns:
#         if dataset[column_name].dtype == object:
#             dataset[column_name] = le.fit_transform(dataset[column_name])
#         else:
#             pass
#     dataset.fillna(dataset.mean(), inplace=True)
#     return dataset
#
#
# def evaluateModel(model, X, y, k=5, repeats=2):
#     rkfcv = RepeatedKFold(n_splits=k, n_repeats=repeats, random_state=1)
#     return cross_validate(estimator=model, scoring=['accuracy', 'roc_auc'], X=X, y=y, cv=rkfcv, n_jobs=-1)
#
#
# def plotModelScore(scoresRegular, scoresSoftSplit,dataset_name ,title, metric):
#     colors = sns.color_palette("Paired")
#     plt.plot(scoresRegular[metric], label=f'regular classifier {title}', color=colors[0])
#     plt.plot([scoresRegular[metric].mean() for x in scoresRegular[metric]], label=f'regular classifier {title} mean',
#              color=colors[1], linewidth=0.5, marker="_")
#     plt.plot(scoresSoftSplit[metric], label=f'soft split classifier {title}', color=colors[2])
#     plt.plot([scoresSoftSplit[metric].mean() for x in scoresSoftSplit[metric]], label=f'soft split classifier {title} mean',
#              color=colors[3], linewidth=0.5, marker="_")
#     plt.legend()
#     plt.title(f'{title.capitalize()} of models on {dataset_name.capitalize()}')
#     plt.xlabel('Iteration')
#     plt.ylabel(f'{title.capitalize()}')
#     plt.show()
#
#
# # strokeDataset = preprocess(strokeDataset)
# # X, y = strokeDataset.loc[:, strokeDataset.columns != 'stroke'], strokeDataset['stroke']
# #
# # treeClassifier = DecisionTreeClassifier()
# # treeSoftSplitClassifier = SoftSplitDecisionTreeClassifier(n=100, alphaProbability=0.1)
# #
# # scoresRegular = evaluateModel(treeClassifier, X, y)
# # scoresSoftSplit = evaluateModel(treeSoftSplitClassifier, X, y)
# # plotModelScore(scoresRegular, scoresSoftSplit,'accuracy','test_accuracy')
# # plotModelScore(scoresRegular, scoresSoftSplit,'auc','test_roc_auc')
# # print(f'Regular Model accuracy {scoresRegular.mean()} SoftSplit Model accuracy {scoresSoftSplit.mean()}')
#
#
# # waterQualityDataset = waterQualityDataset[waterQualityDataset['Lowest distortion'] == 'cubic' or waterQualityDataset['Lowest distortion']== 'orthorhombic']
# # waterQualityDataset=waterQualityDataset.loc[(waterQualityDataset['Lowest distortion'] == 'cubic') | (waterQualityDataset['Lowest distortion'] == 'orthorhombic')]
# # waterQualityDataset['Lowest distortion'].mask(waterQualityDataset['Lowest distortion'] != 'cubic', 'no cubic', inplace=True)
# # waterQualityDataset=preprocess(waterQualityDataset)
# #
# # X,y = waterQualityDataset.loc[:, waterQualityDataset.columns!='Lowest distortion'],waterQualityDataset['Lowest distortion']
# #
# # treeClassifier = DecisionTreeClassifier()
# # treeSoftSplitClassifier = SoftSplitDecisionTreeClassifier(n=100,alphaProbability=0.1)
# #
# # scoresRegular =evaluateModel(treeClassifier,X,y)
# # scoresSoftSplit =evaluateModel(treeSoftSplitClassifier,X,y)
# # plotModelScore(scoresRegular, scoresSoftSplit,'WaterQuality','accuracy','test_accuracy')
# # plotModelScore(scoresRegular, scoresSoftSplit,'WaterQuality','auc','test_roc_auc')
#
# def plotPieChart(x, labels, title, colors=sns.color_palette("pastel"), autopct='%.0f%%', shadow=True, startangle=90,
#                  explode=(0.1, 0)):
#     plt.title(title.capitalize(), fontsize=16)
#     plt.pie(x=x, labels=labels, colors=colors, autopct=autopct, shadow=shadow, startangle=startangle)
#     plt.legend()
#     plt.show()
#
#
# def plotChartForAllDataset(dataset, excludeList=[]):
#     for column_name in set(dataset.columns).difference(excludeList):
#         if dataset[column_name].dtype == object or (
#                 dataset[column_name].dtype == 'int64' and len(dataset[column_name].unique()) < 5):
#             data = [len(dataset[dataset[column_name] == value]) for value in dataset[column_name].unique()]
#             plotPieChart(x=data, title=column_name, labels=dataset[column_name].unique())
#         else:
#             his = sns.histplot(data=dataset, x=column_name)
#             his.set_ylabel("# of Records")
#             plt.show()
#
# strokeDataset = pd.read_csv('datasets/classification/healthcare-dataset-stroke-data.csv')
# # filter out sample with gender = 'Other'
# strokeDataset=strokeDataset[(strokeDataset['bmi'] > strokeDataset['bmi'].quantile(0.01)) & (strokeDataset['bmi'] < strokeDataset['bmi'].quantile(0.99))]
# strokeDataset = strokeDataset[strokeDataset['gender']!='Other']
# strokeDataset=pd.get_dummies(strokeDataset,columns=['gender','work_type'],drop_first=False)
# strokeDataset['ever_married'].replace({'Yes':1,'No':0},inplace=True)
# strokeDataset['Residence_type'].replace({'Urban':1,'Rural':0},inplace=True)
# strokeDataset['smoking_status'].replace({'never smoked':0,'formerly smoked':1,'smokes':2,'Unknown':3},inplace=True)
# strokeDataset.fillna(strokeDataset.mean(), inplace=True)
#
# plotChartForAllDataset(strokeDataset)
# # q_hi  = df["col"].quantile(0.99)
# # df_filtered = df[(df["col"] < q_hi) & (df["col"] > q_low)]
# print('a')


waterQualityDataset = pd.read_csv('datasets/classification/waterQuality.csv')
# We will convert this multi-class problem to binary problem. We will try to classify a samples to 'Low' or 'High'
# By converting this to binary classification problem get better balanced dataset
# and also we could measure our model performance using the AUC metric
# which can be apply only on binary classification problems.
waterQualityDataset=waterQualityDataset.loc[:, waterQualityDataset.columns!='Id']
waterQualityDataset['quality'].mask(waterQualityDataset['quality'] <= 5, 'Low', inplace=True)
waterQualityDataset['quality'].mask(waterQualityDataset['quality'] > 5, 'High', inplace=True)
waterQualityDataset=waterQualityDataset[(waterQualityDataset['chlorides'] > waterQualityDataset['chlorides'].quantile(0.01)) & (waterQualityDataset['chlorides'] < waterQualityDataset['chlorides'].quantile(0.99))]
waterQualityDataset=waterQualityDataset[(waterQualityDataset['free sulfur dioxide'] > waterQualityDataset['free sulfur dioxide'].quantile(0.01)) & (waterQualityDataset['free sulfur dioxide'] < waterQualityDataset['free sulfur dioxide'].quantile(0.99))]
waterQualityDataset=waterQualityDataset[(waterQualityDataset['total sulfur dioxide'] > waterQualityDataset['total sulfur dioxide'].quantile(0.01)) & (waterQualityDataset['total sulfur dioxide'] < waterQualityDataset['total sulfur dioxide'].quantile(0.99))]
waterQualityDataset=waterQualityDataset[(waterQualityDataset['sulphates'] > waterQualityDataset['sulphates'].quantile(0.01)) & (waterQualityDataset['sulphates'] < waterQualityDataset['sulphates'].quantile(0.99))]
waterQualityDataset=waterQualityDataset[(waterQualityDataset['residual sugar'] > waterQualityDataset['residual sugar'].quantile(0.01)) & (waterQualityDataset['residual sugar'] < waterQualityDataset['residual sugar'].quantile(0.99))]

X,y = waterQualityDataset.loc[:, waterQualityDataset.columns!='quality'],waterQualityDataset['quality']
print('a')