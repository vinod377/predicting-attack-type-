## importing libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import re
plt.rcParams['figure.figsize'] = (10.0,8.0)
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


df_train = pd.read_csv('/home/vinod/PycharmProjects/dataset/c771afa0-c-HackerEarthML4Updated/train_data.csv')
df_test = pd.read_csv('/home/vinod/PycharmProjects/dataset/c771afa0-c-HackerEarthML4Updated/test_data.csv')

def numeric_id(x):
    return int(re.search(r'\d+',x).group())
df_train['connection_id'] = df_train['connection_id'].apply(numeric_id)
connection = df_test['connection_id']
df_test['connection_id'] = df_test['connection_id'].apply(numeric_id)

# df1.describe()
miss = df_train.isnull().sum()/len(df_train)
miss = [miss>0]
miss
##no null values

correlation = df_train.corr()
correlation['target'].sort_values(ascending = False)
#sns.heatmap(correlation)

df2 = pd.DataFrame()
df2['target'] = df_train['target']
df_train.drop('target',axis = 1,inplace = True)
df_train.drop('connection_id',axis = 1,inplace = True)

##taking feature having correlation greater than or equal to zero
df_train = df_train[['cat_9','cont_10','cont_1','cont_8','cont_14','cont_11','cont_12','cat_19','cat_23',
           'cat_16','cont_3','cat_7','cat_14','cat_18','cat_13','cat_15','cat_12','cat_10','cat_6','cat_4','cat_2']]

df_test = df_test[['cat_9','cont_10','cont_1','cont_8','cont_14','cont_11','cont_12','cat_19','cat_23',
           'cat_16','cont_3','cat_7','cat_14','cat_18','cat_13','cat_15','cat_12','cat_10','cat_6','cat_4','cat_2']]


##creating and traing a model
X_train,X_test,Y_train,Y_test = train_test_split(df_train,df2,test_size = 0.3,random_state = 377)
scaler = preprocessing.StandardScaler().fit(X_train)

scaler.transform(X_train)
scaler.transform(X_test)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,Y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(Y_test,y_pred)

y_predict2 = clf.predict(df_test)

redictions = pd.DataFrame()
predictions['connection_id'] = connection
predictions['target'] = y_predict2

predictions.set_index('connection_id')


prediction_csv = predictions.to_csv("predictions.csv",index = False)





