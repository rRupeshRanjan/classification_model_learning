import sklearn
from sklearn import preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

titanic_df = pd.read_csv('datasets/titanic_train.csv')
titanic_df.drop(['PassengerId','Name','Cabin','Ticket'], 'columns', inplace=True)

titanic_df = titanic_df.dropna()

# labeling string data to 0/1 numeric, for gender
label_encoding = preprocessing.LabelEncoder()
titanic_df['Sex'] = label_encoding.fit_transform(titanic_df['Sex'].astype(str))

# one- hot encoding for multiple values under a column (embarked)
titanic_df = pd.get_dummies(titanic_df, columns=['Embarked'])

# shuffle data before training
titanic_df = titanic_df.sample(frac=1).reset_index(drop=True)

# save to csv file without index
titanic_df.to_csv('datasets/titanic_processed.csv', index=False)