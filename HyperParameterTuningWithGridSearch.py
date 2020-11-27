import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

titanic_df = pd.read_csv('datasets/titanic_processed.csv')

X = titanic_df.drop('Survived', axis=1)
Y = titanic_df['Survived']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

def summarize_classification(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred, normalize=True)
    acc_num = accuracy_score(y_test, y_pred, normalize=False)

    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Test data count: ", len(y_test))
    print("Acc count: ", acc_num)
    print("Accuracy score: ", acc)
    print("Precision score: ", prec)
    print("Recall score: ", recall)
    print()

parameters = {'max_depth': [2,4,5,7,9,10]}
grid_search = GridSearchCV(DecisionTreeClassifier(), parameters, cv=3, return_train_score=True)
grid_search.fit(x_train, y_train)

print(grid_search.best_params_)

for i in range(6):
    print('Parameters: ', grid_search.cv_results_['params'][i])
    print('Mean Test score: ', grid_search.cv_results_['mean_test_score'][i])
    print('Rank: ', grid_search.cv_results_['rank_test_score'][i])

decision_tree_model = DecisionTreeClassifier(max_depth=grid_search.best_params_['max_depth']).fit(x_train, y_train)
y_pred = decision_tree_model.predict(x_test)
summarize_classification(y_test, y_pred)

parameters = {'penalty': ['l1','l2'],'C': [0.1,0.4,0.8,1,2,5]}
grid_search = GridSearchCV(LogisticRegression(solver='liblinear'), parameters, cv=3, return_train_score=True)
grid_search.fit(x_train, y_train)

logistic_regression_model = LogisticRegression(penalty=grid_search.best_params_['penalty'], C=grid_search.best_params_['C'])\
    .fit(x_train, y_train)
y_pred = logistic_regression_model.predict(x_test)
summarize_classification(y_test, y_pred)