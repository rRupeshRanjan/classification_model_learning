from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

titanic_df = pd.read_csv('datasets/titanic_processed.csv')
titanic_df.head()

X = titanic_df.drop('Survived', axis = 1)
Y = titanic_df['Survived']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

logistic_model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear').fit(x_train, y_train)

y_pred = logistic_model.predict(x_test)

pred_results = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
titanic_crosstab = pd.crosstab(pred_results.y_pred, pred_results.y_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
