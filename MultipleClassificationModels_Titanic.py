import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

titanic_df = pd.read_csv('datasets/titanic_processed.csv')

features = list(titanic_df.columns[1:])
result_dict = {}

def summarize_classification(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred, normalize=True)
    acc_num = accuracy_score(y_test, y_pred, normalize=False)

    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return {'accuracy': acc,
            'precison': prec,
            'recall': recall,
            'accuracy_count': acc_num}

def build_model(classifier_fn, name_of_y_col, names_of_x_cols, dataset, test_frac=0.2):

    X = dataset[names_of_x_cols]
    Y = dataset[name_of_y_col]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_frac)

    model = classifier_fn(x_train, y_train)

    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)

    train_summary = summarize_classification(y_train, y_pred_train)
    test_summary = summarize_classification(y_test, y_pred)

    pred_results = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})

    model_crosstab = pd.crosstab(pred_results.y_pred, pred_results.y_test)

    return {'training' : train_summary,
            'test' : test_summary,
            'confusion_matrix': model_crosstab}

def compare_results():
    for key in result_dict:
        print('Classification key: ', key)

        print()
        print('Training Data')
        for score in result_dict[key]['training']:
            print(score, result_dict[key]['training'][score])

        print()
        print('Test Data')
        for score in result_dict[key]['test']:
            print(score, result_dict[key]['test'][score])

        print()

def logistic_fn(x_train, y_train):
    model = LogisticRegression(solver='liblinear')
    model.fit(x_train, y_train)

    return model

def linear_discriminant_fn(x_train, y_train, solver='svd'):
    model = LinearDiscriminantAnalysis(solver=solver)
    model.fit(x_train, y_train)

    return model

def quadratic_discriminant_fn(x_train, y_train):
    model = QuadraticDiscriminantAnalysis()
    model.fit(x_train, y_train)

    return model

def sgd_fn(x_train, y_train, max_iter=10000, tol=1e-4):
    model = SGDClassifier(max_iter=max_iter, tol=tol)
    model.fit(x_train, y_train)

    return model

def linear_svc_fn(x_train, y_train, C=1.0, max_iter=1000, tol=1e-3):
    model = LinearSVC(C=C, max_iter=max_iter, tol=tol, dual=False)
    model.fit(x_train, y_train)

    return model

def radius_neighbor_fn(x_train, y_train, radius=40.0):
    model = RadiusNeighborsClassifier(radius=radius)
    model.fit(x_train, y_train)

    return model

def decision_tree(x_train, y_train, max_depth=None, max_features=None):
    model = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features)
    model.fit(x_train, y_train)

    return model

def naive_bayes(x_train, y_train, priors=None):
    model = GaussianNB(priors=priors)
    model.fit(x_train, y_train)

    return model

result_dict['survived ~ logistic'] = build_model(logistic_fn, 'Survived', features, titanic_df)

# we dont take all the features, coz tone hot encoding can result into collinear data. Also called dummy trap.
# We can drop one of the hot encoded columns in this case
result_dict['survived ~ LDA'] = build_model(linear_discriminant_fn, 'Survived', features[0:-1], titanic_df)
result_dict['survived ~ QDA'] = build_model(quadratic_discriminant_fn, 'Survived', features[0:-1], titanic_df)

result_dict['Survived ~ SGD'] = build_model(sgd_fn, 'Survived', features, titanic_df)
result_dict['Survived ~ LinearSVC'] = build_model(linear_svc_fn, 'Survived', features, titanic_df)

# Constraints like radius can be played around to find the best fit
result_dict['Survived ~ radius_neighbors'] = build_model(radius_neighbor_fn, 'Survived', features, titanic_df)

# Constraints needs to be defined carefully as it leads to over-fitting on training data
# and might not perform well on test data
result_dict['Survived ~ decision_tree'] = build_model(decision_tree, 'Survived', features, titanic_df)

result_dict['Survived ~ naive_bayes'] = build_model(naive_bayes, 'Survived', features, titanic_df)

compare_results()