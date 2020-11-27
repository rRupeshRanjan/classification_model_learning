import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier


fashion_mnist_df = pd.read_csv('datasets/fashion-mnist_train.csv')
fashion_mnist_df = fashion_mnist_df.sample(frac=0.3).reset_index(drop=True)

lookup = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


def display_image(features, actual_label):
    print("Actual label: ", lookup[actual_label])

    plt.imshow(features.reshape(28,28))


def summarize_classification(y_test, y_pred, avg_method='weighted'):

    acc = accuracy_score(y_test, y_pred, normalize=True)
    acc_num = accuracy_score(y_test, y_pred, normalize=False)

    prec = precision_score(y_test, y_pred, average=avg_method)
    recall = recall_score(y_test, y_pred, average=avg_method)

    print('Test Data count: ', len(y_test))
    print('Accuracy count: ', acc_num)
    print('Accuracy score: ', acc)
    print('Precision score: ', prec)
    print('Recall score: ', recall)


X = fashion_mnist_df[fashion_mnist_df.columns[1:]]/255.
Y = fashion_mnist_df['label']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


logistic_model = LogisticRegression(solver='sag', multi_class='auto', max_iter=10000).fit(x_train, y_train)
y_pred = logistic_model.predict(y_test)

summarize_classification(y_test, y_pred)
