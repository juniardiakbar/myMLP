import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Preprocessing the dataset
df = pd.read_csv('../iris.csv')
X = df.drop(['variety'], axis=1).values
y = df['variety'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Training the dataset
mlp = MLPClassifier(solver='sgd', activation='logistic', batch_size=20,
                    learning_rate_init=0.01, max_iter=1000, hidden_layer_sizes=(6, 3))
mlp.fit(X_train, y_train)


# Testing
pred = mlp.predict(X_test)
print("train = " + str(mlp.score(X_train, y_train)))  # on training dataset
print("test = " + str(accuracy_score(y_test, pred)))  # on testing dataset
