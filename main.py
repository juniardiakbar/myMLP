from mlp import *
import pandas as pd
from datetime import datetime

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

startTime = datetime.now()


df = pd.read_csv("iris.csv")

df, goal_idx = encode_target_attribute(df)

train_X, train_y, test_X, test_y = split_dataset_to_train_and_test(
    df, 0.9)

mlp = myMLP(train_X, train_y, goal_idx, learning_rate=0.01,
            max_iteration=600, hidden_layer=[6, 3], batch_size=20)

pred_y = predict(mlp, test_X)

print(pred_y)
print(test_y.values.tolist())

results = confusion_matrix(pred_y, test_y.values.tolist())

print('Confusion Matrix :')
print(results)
print('Accuracy Score :', accuracy_score(pred_y, test_y) * 100, "%")
print('Report : ')
print(classification_report(pred_y, test_y))
