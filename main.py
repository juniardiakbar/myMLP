from mlp import *
import pandas as pd
from datetime import datetime
startTime = datetime.now()


df = pd.read_csv("iris.csv")

learning_rate = float(input("Learning rate: "))
max_iteration = int(input("Maximum iteration: "))
batch_size = int(input("Batch size: "))
hidden_layers = []
num_hidden_layers = int(input("Num of hidden layers: "))

for i in range(num_hidden_layers):
    hidden_layers.append(int(input("Num nodes layer " + str(i) + ": ")))


myMLP(df, learning_rate, max_iteration, hidden_layers, batch_size)
print(datetime.now() - startTime)
