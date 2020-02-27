import pandas as pd
from mlp import *

df = pd.read_csv("iris.csv")
myMLP(df)