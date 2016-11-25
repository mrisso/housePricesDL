import pandas as pd

data = pd.read_csv("./train.csv")
data = pd.get_dummies(data)
data = data.fillna(data.mean())

data.to_csv("processedData.csv", sep=',')
