import pandas as pd

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

allData = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']))

allData = pd.get_dummies(allData)
allData = allData.fillna(allData.mean())

newTrain = pd.concat((train.loc[:,'SalePrice'], allData.iloc[0:1460]),axis=1)

newTest = allData.iloc[1461:]

newTrain.to_csv('./trainPD.csv')
newTest.to_csv('./testPD.csv')
