require 'torch'
require 'cunn'
require 'nn'
require 'csvigo'

net = torch.load("network.dat")
net = net:cuda()

normal = torch.load("normal.dat")

testData = csvigo.load({path='./testPD.csv', verbose='false', mode='raw'}); --load data from csv

table.remove(testData,1) --remove first row

for i=1, #testData do
	--removes first two columns of the entire set (id columns)
	table.remove(testData[i],1)
	for j=1, #testData[i] do
		--transforms string data into numbers
		testData[i][j] = tonumber(testData[i][j])
	end
end

testData = torch.Tensor(testData)
testData = testData:cuda()

nInputs = #(testData[1]) - 1

--Normalizing data based on training model
for i=1, (#testData)[1] do
	for j=1, (#testData[1])[1] do
		testData[i][j] = testData[i][j] / normal[j]
	end
end

for i = 1,(#testData)[1] do
	local sample = testData[i]
	local myPrediction = net:forward(sample)
	local predictionValue = myPrediction[1] * normal[1]
	print(string.format("%2d  %6.2f", i, predictionValue))
end
