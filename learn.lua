require 'torch'
require 'cunn'
require 'optim'
require 'csvigo'

--logger = optim.Logger('predict_log.txt')

allData = csvigo.load({path='./trainPD.csv', verbose='false', mode='raw'}); --load data from csv

table.remove(allData,1) --remove first row

trainIndex = 1
validateIndex = 1

nInputs = #(allData[1]) - 1

data = torch.Tensor(1022,(nInputs+1))
validate = torch.Tensor(1460-1022,(nInputs+1))

for i=1, #allData do
	--removes first two columns of the entire set (id columns)
	table.remove(allData[i],1)
	if(i<=1022) then
		for j=1, #allData[i] do
			--transforms string data into numbers
			data[trainIndex][j] = tonumber(allData[i][j])
		end
		trainIndex = trainIndex + 1
	else
		for j=1, #allData[i] do
			--transforms string data into numbers
			validate[validateIndex][j] = tonumber(allData[i][j])
		end
		validateIndex = validateIndex + 1
	end
end

--create Tensor with data table
--data = torch.Tensor(data)
--validate = torch.Tensor(validate)
--Using GPU
data = data:cuda()
validate = validate:cuda()


--Creating zero tensor with the data size (columns)
--this tensor will be resposible for saving the
--highest value of each column
normal = torch.Tensor(nInputs+1)
normal:zero()

--Searching for the highest value
for i=1, (#data)[1] do
	for j=1, (nInputs+1) do
		if(data[i][j] > normal[j]) then
			normal[j] = data[i][j]
		end
	end
end

--Searching for zeros
for i=1, (nInputs+1) do
	if(normal[i] == 0) then
		normal[i] = 1
	end
end

torch.save("normal.dat",normal)
--Normalizing data: Dividing every value of each
--column with the highest one
for i=1, (#data)[1] do
	for j=1, (nInputs+1) do
		data[i][j] = data[i][j] / normal[j]
		if(i < validateIndex) then
			validate[i][j] = validate[i][j] / normal[j]
		end
	end
end

--Neural Network
require 'nn'

net = nn.Sequential()
net:add(nn.Linear(nInputs,200))
--net:add(nn.Tanh())
net:add(nn.Linear(200,200))
--net:add(nn.Tanh())
net:add(nn.Linear(200,200))
--net:add(nn.Tanh())
net:add(nn.Linear(200,200))
--net:add(nn.Tanh())
net:add(nn.Linear(200,200))
--net:add(nn.Tanh())
net:add(nn.Linear(200,1))
--Using GPU
net = net:cuda()

criterion = nn.MSECriterion()
criterion = criterion:cuda()

x, dl_dx = net:getParameters()

feval = function(x_new)
	if x ~= x_new then
		x:copy(x_new)
	end

	_nidx_ = (_nidx_ or 0) + 1
	if _nidx_ > (#data)[1] then _nidx_ = 1 end

	local sample = data[_nidx_]
	local target = sample[{ {1} }]
	local inputs = sample[{ {2,(nInputs+1)} }]

	dl_dx:zero()

	forwardOut = net:forward(inputs)

	local loss_x = criterion:forward(forwardOut,target)
	net:backward(inputs, criterion:backward(net.output, target))

	result = result + math.abs(forwardOut[1]-target[1])

	return loss_x, dl_dx
end

sgdParams = {
	learningRate = 1e-3,
	learningRateDecay = 1e-4,
	weightDecay = 0,
	momentum = 0
}

for i = 1,1e4 do
	--lossAtual = 0
	result = 0

	for i = 1, (#data)[1] do
		_,_= optim.sgd(feval,x,sgdParams)
		--lossAtual = lossAtual + fs[1]
	end

	--lossAtual = lossAtual / (#data)[1]
	--print('current loss = ' .. lossAtual)
	result = (result / (#data)[1]) * normal[1]

	print(result)

	logger:add{['training prediction'] = result}
	logger:style{['training prediction'] = '-'}
	logger:plot()
end

for i = 1,(#validate)[1] do
	local sample = validate[i]
	local inputs = sample[{ {2,(nInputs+1)} }]
	local myPrediction = net:forward(inputs)
	local value = (validate[i][{{1}}])[1]
	local dif = math.abs(myPrediction[1]-value) * normal[1]
	print(string.format("%2d  %6.2f %6.2f %6.2f", i, myPrediction[1], value, dif))
end

torch.save("network.dat", net)
