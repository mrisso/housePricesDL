require 'torch'
require 'cunn'
require 'optim'
require 'csvigo'

logger = optim.Logger('predict_log.txt')

data = csvigo.load({path='./trainPD.csv', verbose='false', mode='raw'}) --load data from csv

table.remove(data,1) --remove first row

for i=1, #data do
	--removes first two columns of the entire set (id columns)
	table.remove(data[i],1)
	for j=1, #data[i] do
		--transforms string data into numbers
		data[i][j] = tonumber(data[i][j])
	end
end

--create Tensor with data table
data = torch.Tensor(data)
--Using GPU
data = data:cuda()

nInputs = (#data[1])[1] - 1

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

--Normalizing data: Dividing every value of each
--column with the highest one
for i=1, (#data)[1] do
	for j=1, (nInputs+1) do
		data[i][j] = data[i][j] / normal[j]
	end
end

--Neural Network
require 'nn'

net = nn.Sequential()
net:add(nn.Linear(nInputs,200))
net:add(nn.Linear(200,200))
net:add(nn.Linear(200,200))
net:add(nn.Linear(200,200))
net:add(nn.Linear(200,200))
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

test = csvigo.load({path='./.csv', verbose='false', mode='raw'}) --load test from csv


table.remove(test,1) --remove first row

for i=1, #test do
	--removes first two columns of the entire set (id columns)
	table.remove(test[i],1)
	table.remove(test[i],1)
	for j=1, #test[i] do
		--transforms string test into numbers
		test[i][j] = tonumber(test[i][j])
	end
end

test = torch.Tensor(test)
test:cuda()

print(nInputs)
print((#test[1])[1])
for i=1, (#test)[1] do
	for j=1, nInputs do
		print(j)
		test[i][j] = test[i][j] / normal[j+1]
	end
end
