require 'torch'
require 'cunn'
require 'optim'
require 'csvigo'

logger = optim.Logger('loss_log.txt')

data = csvigo.load({path='./processedData.csv', verbose='false', mode='raw'}) --load data from csv

table.remove(data,1) --remove first row

for i=1, #data do
	--removes first two columns of the entire set (id columns)
	table.remove(data[i],1)
	table.remove(data[i],1)
	for j=1, #data[i] do
		--transforms string data into numbers
		data[i][j] = tonumber(data[i][j])
	end
	value = data[i][37]
	table.remove(data[i],37)
	table.insert(data[i],1,value)
end

--create Tensor with data table
data = torch.Tensor(data)

nInputs = (#data[1])[1]

--Neural Network
require 'nn'

net = nn.Sequential()
net:add(nn.Linear(nInputs,1))
--net:add(nn.Linear(200,200))
--net:add(nn.Linear(200,200))
--net:add(nn.Linear(200,200))
--net:add(nn.Linear(200,200))
--net:add(nn.Linear(200,1))

criterion = nn.MSECriterion()

x, dl_dx = net:getParameters()

feval = function(x_new)
	if x ~= x_new then
		x:copy(x_new)
	end

	_nidx_ = (_nidx_ or 0) + 1
	if _nidx_ > (#data)[1] then _nidx_ = 1 end

	local sample = data[_nidx_]
	local target = sample[{ {1} }]
	local inputs = sample[{ {2,nInputs} }]

	dl_dx:zero()

	local loss_x = criterion:forward(net:forward(inputs),target)
	net:backward(inputs, criterion:backward(net.output, target))

	return loss_x, dl_dx
end

sgdParams = {
	learningRate = 1e-3,
	learningRateDecay = 1e-4,
	weightDecay = 0,
	momentum = 0
}

for i = 1,1e4 do
	lossAtual = 0

	for i = 1, (#data)[1] do
		_,fs = optim.sgd(feval,x,sgdParams)
		lossAtual = lossAtual + fs[1]
	end

	lossAtual = lossAtual / (#data)[1]
	print('current loss = ' .. lossAtual)

	logger:add{['training error'] = lossAtual}
	logger:style{['training error'] = '-'}
	logger:plot()
end
