require 'csvigo'

data = csvigo.load({path='./processedData.csv', verbose='false', mode='raw'}) --load data from csv

table.remove(data,1) --remove first row

count = 0

for i=1, #data do
	for j=1, #data[i] do
		--transforms string data into numbers
		data[i][j] = tonumber(data[i][j])
		if i==1 then
			count = count + 1
		end
	end
	--removes first two columns of the entire set (id columns)
	table.remove(data[i],1)
	table.remove(data[i],1)
end

nInputs = count - 2

--create Tensor with data table
torch.Tensor(data)

--Neural Network
require 'nn'

net = nn.Sequential()
net:add(nn.Linear(nInputs,200))
net:add(nn.Linear(200,200))
net:add(nn.Linear(200,200))
net:add(nn.Linear(200,200))
net:add(nn.Linear(200,200))
net:add(nn.Linear(200,1))

criterion = nn.MSECriterion()

x, dl_dx = net:getParameters()

feval = function(x_new)
	if x ~= x_new then
		x:copy(x_new)
	end

	_nidx_ = (_nidx_ or 0) + 1
	if _nidx_ > (#data)[1] then _nidx_ = 1 end

	local sample = data[_nidx_]
	local target = sample[{ {nInputs} }]
	local inputs = sample[{ {1,(nInputs-1)} }]

	dl_dx:zero()

	local loss_x = criterion:forward(net:forward(inputs),target)
	net:backward(inputs, criterion:backward(net.output, target))

	return loss_x, dl_dx
end
