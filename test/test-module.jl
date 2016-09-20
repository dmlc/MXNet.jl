using MXNet

# Create Network
symbol = mx.@chain mx.Variable(:data) =>
mx.Convolution(kernel = (3,3), pad = (1,1), stride = (1,1), num_filter = 64) =>
mx.SoftmaxOutput(name=:softmax, multi_output = true)

m1 = mx.Module.SymbolModule(symbol)

mx.Module.bind(m1, [(20,20,1,10)], [(20,20,1,10)])
