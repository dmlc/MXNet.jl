module TestSymbolModule
using MXNet
using Base.Test

################################################################################
# Utils
################################################################################

function create_network()
  arch = mx.@chain mx.Variable(:data) =>
         mx.Convolution(kernel = (3,3), pad = (1,1), stride = (1,1), num_filter = 64) =>
         mx.SoftmaxOutput(name=:softmax, multi_output = true)

  return arch
end

function create_single_neuron()
  arch = @mx.chain mx.Variable(:data) =>
         mx.FullyConnected(name=:fc1, num_hidden=1) =>
         mx.LinearRegressionOutput(name=:linout)
  return arch
end

################################################################################
# Test Implementations
################################################################################

function test_basic()
  info("SymbolModule::basic")
  
  m1 = mx.Module.SymbolModule(create_network())

  @test !mx.Module.isbinded(m1)
  @test !mx.Module.allows_training(m1)
  @test !mx.Module.isinitialized(m1)
  @test !mx.Module.hasoptimizer(m1)

  @test mx.Module.data_names(m1) == [:data]
  @test mx.Module.output_names(m1) == [:softmax_output]
  
  mx.Module.bind(m1, [(20, 20, 1, 10)], [(20, 20, 1, 10)])
  @test mx.Module.isbinded(m1)
  @test !mx.Module.isinitialized(m1)
  @test !mx.Module.hasoptimizer(m1)

  mx.Module.init_params(m1)
  @test mx.Module.isinitialized(m1)

  mx.Module.init_optimizer(m1)
  @test mx.Module.hasoptimizer(m1)
end

function test_init_params()
  info("SymbolModule::InitParams")
  m1 = mx.Module.SymbolModule(create_single_neuron(), 
                              label_names = [:linout_label])
  mx.Module.bind(m1, [(1, 10)], [(1, 10)])
  mx.Module.init_params(m1)

  # TODO Should be changed to tests
  info(mx.Module.get_params(m1))

  x = reshape(collect(1:10), (1, 10))
  y = reshape(collect(2:11), (1, 10))
  data = mx.ArrayDataProvider(:data => x, :linout_label => y; batch_size = 10)
  for batch in mx.eachdatabatch(data)
    mx.Module.forward(m1, batch)
  end
end

################################################################################
# Run tests
################################################################################

@testset "Symbol Module Test" begin
  test_basic()
  test_init_params()
end

end
