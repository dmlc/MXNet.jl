module TestSymbolModule
using MXNet
using Base.Test

using ..Main: reldiff

################################################################################
# Utils
################################################################################

function create_network()
  arch = mx.@chain mx.Variable(:data) =>
  mx.Convolution(kernel = (3,3), pad = (1,1), stride = (1,1), num_filter = 64) =>
  mx.SoftmaxOutput(name=:softmax, multi_output = true)

  return arch
end

function create_linreg(num_hidden::Int=1)
  arch = @mx.chain mx.Variable(:data) =>
  mx.FullyConnected(name=:fc1, num_hidden=num_hidden) =>
  mx.FullyConnected(name=:fc2, num_hidden=1) =>
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

  mx.Module.bind(m1, [(20, 20, 1, 10)], [(400, 10)])
  @test mx.Module.isbinded(m1)
  @test !mx.Module.isinitialized(m1)
  @test !mx.Module.hasoptimizer(m1)

  mx.Module.init_params(m1)
  @test mx.Module.isinitialized(m1)

  mx.Module.init_optimizer(m1)
  @test mx.Module.hasoptimizer(m1)
end

function test_shapes()
  info("SymbolModule::Shapes")

  m1 = mx.Module.SymbolModule(create_network())
  mx.Module.bind(m1, [(20, 20, 1, 10)], [(20, 20, 1, 10)])

  @test mx.Module.data_shapes(m1) == Dict(:data => (20, 20, 1, 10))
  @test mx.Module.label_shapes(m1) == Dict(:softmax_label => (20, 20, 1, 10))
  @test mx.Module.output_shapes(m1) == Dict(:softmax_output => (20, 20, 64, 10))

  m2 = mx.Module.SymbolModule(create_network(), label_names=[])
  mx.Module.bind(m2, [(20, 20, 1, 10)])
  @test isempty(mx.Module.label_shapes(m2))
end

function test_linear_regression(n_epoch::Int = 10)
  info("SymbolModule::LinearRegression")

  srand(123456)
  epsilon = randn(1, 10)
  x = rand(4, 10)
  y = mapslices(sum, [1, 2, 3, 4] .* x, 1) .+ epsilon
  data = mx.ArrayDataProvider(:data => x, :linout_label => y; batch_size = 5)

  metric = mx.MSE()
  m1 = mx.Module.SymbolModule(create_linreg(4), 
                              label_names = [:linout_label],
                              context=[mx.cpu(), mx.cpu()])
  mx.Module.bind(m1, data)
  mx.Module.init_params(m1)
  mx.Module.init_optimizer(m1)

  # TODO Should be changed to tests
  #= info(mx.Module.get_params(m1)) =#

  for i in 1:n_epoch
    for batch in mx.eachdatabatch(data)
      mx.Module.forward(m1, batch)
      mx.Module.backward(m1)
      mx.Module.update(m1)

      mx.Module.update_metric(m1, metric, batch)
    end

    for (name, value) in get(metric)
      info("Epoch: $i: $name = $value")
    end
    mx.reset!(metric)
  end

  y_pred = Float64[]
  for batch in mx.eachdatabatch(data)
    mx.Module.forward(m1, batch, false)
    append!(y_pred, Array{Float64}(mx.Module.get_outputs(m1)[1]))
  end
  y_pred = reshape(y_pred, 1, 10)

  info("Prediction: $y_pred")
  info("Actual:     $y")
  info("No noise:   $(mapslices(sum, [1, 2, 3, 4] .* x, 1))")

  # High Level Api
  name, score = mx.Module.score(m1, data, metric)[1]
  ha_pred = mx.copy(mx.Module.predict(m1, data))
  info("Predict result: ", ha_pred)
  info("Score $name     : ", score)
  
  @test sum(abs(ha_pred-y_pred)) < 1e-6

  m2 = mx.Module.SymbolModule(create_linreg(4), 
                              label_names = [:linout_label],
                              context=[mx.cpu(), mx.cpu()])
  mx.Module.fit(m2, data, 10, eval_metric=mx.MSE())
  name, score = mx.Module.score(m2, data, metric)[1]
  ha_pred = mx.copy(mx.Module.predict(m2, data))
  info("Predict result: ", ha_pred)
  info("Score $name     : ", score)
  @test sum(abs(ha_pred-y_pred)) < 1e-1
end

################################################################################
# Run tests
################################################################################

@testset "Symbol Module Test" begin
  test_basic()
  test_shapes()
  #= test_init_params(500) =#
  test_linear_regression()
end

end
