module TestSequentialModule
using MXNet
using Base.Test

using ..Main: reldiff

################################################################################
# Utils
################################################################################

################################################################################
# Test Implementations
################################################################################

function test_basic()
  info("SequentialModule::basic")

  net1 = @mx.chain mx.Variable(:data) =>
           mx.FullyConnected(name=:fc1, num_hidden=4)
  net2 = @mx.chain mx.FullyConnected(mx.SymbolicNode, name=:fc2, num_hidden=1) =>
           mx.LinearRegressionOutput(name=:linout)

  m1 = mx.Module.SymbolModule(net1, label_names=Symbol[])
  m2 = mx.Module.SymbolModule(net2)
  seq_mod = mx.Module.SequentialModule()
  mx.Module.add(seq_mod, m1)
  mx.Module.add(seq_mod, m2, take_labels=true)
  @test !mx.Module.isbinded(seq_mod)
  @test !mx.Module.allows_training(seq_mod)
  @test !mx.Module.isinitialized(seq_mod)
  @test !mx.Module.hasoptimizer(seq_mod)

  @test mx.Module.data_names(seq_mod) == [:data]
  @test mx.Module.output_names(seq_mod) == [:linout_output]

  mx.Module.bind(seq_mod, [(4, 10)], [(1, 10)])
  @test mx.Module.isbinded(seq_mod)
  @test !mx.Module.isinitialized(seq_mod)
  @test !mx.Module.hasoptimizer(seq_mod)
end

################################################################################
# Run tests
################################################################################

@testset "  Sequential Module Test" begin
  test_basic()
end

end
