module TestAutoGrad

using Base.Test

using MXNet


function test_grad()
  info("AutoGrad::grad")

  info("AutoGrad::grad::unattached")
  @test nothing == mx.grad(mx.zeros(10))

  info("AutoGrad::grad::attached")
  x = mx.NDArray([1 2; 3 4])
  grad = mx.attach_grad(x)
  @test eltype(grad) ≡ Int
  @test copy(grad) == [0 0; 0 0]

  grad[:] = 42
  @test copy(mx.grad(x)) == [42 42; 42 42]
end


function test_mark_variables()
  info("AutoGrad::mark_variables")
  x = mx.zeros(4)
  ẋ = mx.zeros(4)
  y = mx.zeros(4)
  ẏ = mx.zeros(4)
  mx.mark_variables([x, y], [ẋ, ẏ], [:nop, :nop])
  ẋ[:] = 42
  ẏ[:] = 24

  @test copy(mx.grad(x)) == [42, 42, 42, 42]
  @test copy(mx.grad(y)) == [24, 24, 24, 24]

  info("AutoGrad::mark_variables::invalid grad_reqs")
  x = mx.zeros(4)
  y = mx.zeros(4)
  @test_throws ArgumentError mx.mark_variables(x, y, :magic)
  @test_throws ArgumentError mx.mark_variables([x], [y], [:magic])

  info("AutoGrad::mark_variables::args length mismatch")
  x = mx.zeros(4)
  y = mx.zeros(4)
  z = mx.zeros(4)
  @test_throws ArgumentError mx.mark_variables([x], [y, z])
end


@testset "AutoGrad Test" begin
  test_grad()
  test_mark_variables()
end


end  # model TestAutoGrad
