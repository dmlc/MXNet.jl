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


@testset "AutoGrad Test" begin
  test_grad()
end

end  # model TestAutoGrad
