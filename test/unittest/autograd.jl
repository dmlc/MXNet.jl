module TestAutoGrad

using Base.Test

using MXNet


function test_getgrad()
  info("AutoGrad::getgrad")

  info("AutoGrad::getgrad::unattached")
  @test nothing == mx.getgrad(mx.zeros(10))

  info("AutoGrad::getgrad::attached")
  x = mx.NDArray([1 2; 3 4])
  grad = mx.attach_grad(x)
  @test eltype(grad) ≡ Int
  @test copy(grad) == [0 0; 0 0]

  grad[:] = 42
  @test copy(mx.getgrad(x)) == [42 42; 42 42]
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

  @test copy(mx.getgrad(x)) == [42, 42, 42, 42]
  @test copy(mx.getgrad(y)) == [24, 24, 24, 24]

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


function test_record()
  let x = mx.NDArray([1 2; 3 4])
    info("AutoGrad::record::backward")

    mx.attach_grad(x)
    y = mx.record() do
      mx.square(x)
    end

    @test copy(y) == [1 4; 9 16]

    mx.backward(y)
    # gradient is 2x
    @test copy(mx.getgrad(x)) == [2 4; 6 8]
  end

  let x = mx.NDArray([1 2; 3 4])
    info("AutoGrad::record::getsymbol")

    mx.attach_grad(x)
    y = mx.record() do
      mx.square(x)
    end

    @test copy(y) == [1 4; 9 16]

    @test isa(mx.getsymbol(y), mx.SymbolicNode)
  end

  let x = mx.NDArray([1 2; 3 4])
    info("AutoGrad::record::backward(retain_graph=true)")

    mx.attach_grad(x)
    y = mx.record() do
      mx.square(x)
    end

    @test copy(y) == [1 4; 9 16]

    mx.backward(y, retain_graph=true)
    # gradient is 2x
    @test copy(mx.getgrad(x)) == [2 4; 6 8]

    @test isa(mx.getsymbol(y), mx.SymbolicNode)
  end
end  # function test_record()


function test_getsymbol()
  info("AutoGrad::getsymbol")

  let x = mx.zeros(4)
    mx.attach_grad(x)
    @test isa(mx.getsymbol(x), mx.SymbolicNode)
  end
end


@testset "AutoGrad Test" begin
  test_getgrad()
  test_mark_variables()
  test_record()
  test_getsymbol()
end


end  # model TestAutoGrad
