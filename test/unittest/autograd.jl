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


function test_add()
  info("AutoGrad::add")

  info("AutoGrad::add::+x")
  let x = mx.NDArray([1 2; 3 4])
    g = mx.attach_grad(x)
    y = mx.record() do
      +x
    end

    @test copy(g) == [0 0; 0 0]
    @test copy(y) == [1 2; 3 4]

    mx.backward(y)
    # gradient is 0
    @test copy(g) == [0 0; 0 0]
  end

  info("AutoGrad::add::x .+ 42")
  let x = mx.NDArray([1 2; 3 4])
    g = mx.attach_grad(x)
    y = mx.record() do
      x .+ 42
    end

    @test copy(g) == [0 0; 0 0]
    @test copy(y) == [43 44; 45 46]

    mx.backward(y)
    # gradient is 1
    @test copy(g) == [1 1; 1 1]
  end

  info("AutoGrad::add::42 .+ x")
  let x = mx.NDArray([1 2; 3 4])
    g = mx.attach_grad(x)
    y = mx.record() do
      42 .+ x
    end

    @test copy(g) == [0 0; 0 0]
    @test copy(y) == [43 44; 45 46]

    mx.backward(y)
    # gradient is 1
    @test copy(g) == [1 1; 1 1]
  end

  # TODO: info("AutoGrad::add::x .+ y")
end  # function test_add


function test_sub()
  info("AutoGrad::sub")

  info("AutoGrad::sub::-x")
  let x = mx.NDArray([1 2; 3 4])
    g = mx.attach_grad(x)
    y = mx.record() do
      -x
    end

    @test copy(g) == [0 0; 0 0]
    @test copy(y) == [-1 -2; -3 -4]

    mx.backward(y)
    # gradient is -1
    @test copy(g) == [-1 -1; -1 -1]
  end

  info("AutoGrad::sub::x .- 42")
  let x = mx.NDArray([1 2; 3 4])
    g = mx.attach_grad(x)
    y = mx.record() do
      x .- 42
    end

    @test copy(g) == [0 0; 0 0]
    @test copy(y) == [-41 -40; -39 -38]

    mx.backward(y)
    # gradient is 1
    @test copy(g) == [1 1; 1 1]
  end

  info("AutoGrad::sub::42 .- x")
  let x = mx.NDArray([1 2; 3 4])
    g = mx.attach_grad(x)
    y = mx.record() do
      42 .- x
    end

    @test copy(g) == [0 0; 0 0]
    @test copy(y) == [41 40; 39 38]

    mx.backward(y)
    # gradient is -1
    @test copy(g) == [-1 -1; -1 -1]
  end

  # TODO: info("AutoGrad::add::x .- y")
end  # function test_sub


function test_mul()
  info("AutoGrad::mul")

  let x = mx.NDArray([1 2; 3 4])
    g = mx.attach_grad(x)
    y = mx.record() do
      2x .* x
    end

    @test copy(g) == [0 0; 0 0]
    @test copy(y) == [2 8; 18 32]

    mx.backward(y)
    # gradient is 4x
    @test copy(g) == [4 8; 12 16]
  end

  let x = mx.NDArray([1 2; 3 4])
    g = mx.attach_grad(x)
    y = mx.record() do
      x * 2 .* x
    end

    @test copy(g) == [0 0; 0 0]
    @test copy(y) == [2 8; 18 32]

    mx.backward(y)
    # gradient is 4x
    @test copy(g) == [4 8; 12 16]
  end
end


@testset "AutoGrad Test" begin
  test_getgrad()
  test_mark_variables()
  test_record()
  test_getsymbol()
  test_add()
  test_sub()
  test_mul()
end


end  # model TestAutoGrad
