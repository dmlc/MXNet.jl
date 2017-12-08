module TestAutoGrad

using Base.Test

using MXNet


function checkgradient(f, x, y, ∇)
  ∇x = mx.attach_grad!(x)
  y′ = mx.record(f)
  @test copy(y′) ≈ y
  @test copy(∇x) |> sum == 0
  mx.backward!(y′)
  @test copy(mx.getgrad(x)) ≈ ∇
end  # function checkgradient


function test_getgrad()
  info("AutoGrad::getgrad")

  info("AutoGrad::getgrad::unattached")
  @test nothing == mx.getgrad(mx.zeros(10))

  info("AutoGrad::getgrad::attached")
  x = mx.NDArray([1 2; 3 4])
  grad = mx.attach_grad!(x)
  @test eltype(grad) ≡ Int
  @test copy(grad) == [0 0; 0 0]

  grad[:] = 42
  @test copy(mx.getgrad(x)) == [42 42; 42 42]
end


function test_mark_variables!()
  info("AutoGrad::mark_variables!")
  x = mx.zeros(4)
  ẋ = mx.zeros(4)
  y = mx.zeros(4)
  ẏ = mx.zeros(4)
  mx.mark_variables!([x, y], [ẋ, ẏ], [:nop, :nop])
  ẋ[:] = 42
  ẏ[:] = 24

  @test copy(mx.getgrad(x)) == [42, 42, 42, 42]
  @test copy(mx.getgrad(y)) == [24, 24, 24, 24]

  info("AutoGrad::mark_variables!::invalid grad_reqs")
  x = mx.zeros(4)
  y = mx.zeros(4)
  @test_throws ArgumentError mx.mark_variables!(x, y, :magic)
  @test_throws ArgumentError mx.mark_variables!([x], [y], [:magic])

  info("AutoGrad::mark_variables!::args length mismatch")
  x = mx.zeros(4)
  y = mx.zeros(4)
  z = mx.zeros(4)
  @test_throws ArgumentError mx.mark_variables!([x], [y, z])
end


function test_record()
  let x = mx.NDArray([1 2; 3 4])
    info("AutoGrad::record::backward!")

    y = [1 4; 9 16]
    ∇ = [2 4; 6 8]  # gradient is 2x
    checkgradient(x, y, ∇) do
      mx.square(x)
    end
  end

  let x = mx.NDArray([1 2; 3 4])
    info("AutoGrad::record::symbol")

    mx.attach_grad!(x)
    y = mx.record() do
      mx.square(x)
    end

    @test copy(y) == [1 4; 9 16]

    @test isa(mx.symbol(y), mx.SymbolicNode)
  end

  let x = mx.NDArray([1 2; 3 4])
    info("AutoGrad::record::backward!(retain_graph=true)")

    mx.attach_grad!(x)
    y = mx.record() do
      mx.square(x)
    end

    @test copy(y) == [1 4; 9 16]

    mx.backward!(y, retain_graph=true)
    # gradient is 2x
    @test copy(mx.getgrad(x)) == [2 4; 6 8]

    @test isa(mx.symbol(y), mx.SymbolicNode)
  end
end  # function test_record


function test_pause()
  info("AutoGrad::pause")
  let x = mx.NDArray([1 2; 3 4])
    ∇ = mx.attach_grad!(x)
    y = mx.record() do
      y = mx.square(x)
      mx.pause() do
        z = mx.square(y)
        @test copy(z) == [1 16; 81 256]
      end
      y
    end

    @test copy(y) == [1 4; 9 16]

    mx.backward!(y)
    @test copy(∇) == [2 4; 6 8]
  end
end  # function test_pause


function test_symbol()
  info("AutoGrad::symbol")

  let x = mx.zeros(4)
    mx.attach_grad!(x)
    @test isa(mx.symbol(x), mx.SymbolicNode)
  end
end


function test_add()
  info("AutoGrad::add")

  info("AutoGrad::add::x")
  let x = mx.NDArray([1 2; 3 4])
    y = [1 2; 3 4]
    ∇ = [1 1; 1 1]  # gradient is 1
    checkgradient(x, y, ∇) do
      x
    end
  end

  info("AutoGrad::add::+x")
  let x = mx.NDArray([1 2; 3 4])
    y = [1 2; 3 4]
    ∇ = [1 1; 1 1]  # gradient is 1
    checkgradient(x, y, ∇) do
      +x
    end
  end

  info("AutoGrad::add::x .+ 42")
  let x = mx.NDArray([1 2; 3 4])
    y = [43 44; 45 46]
    ∇ = [1 1; 1 1]  # gradient is 1
    checkgradient(x, y, ∇) do
      x .+ 42
    end
  end

  info("AutoGrad::add::42 .+ x")
  let x = mx.NDArray([1 2; 3 4])
    y = [43 44; 45 46]
    ∇ = [1 1; 1 1]
    checkgradient(x, y, ∇) do
      42 .+ x
    end
  end

  # TODO: info("AutoGrad::add::x .+ y")
end  # function test_add


function test_sub()
  info("AutoGrad::sub")

  info("AutoGrad::sub::-x")
  let x = mx.NDArray([1 2; 3 4])
    y = [-1 -2; -3 -4]
    ∇ = [-1 -1; -1 -1]  # gradient is -1
    checkgradient(x, y, ∇) do
      -x
    end
  end

  info("AutoGrad::sub::x .- 42")
  let x = mx.NDArray([1 2; 3 4])
    y = [-41 -40; -39 -38]
    ∇ = [1 1; 1 1]
    checkgradient(x, y, ∇) do
      x .- 42
    end
  end

  info("AutoGrad::sub::42 .- x")
  let x = mx.NDArray([1 2; 3 4])
    y = [41 40; 39 38]
    ∇ = -[1 1; 1 1]
    checkgradient(x, y, ∇) do
      42 .- x
    end
  end

  # TODO: info("AutoGrad::add::x .- y")
end  # function test_sub


function test_mul()
  info("AutoGrad::mul")

  info("AutoGrad::mul::2x .* x")
  let x = mx.NDArray([1 2; 3 4])
    y = [2 8; 18 32]
    ∇ = [4 8; 12 16]  # 4x
    checkgradient(x, y, ∇) do
      2x .* x
    end
  end

  info("AutoGrad::mul::x * 2 .* x")
  let x = mx.NDArray([1 2; 3 4])
    y = [2 8; 18 32]
    ∇ = [4 8; 12 16]  # 4x
    checkgradient(x, y, ∇) do
      x * 2 .* x
    end
  end
end


function test_div()
  info("AutoGrad::div")

  info("AutoGrad::div::x ./ 2")
  let x = mx.NDArray(Float32[1 2; 3 4])
    y = Float32[.5 1; 1.5 2]
    ∇ = [.5 .5; .5 .5]
    checkgradient(x, y, ∇) do
      x ./ 2
    end
  end

  info("AutoGrad::rdiv::2 ./ x")
  let A = Float32[1 2; 3 4], x = mx.NDArray(A)
    y = 2 ./ A
    ∇ = @. -2 / A^2  # -2 / x²
    checkgradient(x, y, ∇) do
      2 ./ x
    end
  end
end  # function test_div


@testset "AutoGrad Test" begin
  test_getgrad()
  test_mark_variables!()
  test_record()
  test_pause()
  test_symbol()
  test_add()
  test_sub()
  test_mul()
  test_div()
end


end  # model TestAutoGrad
