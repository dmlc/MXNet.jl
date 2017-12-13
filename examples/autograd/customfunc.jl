using MXNet

###############################################################################
#  swish: option 1, with inner constructor
###############################################################################

"""
    swish(x) = x * σ(x)

See [Swish: a Self-Gated Activation Function]
(https://arxiv.org/pdf/1710.05941.pdf).
"""
mutable struct swish
  x
  y
  σ
  @mx.custom function swish(x)
    σ = @. 1 / (1 + e^(-x))  # assume there is no mx.sigmoid; we need to get hand dirty
    y = x .* σ
    new(x, σ, y)  # must return a object instance for @custom
  end
end

# the actual return value
mx.forward(f::swish, x) = f.y

mx.backward!(f::swish, Δy #= coefficient of gradient =#) =
  @. (f.y + f.σ * (1 - f.y)) * Δy

###############################################################################
#  swish2: option 2, with outer constructor
###############################################################################

mutable struct swish2
  x
  y
  σ
end

@mx.custom function swish2(x)
  σ = @. 1 / (1 + e^(-x))
  y = x .* σ
  swish2(x, σ, y)  # must return a object instance for @custom
end

mx.forward(f::swish2, x) = f.y

mx.backward!(f::swish2, Δy #= coefficient of gradient =#) =
  @. (f.y + f.σ * (1 - f.y)) * Δy

###############################################################################
#  example usage
###############################################################################

x = mx.NDArray(Float32[1 2; 3 4])
∇ = mx.attach_grad!(x)
y = mx.record() do
  swish(x)
end
mx.backward!(y)
∇


# For the record, here shows the overhead of custom function
#
# julia> @benchmark g()  # custom func swish
# BenchmarkTools.Trial:
#   memory estimate:  29.83 KiB
#   allocs estimate:  608
#   --------------
#   minimum time:     372.205 μs (0.00% GC)
#   median time:      475.992 μs (0.00% GC)
#   mean time:        565.441 μs (3.65% GC)
#   maximum time:     33.960 ms (47.71% GC)
#   --------------
#   samples:          8723
#   evals/sample:     1
#
# julia> @benchmark h()  # with native NDArray operator
# BenchmarkTools.Trial:
#   memory estimate:  9.39 KiB
#   allocs estimate:  184
#   --------------
#   minimum time:     179.940 μs (0.00% GC)
#   median time:      234.188 μs (0.00% GC)
#   mean time:        264.236 μs (1.47% GC)
#   maximum time:     35.323 ms (28.11% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1
