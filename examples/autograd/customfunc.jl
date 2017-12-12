###############################################################################
#  swish
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
  swish() = new()  # with undefined fields
end

# create callable function
@mx.customfunc swish

function mx.forward(f::swish, x)
  f.x = x
  f.σ = @. 1 / (1 + e^(-x))  # assume there is no mx.sigmoid; we need to get hand dirty
  f.y = x .* f.σ
end

mx.backward!(f::swish, Δy #= coefficient of gradient =#) =
  @. (f.y + f.σ * (1 - f.y)) * Δy


###############################################################################
#  example usage
###############################################################################

x = mx.NDArray(Float32[1 2; 3 4])
∇ = mx.attach_grad!(x)
f = swish()
y = mx.record() do
  f(x)
end
mx.backward!(y)
∇


# For the record, here shows the overhead of custom function
#
# julia> @benchmark g()  # custom func swish
# BenchmarkTools.Trial:
#   memory estimate:  29.58 KiB
#   allocs estimate:  599
#   --------------
#   minimum time:     352.222 μs (0.00% GC)
#   median time:      391.971 μs (0.00% GC)
#   mean time:        443.911 μs (1.73% GC)
#   maximum time:     22.587 ms (28.61% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1
#
# julia> @benchmark h()  # with native NDArray operator
# BenchmarkTools.Trial:
#   memory estimate:  9.39 KiB
#   allocs estimate:  184
#   --------------
#   minimum time:     173.886 μs (0.00% GC)
#   median time:      202.246 μs (0.00% GC)
#   mean time:        250.379 μs (1.26% GC)
#   maximum time:     48.067 ms (36.39% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1
