"""
    NativeModule

Allows the implementation of a MXNet module in native Julia. NDArrays
will be translated into native Julia arrays.
"""
type NativeModule{F<:Function,B<:Function} <: AbstractModule
   forward :: F
   backward :: B
end
