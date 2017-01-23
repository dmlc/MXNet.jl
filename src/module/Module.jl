module Module
import ....MXNet: mx
import ..mx: DataBatch, AbstractDataProvider, AbstractDataBatch, DataBatchProvider
import ..mx: SymbolicNode, NDArray, Context, Executor, list_arguments, infer_shape,
             GRAD_NOP, AbstractExecutorGroup, list_outputs, DataParallelExecutorGroup,
             KVStore, OptimizationState, ADAM, UniformInitializer, set_params!,
             AbstractOptimizer, get_updater, update_params, provide_data,
             provide_label, AbstractEvalMetric, StubProvider, init, copy!,
             concatenate, eachdatabatch, reset!, Accuracy

"""
    AbstractModule

A module represnets a computation component. The design purpose of a module is
that abstracts a computation unit, that one can run forward, backward, update parameters, etc.
We aim to make the APIs easy to use, especially in the case when we need to use
an imperative API to work with multiple modules (e.g. stochastic depth networks).

A module has several states:

* Initial state. Memory is not allocated yet, not ready for computation.
* Binded. Shapes for inputs, outputs, and parameters are all known, memory allocated.
* Parameter initialized. For modules with parameters, doing computation before intitializing
  the parameters might result in undefined outputs.
* Optimizer installed. An optimizer can be installed to a module. After this, the parameters.
  of the module can be updated according to the optimizers after gradients are computed
  (forward-backward).

In order for a module to interact with others, a module should be able to report the following
information in its raw stage (before binded):

* [`data_names`](@ref): Names of required data.
* [`output_names`](@ref): Names of the defined outputs.

And also the following richer information after being binded:

* State information:
  * [`isbinded`](@ref): indicating whether the memory buffers needed for computation
    have been allocated.
  * [`allows_training`](@ref): whether the module is binded for training (if binded).
  * [`isinitialized`](@ref): indicating whether the parameters of this module have
    been initialized.
  * [`hasoptimizer`](@ref): indicating wherger an optimizers is defined and intialized.
* Input/Output information:
  * [`data_shapes`](@ref):
  * [`label_shapes`](@ref):
  * [`output_shapes`](@ref):
* Parameters (for modules with parameters)
  * [`get_params`](@ref):
  * [`set_params`](@ref):
  * [`init_params`](@ref):
* Setup:
  * [`bind`](@ref):
  * [`init_optimizer`](@ref):
* Computation:
  * [`forward`](@ref):
  * [`backward`](@ref):
  * [`update!`](@ref):
  * [`get_outputs`](@ref):
  * [`get_input_grads`](@ref):
  * [`update_metric`](@ref):
* Optional:
  * [`get_symbol`](@ref): Access the associated `SymbolicNode` if the module has one.
    The returned value needs not to be constant (or defined)

Based on the underlying API a high-level API is implemented:
* [`fit`](@ref):
* [`predict`](@ref):
* [`score`](@ref):
* [`forward_backward`](@ref):
"""
abstract AbstractModule

##
# Names
##
"""
    data_names(self::AbstractModule) -> Vector{Symbol}
"""
function data_names(self::AbstractModule)
  throw(MethodError(data_names, (self,)))
end

"""
    output_names(self::AbstractModule) -> Vector{Symbol}
"""
function output_names(self::AbstractModule)
  throw(MethodError(output_names, (self,)))
end

##
# State information
##

"""
    isbinded(self::AbstractModule) -> Bool
"""
function isbinded(self::AbstractModule)
  throw(MethodError(isbinded, (self,)))
end

"""
    allows_training(self::AbstractModule) -> Bool
"""
function allows_training(self::AbstractModule)
  throw(MethodError(allows_training, (self,)))
end

"""
    isinitialized(self::AbstractModule) -> Bool
"""
function isinitialized(self::AbstractModule)
  throw(MethodError(isinitialized, (self,)))
end

"""
    hasoptimizer(self::AbstractModule) -> Bool
"""
function hasoptimizer(self::AbstractModule)
  throw(MethodError(hasoptimizer, (self,)))
end

##
#  Input/Output information
##

"""
    data_shapes(AbstractModule)

A Dict of (name, shape) pairs specifying the data inputs to this module.
"""
function data_shapes(self :: AbstractModule)
  throw(MethodError(data_shapes, (self,)))
end

"""
    label_shapes(AbstractModule)

A Dict of (name, shape) pairs specifying the label inputs to this module.
If this module does not accept labels -- either it is a module without loss function,
or it is not binded for training, then this should return an empty Dict.
"""
function label_shapes(self :: AbstractModule)
  throw(MethodError(label_shapes, (self,)))
end

"""
    output_shapes(AbstractModule)

A Dict of (name, shape) pairs specifying the outputs of this module.
"""
function output_shapes(self :: AbstractModule)
  throw(MethodError(output_shapes, (self,)))
end

##
# Parameters
##

"""
    get_params(self::AbstractModule)

Get parameters, those are potentially copies of the the actual parameters used to do computation on the device.

# Returns
`(arg_params, aux_params)`, a pair of dictionary of name to value mapping.
"""
function get_params(self :: AbstractModule)
  throw(MethodError(get_params, (self,)))
end

"""
    set_params(self::AbstractModule; arg_params, aux_params, allow_missing, force_init)

Assign parameter and aux state values.

# Arguments
* `arg_params` : `Dict`. Dictionary of name to value (`NDArray`) mapping.
* `aux_params` : `Dict`.  Dictionary of name to value (`NDArray`) mapping.
* `allow_missing` : `Bool`.  If true, params could contain missing values, and the initializer will be called to fill those missing params.
* `force_init` : `Bool`.  If true, will force re-initialize even if already initialized.
"""
function set_params(self::AbstractModule,
    arg_params::Dict{Symbol, NDArray},
    aux_params::Dict{Symbol, NDArray};
    allow_missing=false, force_init=false)
  init_params(self, arg_params=arg_params, aux_params=aux_params, allow_missing=allow_missing, force_init=force_init)
end

"""
    init_params!(self; kwargs...)

Initialize the parameters and auxiliary states.

# Arguments
* `self` : `AbstractModule`
* `initializer` : `AbstractInitializer`.  Called to initialize parameters if needed.
* `arg_params` : `Dict{Symbol, NDArray}`.  If not empty, should be a dictionary of existing `arg_params`. Initialization will be copied from that.
* `aux_params` : `Dict{Symbol, NDArray}`.  If not empty, should be a dictionary of existing `aux_params`. Initialization will be copied from that.
* `allow_missing` : `Bool`.  If true, params could contain missing values, and the initializer will be called to fill those missing params.
* `force_init` : `Bool`.  If true, will force re-initialize even if already initialized.
"""
function init_params(self :: AbstractModule, args...)
  throw(MethodError(init_params, (self, args...)))
end

###
# Setup
###
"""
"""
function bind(self :: AbstractModule, )
end

"""
"""
function init_optimizer(self :: AbstractModule, )
end

###
# Computation
###
"""
"""
forward(self :: AbstractModule, data_batch :: DataBatch, is_train) = forward(self, StubProvider(), data_batch, is_train)
function forward(self :: AbstractModule, provider :: AbstractDataProvider, data_batch :: AbstractDataBatch, is_train)
  throw(MethodError(forward, (self, )))
end

"""
"""
function backward(self :: AbstractModule, )
end

"""
"""
function update(self :: AbstractModule, )
end

"""
"""
function get_outputs(self :: AbstractModule, )
end

"""
"""
function get_input_grads(self :: AbstractModule, )
end

"""
"""

update_metric(self :: AbstractModule, eval_metric::AbstractEvalMetric, batch::AbstractDataBatch) = update_metric(self, eval_metric, StubProvider(), batch)
function update_metric(self :: AbstractModule, eval_metric::AbstractEvalMetric, provider::AbstractDataProvider, batch::AbstractDataBatch)
  throw(MethodError(update_metric, (self, )))
end

###
# Optional
##
"""
    get_symbol(self::AbstractModule) -> Nullable{SymbolicNode}

Returns the associated [`SymbolicNode`](@ref) of the module. It might not be defined or change over time.
"""
function get_symbol(self::AbstractModule)
  return Nullable{SymbolicNode}()
end

###
# High-level
###

"""
    fit(self::AbstractModule, train_data::AbstractDataProvider; kwargs...)

Train the module parameters.

# Arguments
* `train_data` : AbstractDataProvider
* `eval_data` : AbstractDataProvider
  If not `nothing`, will be used as validation set and evaluate the performance
  after each epoch.
* `eval_metric` : str or EvalMetric
  Default `'acc'`. The performance measure used to display during training.
* `epoch_end_callback` : function or list of function
  Each callback will be called with the current `epoch`, `symbol`, `arg_params`
  and `aux_params`.
* `batch_end_callback` : function or list of function
  Each callback will be called with a `BatchEndParam`.
* `kvstore` : Symbol or KVStore
  Default `:local`.
* `optimizer` : AbstractOptimizer
  Default `ADAM`
* `eval_end_callback` : function or list of function
  These will be called at the end of each full evaluation, with the metrics over
  the entire evaluation set.
* `eval_batch_end_callback` : function or list of function
  These will be called at the end of each minibatch during evaluation
* `initializer` : Initializer
  Will be called to initialize the module parameters if not already initialized.
* `arg_params` : dict
  Default `nothing`, if not `nothing`, should be existing parameters from a trained
  model or loaded from a checkpoint (previously saved model). In this case,
  the value here will be used to initialize the module parameters, unless they
  are already initialized by the user via a call to `init_params` or `fit`.
`arg_params` has higher priority to `initializer`.
* `aux_params` : dict
  Default `None`. Similar to `arg_params`, except for auxiliary states.
* `allow_missing` : bool
  Default `False`. Indicate whether we allow missing parameters when `arg_params`
  and `aux_params` are not `None`. If this is `True`, then the missing parameters
  will be initialized via the `initializer`.
* `force_rebind` : bool
  Default `False`. Whether to force rebinding the executors if already binded.
* `force_init` : bool
  Default `False`. Indicate whether we should force initialization even if the
  parameters are already initialized.
* `begin_epoch` : int
  Default `1`. Indicate the starting epoch. Usually, if we are resuming from a
  checkpoint saved at a previous training phase at epoch N, then we should specify
  this value as N+1.
* `num_epoch` : int
  Number of epochs to run training.

# Examples
An example of using fit for training::
```julia
# Assume training train_data and validation eval_data are ready
mx.Module.fit(mod, train_data, 10, eval_data=eval_data,
  optimizer=mx.SGD(lr=0.01, momentum=0.9))
```
"""
function fit(self::AbstractModule, train_data, num_epoch;
             initializer=UniformInitializer(0.07),
             optimizer = ADAM(),
             eval_data=nothing,
             eval_metric::AbstractEvalMetric=Accuracy(),
             validation_metric::AbstractEvalMetric = eval_metric,
             epoch_end_callback = nothing,
             batch_end_callback = nothing,
             kvstore = :local,
             eval_end_callback = nothing,
             eval_batch_end_callback = nothing,
             arg_params = Dict{Symbol, NDArray}(),
             aux_params = Dict{Symbol, NDArray}(),
             allow_missing = false,
             force_rebind = false,
             force_init = false,
             begin_epoch = 1)
  bind(self, train_data, for_training=true, force_rebind=force_rebind)
  init_params(self, initializer=initializer, arg_params=arg_params,
              aux_params=aux_params, allow_missing=allow_missing,
              force_init=force_init)
  init_optimizer(self, kvstore=kvstore, optimizer=optimizer, force_init=force_init)

  if validation_metric == nothing
    validation_metric = eval_metric
  end

  ####################################################################
  # training loop
  ####################################################################
  for epoch in begin_epoch:num_epoch
    time_start = time()
    reset!(eval_metric)
    for (nbatch, batch) in enumerate(eachdatabatch(train_data))
      forward_backward(self, batch)
      update(self)
      update_metric(self, eval_metric, batch)

      if batch_end_callback !== nothing
        error("Not implemented yet")
      end
    end

    # one epoch of training is finished
    for (name, val) in get(eval_metric)
      info("Epoch[$epoch] Train-$name=$val")
    end
    time_stop = time()
    info("Epoch[$epoch] Time cost=$(time_stop - time_start)")

     # sync aux params across devices
    arg_params, aux_params = get_params(self)
    set_params(self, arg_params, aux_params)

    if epoch_end_callback !== nothing
      error("Not implemented yet")
    end

    ##################################################################
    # evaluation on validation set
    ##################################################################
    if eval_data !== nothing
      res = score(self, eval_data, validation_metric,
                  score_end_callback=eval_end_callback,
                  batch_end_callback=eval_batch_end_callback,
                  epoch=epoch)
      #TODO: pull this into default
      for (name, val) in res
        info("Epoch[$epoch] Validation-$name=$val")
      end
    end
  end
end

# XXX: warning, this function is not type stable.
"""
    predict

Run prediction and collect the outputs.

# Arguments

* `eval_data` : `AbstractDataProvider`
* `num_batch` : Int
  Default is `None`, indicating running all the batches in the data iterator.
* `merge_batches` : `Bool`
  Default is `true`, see the doc for return values.
* `always_output_list` : `Bool`
  Default is `false`, see the doc for return values.

# Returns
When `merge_batches` is `true` (by default), the return value will be a vector
`[out1, out2, out3]`.  Where each element is concatenation of the outputs for
all the mini-batches. If further that `always_output_list` is `false` (by default),
then in the case of a single output, `out1` is returned instead of `[out1]`.
When `merge_batches` is `false`, the return value will be a nested list like
`[[out1_batch1, out2_batch1], [out1_batch2], ...]`. This mode is useful because
in some cases (e.g. bucketing), the module does not necessarily produce the same
number of outputs.
The objects in the results are `NDArray`s. If you need to work with julia array,
just call `Array{Float32}` on each of the `NDArray`.

# Examples
An example of using predict for prediction::
```julia
# Predict on the first 10 batches of `data` DataProvider
predict(m1, data, num_batch=10)
```
"""
function predict(self::AbstractModule, eval_data::AbstractDataProvider;
                 num_batch=nothing, merge_batches=true, always_output_list::Bool=false)
  @assert isbinded(self) && isinitialized(self)

  output_list = []
  for (nbatch, eval_batch) in enumerate(eachdatabatch(eval_data))
    if num_batch !== nothing && nbatch == num_back
      break
    end
    forward(self, eval_batch, false)

    outputs = get_outputs(self)
    push!(output_list, outputs)

  end

  if length(output_list) == 0
    return output_list
  end

  if merge_batches
    num_outputs = length(output_list[1])
    for out in output_list
      @assert(length(out) == num_outputs,
              "Cannot merge batches, as num of outputs is not the same in mini-batches. Maybe bucketing is used?")
    end
    output_list2 = [concatenate([out[i] for out in output_list]) for i = 1:num_outputs]

    if num_outputs == 1 && !always_output_list
      return output_list2[1]
    end

    return output_list2
  end

  return output_list
end

"""
    score(self::AbstractModule, eval_data, eval_metric; num_batch, batch_end_callback, reset=true, epoch=0)
"""
function score(self :: AbstractModule, eval_data, eval_metric; num_batch=nothing, batch_end_callback=nothing, score_end_callback=nothing,
               epoch=0)
  @assert isbinded(self) && isinitialized(self)

  reset!(eval_metric)

  for (nbatch, eval_batch) in enumerate(eachdatabatch(eval_data))
    if num_batch !== nothing && nbatch == num_back
      break
    end

    forward(self, eval_batch, false)
    update_metric(self, eval_metric, eval_batch)

    if batch_end_callback !== nothing
      error("Not implemented yet!")
    end
  end
  if score_end_callback !== nothing
    error("Not implemented yet!")
  end

  get(eval_metric)
end

"""
    forward_backward(self :: AbstractModule, data_batch)
"""
function forward_backward(self :: AbstractModule, data_batch)
  forward(self, data_batch, true)
  backward(self)
end

# include implementations
include("symbol_module.jl")
include("pipeline.jl")
include("native_module.jl")

end
