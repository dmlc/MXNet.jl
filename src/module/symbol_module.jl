import ....MXNet: mx # in order to use mx.
import ..mx: SymbolicNode, NDArray, Context, Executor, list_arguments, infer_shape, GRAD_NOP, AbstractExecutorGroup, list_outputs, DataParallelExecutorGroup, KVStore, OptimizationState, ADAM, UniformInitializer, set_params!, AbstractOptimizer

"""
    Module

Module is a basic module that wraps a `SymbolicNode`. It is functionally the same
as the `FeedForward` model, except using the module API.

A current limitation is that it only supports one context.

# Parameters
* `symbol :: SymbolicNode`: The wrapped `SymbolicNode`
* `data_names :: Vector{Symbol}`:
"""
type SymbolModule <: AbstractModule
  symbol :: SymbolicNode
  data_names :: Vector{Symbol}
  label_names :: Vector{Symbol}
  aux_names :: Vector{Symbol}
  context :: Vector{Context}

  binded :: Bool
  for_training :: Bool
  inputs_need_grad :: Bool
  params_initialized :: Bool
  optimizer_initialized :: Bool

  data_shapes :: Vector{Tuple{Vararg{Int}}}
  label_shapes :: Vector{Tuple{Vararg{Int}}}
  output_shapes :: Vector{Tuple{Vararg{Int}}}

  arg_arrays :: Nullable{Vector{NDArray}}
  aux_arrays :: Nullable{Vector{NDArray}}
  grad_arrays :: Nullable{Vector{NDArray}}
  params_dirty :: Bool

  fixed_param_names :: Nullable{Vector{Symbol}}
  optimizer
  kvstore
  update_on_kvstore

  arg_params
  aux_params

  exec_group :: AbstractExecutorGroup

  function SymbolModule(symbol::SymbolicNode, data_names::Vector{Symbol},
                        label_names::Vector{Symbol}, context :: Vector{Context},
                  fixed_param_names::Nullable{Vector{Symbol}})

    aux_names = mx.list_auxiliary_states(symbol)
    return new(symbol, data_names, label_names, aux_names, context,
               false, false, false, false, false,
               Vector{Tuple{Int}}(),
               Vector{Tuple{Int}}(),
               Vector{Tuple{Int}}(),
               Nullable{Vector{NDArray}}(),
               Nullable{Vector{NDArray}}(),
               Nullable{Vector{NDArray}}(),
               false,
               fixed_param_names)
  end
end

function SymbolModule(symbol::SymbolicNode;
                data_names = [:data], label_names = [:softmax_label],
                context = [mx.cpu()], fixed_param_names = nothing)
  fixed_param_names = Nullable{Vector{Symbol}}(fixed_param_names)
  if !isa(context, Vector{Context})
    context = [context]
  end
  @assert !isempty(data_names)
  @assert !isempty(context)
  return SymbolModule(symbol, data_names, label_names, context, fixed_param_names)
end

### default API
isbinded(self::SymbolModule) = self.binded
allows_training(self::SymbolModule) = self.for_training
isinitialized(self::SymbolModule) = self.params_initialized
hasoptimizer(self::SymbolModule) = self.optimizer_initialized

data_names(self::SymbolModule) = self.data_names
output_names(self::SymbolModule) = list_outputs(self.symbol)

function data_shapes(self::SymbolModule)
  !isbinded(self) && return Vector{Tuple{Int}}()
  return self.data_shapes
end

function label_shapes(self::SymbolModule)
  !isbinded(self) && return Vector{Tuple{Int}}()
  return self.label_shapes
end

function output_shapes(self::SymbolModule)
  !isbinded(self) && return Vector{Tuple{Int}}()
  return self.output_shapes
end

function get_params(self::SymbolModule)
  if !(isbinded(self) && isinitialized(self))
    return (Nullable{Dict{Symbol, NDArray}}(), Nullable{Dict{Symbol, NDArray}}())
  end
  if self.params_dirty
    sync_params_from_device(self)
  end

  return (self.arg_params, self.aux_params)
end

function init_params(self::SymbolModule; initializer=UniformInitializer(0.07), arg_params=nothing,
                     aux_params=nothing, allow_extra_params=false, force_init=false)
  if isinitialized(self) && !force_init
    return self
  end

  @assert isbinded(self) "Call `bind` before initialization"

  if !isdefined(self, :arg_params) || isempty(self.arg_params)
    self.arg_params = Dict(k => zeros(size(v)) for (k, v) in self.exec_group.arg_params)
  end

  if !isdefined(self, :aux_params) || isempty(self.aux_params)
    self.aux_params = Dict(k => zeros(size(v)) for (k, v) in self.exec_group.aux_params)
  end

  # TODO need initialization

  # copy the initialized parameters to devices
  set_params!(self.exec_group, self.arg_params, self.aux_params, allow_extra_params=allow_extra_params)

  self.params_dirty = false
  self.params_initialized = true

  return self
end

function bind(self::SymbolModule, data_shapes, label_shapes = Vector{Tuple{Int}}();
              for_training=true, inputs_need_grad=true, force_rebind=false,
              grad_req=mx.GRAD_WRITE, shared_group = nothing)
  if force_rebind
    reset_bind(self)
  end

  if isbinded(self)
    warn("Already bound, ignoring bind()")
    return self
  end

  if !for_training
    @assert !inputs_need_grad
  end

  self.for_training = for_training
  self.inputs_need_grad = inputs_need_grad
  self.binded = true


  @assert length(self.data_names)  == length(data_shapes)
  @assert length(self.label_names) == length(label_shapes)

  self.data_shapes = data_shapes
  self.label_shapes = label_shapes

  self.exec_group = DataParallelExecutorGroup(self.symbol, self.context,
                      self.data_shapes, self.data_names,
                      self.label_shapes, self.label_names,
                      self.for_training, self.inputs_need_grad, shared_group,
                      self.fixed_param_names, grad_req)
  return self
end

# TODO add description
function init_optimizer(self::SymbolModule; optimizer::AbstractOptimizer=ADAM(), kvstore :: Union{Base.Symbol, KVStore}=:local, force_init :: Bool=false)
  @assert isbinded(self) && isinitialized(self)

  if hasoptimizer(self) && !force_init
    warn("Optimizer already initialized, ignoring...")
    return self
  end

  # TODO initialize KV store
  # setup kvstore
  #= kvstore, update_on_kvstore = _create_kvstore(kvstore, length(self.context), self.arg_params) =#
  kvstore, update_on_kvstore = nothing, false

  self.optimizer = optimizer
  self.kvstore = kvstore
  self.update_on_kvstore = update_on_kvstore
  self.optimizer_initialized = true

  # add adequate calculation of batch_size
  op_state = OptimizationState(self.data_shapes[1][end])
  optimizer.state = op_state

  if !isa(kvstore, Void)
    if update_on_kvstore
      set_optimizer(kvstore, optimizer)
    end

    info("Initializing KVStore...")
    # init kv with gradients
    for idx = 1:length(param_arrays)
      param_on_devs = param_arrays[idx]

      init!(kvstore, idx, self.arg_params[param_names[idx]])

      if update_on_kvstore
        # pull weights back
        pull!(kvstore, idx, param_on_devs, priority=-idx)
      end
    end
  end
  
	# TODO add preloaded states
  #= if !isa(self.preload_opt_states, Void) =#
  #=   load_optimizer_states!(self, self.preload_opt_states) =#
  #=   self.preload_opt_states = nothing =#
  #= end =#

  return self
end

# TODO add description
"""
    forward(module, data_provider, data_batch; is_train)
Forward computation.
# Arguments
* `data_batch` : AbstractDataBatch
* `is_train` : Nullable{Bool}
  Default is `nothing`, which means `is_train` takes the value of `self.for_training`.
"""
function forward(self::SymbolModule, data_provider :: AbstractDataProvider, data_batch :: AbstractDataBatch, is_train=nothing)
  @assert isbinded(self) && isinitialized(self)
  is_train = convert(Nullable{Bool}, is_train)
  mx.forward(self.exec_group, data_provider, data_batch, is_train)
end

"""
    backward(module, out_grads)
Backward computation.
# Arguments
* `out_grads` : NDArray or list of NDArray, optional
  Gradient on the outputs to be propagated back.
  This parameter is only needed when bind is called
  on outputs that are not a loss function.
"""
function backward(self:: SymbolModule, out_grads=nothing)
  @assert isbinded(self) && isinitialized(self)
  backward(self.exec_group, out_grads=out_grads)
end


"""
    update!(mod)
Update parameters according to the installed optimizer and the gradients computed
in the previous forward-backward batch.
"""
function update!(self::SymbolModule)
  @assert isbinded(self) && isinitialized(self) && hasoptimizer(self)
  self.params_dirty = true
  if self.update_on_kvstore
    _update_params_on_kvstore(self.kvstore,
                              self.exec_group.param_arrays,
                              self.exec_group.grad_arrays)
  else
    _update_params(self.kvstore,
                   self.exec_group.param_arrays,
                   self.exec_group.grad_arrays,
                   updater=self.updater,
                   num_device=length(self.context))
  end
end

##
# Internals
##

function sync_params_from_devices(self::SymbolModule)
  throw(MethodError(sync_params_from_devices, (self,)))
end

"""
    borrow_optimizer!(module, shared_module)
Borrow optimizer from a shared module. Used in bucketing, where exactly the same 
optimizer (esp. kvstore) is used.
# Arguments
* `module` : SymbolModule
* `shared_module` : SymbolModule
"""
function borrow_optimizer!(self::SymbolModule, shared_module::SymbolModule)
  @assert hasoptimizer(shared_module)
  self.optimizer = shared_module.optimizer
  self.kvstore = shared_module.kvstore
  self.update_on_kvstore = shared_module.update_on_kvstore
  self.updater = shared_module.updater
  self.optimizer_initialized = true
end
