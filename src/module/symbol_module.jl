import ....MXNet: mx # in order to use mx.

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

  arg_arrays :: Nullable{Vector{NDArray}}
  aux_arrays :: Nullable{Vector{NDArray}}
  grad_arrays :: Nullable{Vector{NDArray}}
  params_dirty :: Bool

  fixed_param_names :: Nullable{Vector{Symbol}}
  optimizer
  updater
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
               Nullable{Vector{NDArray}}(),
               Nullable{Vector{NDArray}}(),
               Nullable{Vector{NDArray}}(),
               false,
               fixed_param_names)
  end
end

function SymbolModule(symbol::SymbolicNode;
                      data_names = [:data],
                      label_names = [:softmax_label],
                context = [mx.cpu()], fixed_param_names = nothing)
  fixed_param_names = Nullable{Vector{Symbol}}(fixed_param_names)
  label_names = Vector{Symbol}(label_names)
  context = _wrap_context(context)
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
label_names(self::SymbolModule) = self.label_names
output_names(self::SymbolModule) = list_outputs(self.symbol)

"""
    get_symbol(self::SymbolModule) -> Nullable{SymbolicNode}

Returns the associated [`SymbolicNode`](@ref) of the module. It might not be defined or change over time.
"""
function get_symbol(self::SymbolModule)
  return Nullable{SymbolicNode}(self.symbol)
end

function data_shapes(self::SymbolModule)
  !isbinded(self) && return Dict{Symbol, Vector{Tuple{Int}}}()
  return Dict(k => v for (k, v) in zip(data_names(self), self.data_shapes))
end

function label_shapes(self::SymbolModule)
  !isbinded(self) && return Dict{Symbol, Vector{Tuple{Int}}}()
  return Dict(k => v for (k, v) in zip(label_names(self), self.label_shapes))
end

function output_shapes(self::SymbolModule)
  !isbinded(self) && return Dict{Symbol, Vector{Tuple{Int}}}()
  return mx.output_shapes(self.exec_group)
end

function get_params(self::SymbolModule)
  if !(isbinded(self) && isinitialized(self))
    return (Dict{Symbol, NDArray}(), Dict{Symbol, NDArray}())
  end
  if self.params_dirty
    mx.get_params!(self.exec_group, self.arg_params, self.aux_params)
    self.params_dirty = false
  end

  return (self.arg_params, self.aux_params)
end

function init_params(self::SymbolModule; 
    initializer=UniformInitializer(0.07), 
    arg_params::Dict{Symbol, NDArray}=Dict{Symbol, NDArray}(),
    aux_params::Dict{Symbol, NDArray}=Dict{Symbol, NDArray}(),
    allow_missing=false, force_init=false)

  if isinitialized(self) && !force_init
    return self
  end
  @assert isbinded(self) "Call `bind` before initialization"

  if !isdefined(self, :arg_params) || isempty(self.arg_params)
    self.arg_params = Dict(k => mx.empty(size(v)) for (k, v) in self.exec_group.arg_params)
  end

  if !isdefined(self, :aux_params) || isempty(self.aux_params)
    self.aux_params = Dict(k => mx.empty(size(v)) for (k, v) in self.exec_group.aux_params)
  end

  map([[self.arg_params, arg_params], [self.aux_params, aux_params]]) do param_arr
    dst, src = param_arr
    for (name, arr) in dst
      if isempty(src)
        init(initializer, name, arr)
      else
        src = get(src)
        if name in keys(src)
          if src[name] != arr
            copy!(arr, src[name])
          end
        else
          @assert(!allow_missing, "$name is not presented")
          init(initializer, name, arr)
        end
      end
    end
  end

  # copy the initialized parameters to devices
  set_params!(self.exec_group, self.arg_params, self.aux_params)

  self.params_dirty = false
  self.params_initialized = true

  return self
end

bind(self::SymbolModule, data_provider::AbstractDataProvider; kwargs...) = bind(self, map((x) -> x[2], provide_data(data_provider)),
  map((x) -> x[2], provide_label(data_provider)); kwargs...)
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

  # TODO propagate type information
  data_types = [Float32 for _ in 1:length(self.data_names)]
  label_types = [Float32 for _ in 1:length(self.label_names)]

  self.exec_group = DataParallelExecutorGroup(self.symbol, self.context,
                      self.data_shapes, self.data_names, data_types,
                      self.label_shapes, self.label_names, label_types,
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
  self.updater = Nullable()

  # add adequate calculation of batch_size
  op_state = OptimizationState(self.data_shapes[1][end])
  optimizer.state = op_state

  if !update_on_kvstore
    self.updater = Nullable(get_updater(optimizer))
  end
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

  self.optimizer_initialized = true

  return self
end

"""
		get_outputs

Get outputs of the previous forward computation.

# Arguments
* merge_multi_context : bool
	Default is `True`. In the case when data-parallelism is used, the outputs
	will be collected from multiple devices. A `True` value indicate that we
	should merge the collected results so that they look like from a single
	executor.

# Returns
If `merge_multi_context` is `true`, it is like `[out1, out2]`. Otherwise, it
is like `[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]`. All the output
elements are `NDArray`.
"""
function get_outputs(self::SymbolModule, merge_multi_context::Bool=true)
	@assert isbinded(self) && isinitialized(self)

  mx.get_outputs(self.exec_group, merge_multi_context)
end

"""
    forward(module, data_provider, data_batch; is_train)

Forward computation.

# Arguments
* `data_batch` : AbstractDataBatch
* `is_train` : Bool
  Default is `nothing`, which means `is_train` takes the value of `self.for_training`.
"""
forward(self::SymbolModule, data_batch::DataBatch) = forward(self, StubProvider(), data_batch, self.for_training)
function forward(self::SymbolModule, data_provider :: AbstractDataProvider, data_batch :: AbstractDataBatch, is_train::Bool)
  @assert isbinded(self) && isinitialized(self)
  mx.forward(self.exec_group, data_provider, data_batch, is_train)
end

"""
    backward(module, out_grads)
Backward computation.
# Arguments
* `out_grads` : `NDArray` or vector of `NDArray`, default `nothing`.
  Gradient on the outputs to be propagated back.
  This parameter is only needed when bind is called
  on outputs that are not a loss function.
"""
backward(self::SymbolModule, out_grads::Void=nothing) = backward(self, NDArray[])
backward(self::SymbolModule, out_grads::NDArray) = backward(self, [out_grads])
function backward(self::SymbolModule, out_grads::Vector{NDArray})
  @assert isbinded(self) && isinitialized(self)
  mx.backward(self.exec_group, out_grads)
end

"""
    update(module)
Update parameters according to the installed optimizer and the gradients computed
in the previous forward-backward batch.
"""
function update(self::SymbolModule)
  @assert isbinded(self) && isinitialized(self) && hasoptimizer(self)
  self.params_dirty = true
  update_params(self.exec_group, self.updater, self.update_on_kvstore, self.kvstore)
end

"""
		update_metric()

Evaluate and accumulate evaluation metric on outputs of the last forward computation.
# Arguments
* eval_metric : EvalMetric
* labels : Dict of NDArray
	Typically `data_batch.label`.
"""
function update_metric(self::SymbolModule, eval_metric::AbstractEvalMetric, provider::AbstractDataProvider, batch::AbstractDataBatch)
  mx.update_metric(self.exec_group, eval_metric, provider, batch)
end

"""
    get_input_grads(self::SymbolModule, merge_multi_context=true)

Get the gradients with respect to the inputs of the module.

# Arguments

* `merge_multi_context` : `Bool`
  Default is `true`. In the case when data-parallelism is used, the outputs
will be collected from multiple devices. A `true` value indicate that we
should merge the collected results so that they look like from a single
executor.

# Returns
If `merge_multi_context` is `true`, it is like `[grad1, grad2]`. Otherwise, it
is like `[[grad1_dev1, grad1_dev2], [grad2_dev1, grad2_dev2]]`. All the output
elements are `NDArray`.
"""
function get_input_grads(self::SymbolModule, merge_multi_context::Bool=true)
  @assert isbinded(self) && isinitialized(self) && self.inputs_need_grad
  return mx.get_input_grads(self.exec_group, merge_multi_context)
end

##
# Internals
##

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

_wrap_context(context::Context) = [context]
_wrap_context(context::Vector{Context}) = context
