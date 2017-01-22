import ....MXNet: mx # in order to use mx.

@defstruct SequentialModuleMetas (
  take_labels :: Bool = false,
  auto_wiring :: Bool = false
)

"""
    SequentialModule

A SequentialModule is a container module that can chain multiple modules together.
Note building a computation graph with this kind of imperative container is less
flexible and less efficient than the symbolic graph. So this should be only used as a
handy utility.

# Parameters

"""
type SequentialModule <: AbstractModule
  modules :: Vector{AbstractModule}
  metas :: Vector{SequentialModuleMetas}

  binded :: Bool
  for_training :: Bool
  inputs_need_grad :: Bool
  params_initialized :: Bool
  optimizer_initialized :: Bool

  label_names :: Vector{Symbol}
  label_shapes :: Vector{Tuple{Vararg{Int}}}
  function SequentialModule()
    new(Vector{AbstractModule}(),
        Vector{Symbol}[],
        false, false, false, false, false)
  end
end

### default API
isbinded(self::SequentialModule) = self.binded
allows_training(self::SequentialModule) = self.for_training
isinitialized(self::SequentialModule) = self.params_initialized
hasoptimizer(self::SequentialModule) = self.optimizer_initialized

data_names(self::SequentialModule) = length(self.modules) > 0 ? data_names(self.modules[1]) : Symbol[]
label_names(self::SequentialModule) = self.label_names
output_names(self::SequentialModule) = length(self.modules) > 0 ? output_names(self.modules[end]) : Symbol[]

"""
    add(self, module; kwargs...)

Add a module to the chain.
# Arguments
* `module` : AbstractModule
  The new module to add.
* `kwargs` : keywords
  All the keyword arguments are saved as meta information
  for the added module. The currently known meta includes
  * `:take_labels`: indicating whether the module expect to
  take labels when doing computation. Note any module in
  the chain can take labels (not necessarily only the top
  most one), and they all take the same labels passed
  from the original data batch for the `SequentialModule`.
  * `:auto_wiring`: TODO...

# Returns

This function returns `self` to allow us to easily chain a
series of `add` calls.

# Examples
An example of addinging two modules to a chain::
```julia
seq_mod = @mx.chain mx.Module.SequentialModule() =>
          add(mod1) => 
          add(mod2)
```
"""
function add(self::SequentialModule, mod::AbstractModule; kwargs...)
  push!(self.modules, mod)

  metas = SequentialModuleMetas(;kwargs...)
  for (key, _) in (kwargs...)
    @assert(key ∈ fieldnames(metas), "Unknown meta '$key', a typo?")
  end
  push!(self.metas, metas)

  # after adding new modules, we are reset back to raw states, needs
  # to bind, init_params, etc.
  self.binded = false
  self.params_initialized = false
  self.optimizer_initialized = false

  return self # for easier chaining
end

function data_shapes(self::SequentialModule)
  !isbinded(self) && return Dict{Symbol, Vector{Tuple{Int}}}()
  return data_shapes(self.modules[1])
end

function label_shapes(self::SequentialModule)
  !isbinded(self) && return Dict{Symbol, Vector{Tuple{Int}}}()
  return self.label_shapes
end

function output_shapes(self::SequentialModule)
  !isbinded(self) && return Dict{Symbol, Vector{Tuple{Int}}}()
  return output_shapes(self.modules[end])
end

function get_params(self::SequentialModule)
  @assert isbinded(self) && isinitialized(self)

  reduce((Dict{Symbol, NDArray}(), Dict{Symbol, NDArray}()), self.modules) do acc, mod
    arg, aux = get_params(mod)
    merge(acc[1], arg)
    merge(acc[2], aux)
  end
end

"""
"""
function init_params(self::SequentialModule, opts::ModuleInitParamsOptions)
  if isinitialized(self) && !opts.force_init
    return self
  end

  @assert(isbinded(self), "call bind before initializing the parameters")
  # make sure we do not have duplicated parameter names
  arg_dict = Dict()
  aux_dict = Dict()
  duplicates = false
  for (i, mod) in enumerate(self.modules)
    arg_params, aux_params = get_params(mod)
    map((arg_dict, arg_params), (aux_dict, aux_params)) do arg
      dict, params = arg
      for name in keys(params)
        if haskey(dict, name)
          info("Name $name in layer $i ($(typeof(mod))) is already used in layer $(dict[name][1])($(typeof(dict[name][2])))")
          duplicates = true
        else
          dict[name] = (i, typeof(mod))
        end
      end
    end
  end
  if duplicates
    error("Duplicates in layer names")
  end

  for mod in self.modules
    init_params(mod, opts)
  end

  self.params_initialized = true

  return self
end

"""
"""
function bind(self::SequentialModule, data_shapes, label_shapes, opts::ModuleBindOptions)
  info("SequentialModule: label_shapes=$label_shapes")
  if opts.inputs_need_grad
    @assert opts.for_training
  end
  if opts.shared_module !== nothing
    info("Shared module is not supported")
  end
  @assert(length(self.modules) > 0, "Attempting to bind empty SequentialModule")

  # the same label shapes are used for all chained modules
  self.label_shapes = label_shapes

  module_data_shapes = data_shapes
  anybody_ever_needs_label = false
  for (i, mod) in enumerate(self.modules)
    meta = self.metas[i]
    if meta.take_labels
      module_label_shapes = label_shapes
      anybody_ever_needs_label = true
    else
      module_label_shapes = Tuple{Int}[]
    end
    
    module_inputs_need_grad = opts.inputs_need_grad || (opts.for_training && i > 1)
    
    if meta.auto_wiring
      data_names = data_names(mod)
      @assert length(module_data_shapes) == length(data_names)
      module_data_shapes = [(new_name, shape) for (new_name, (_, shape)) in zip(data_names, module_data_shapes)]
    end

    bind(mod, module_data_shapes, module_label_shapes,
         for_training=opts.for_training, inputs_need_grad=module_inputs_need_grad,
         force_rebind=opts.force_rebind, shared_module=nothing, grad_req=opts.grad_req)
    
    # the output of the previous module is the data of the next module
    module_data_shapes = output_shapes(mod)
  end

  if !anybody_ever_needs_label
    # then I do not need label either
    self.label_shapes = Tuple{Int}[]
  end
  self.binded = true

  return self
end

"""
"""
function init_optimizer(self::SequentialModule; optimizer::AbstractOptimizer=ADAM(), kvstore :: Union{Base.Symbol, KVStore}=:local, force_init :: Bool=false)
  @assert isbinded(self) && isinitialized(self)
  if hasoptimizer(self) && !force_init
    warn("Optimizer already initialized, ignoring.")
  end
  for mod in self.modules
    init_optimizer(mod, optimizer=optimizer, kvstore=kvstore, force_init=force_init)
  end

  self.optimizer_initialized = true
  return self
end

"""
"""
function get_outputs(self::SequentialModule, merge_multi_context::Bool=true)
  @assert isbinded(self) && isinitialized(self)
  return get_outputs(last(self.modules), merge_multi_context)
end

"""
"""
function forward(self::SequentialModule, data_provider :: AbstractDataProvider, data_batch :: AbstractDataBatch, is_train::Bool=self.for_training)
  @assert isbinded(self) && isinitialized(self)

  batch = DataBatch(data_provider, data_batch)
  for (i, mod) in enumerate(self.modules)
    forward(mod, batch, is_train)
    batch.data = get_outputs(mod)
  end
end

"""
"""
function backward(self::SequentialModule, out_grads::Vector{NDArray})
  @assert isbinded(self) && isinitialized(self)

  for (i, mod) in reverse(zip(1:length(self.modules), self.modules)) 
    backward(mod, out_grads)
    if i == 1
      break
    end

    out_grads = get_input_grads(mod)
  end
end

"""
"""
function update(self::SequentialModule)
  @assert isbinded(self) && isinitialized(self) && hasoptimizer(self)

  for mod in self.modules
    update(mod)
  end
end

"""
"""
function update_metric(self::SequentialModule, eval_metric::AbstractEvalMetric, provider::AbstractDataProvider, batch::AbstractDataBatch)
  @assert isbinded(self) && isinitialized(self)
  for (meta, mod) in zip(self.metas, self.modules)
    if meta.take_labels
      update_metric(mod, eval_metric, provider, batch)
    end
  end
end

"""
"""
function get_input_grads(self::SequentialModule, merge_multi_context::Bool=true)
  @assert isbinded(self) && isinitialized(self) && self.inputs_need_grad

  return get_input_grads(self.modules[1], merge_multi_context)
end
