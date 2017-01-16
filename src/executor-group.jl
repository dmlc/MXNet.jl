"""
    AbstractExecutorGroup
Executor group is a convenient tool for managing a group of executors.
"""
abstract AbstractExecutorGroup

function forward(self::AbstractExecutorGroup, data_provider :: AbstractDataProvider,
                 data_batch :: AbstractDataBatch, is_train)
  throw(MethodError(forward, (self, )))
end

type DataParallelExecutorGroup <: AbstractExecutorGroup
  symbol :: SymbolicNode
  context :: Vector{Context}
  execs :: Vector{Executor}

  data_shapes :: Dict{Symbol, Tuple{Vararg{Int}}}
  label_shapes :: Dict{Symbol, Tuple{Vararg{Int}}}
  for_training :: Bool

  shared_group :: Nullable{DataParallelExecutorGroup}
  inputs_need_grad :: Bool
  fixed_param_names :: Nullable{Vector{Symbol}}
  grad_req :: Dict{Symbol, GRAD_REQ}
  freeze_idx

  data_arrays :: Vector{Vector{SlicedNDArray}}
  label_arrays :: Vector{Vector{SlicedNDArray}}
  param_arrays :: Vector{Vector{NDArray}}
  grad_arrays :: Vector{Vector{NDArray}}
  aux_arrays :: Vector{Vector{NDArray}}
  input_grad_arrays :: Vector{Vector{NDArray}}

  arg_params :: Dict{Symbol, NDArray}
  aux_params :: Dict{Symbol, NDArray}
end
function DataParallelExecutorGroup(symbol::SymbolicNode, context::Vector{Context},
           data_shapes, data_names, label_shapes, label_names, for_training, inputs_need_grad,
           shared_group, fixed_param_names, grad_req)

  num_dev = length(context)
  arg_names  = list_arguments(symbol)
  input_names = [data_names; label_names]
  param_names  = setdiff(arg_names, input_names)
  aux_names = list_auxiliary_states(symbol) 

  batch_size = data_shapes[1][end]
  for shape in data_shapes
    @assert batch_size == shape[end]
  end
  if !isempty(label_shapes)
    for shape in label_shapes
      @assert batch_size == shape[end]
    end
  end

  # TODO imlplement workload
  slices = _split_inputs(batch_size, num_dev)

  execs = Vector{Executor}(num_dev)

  provided_shapes = merge(Dict(name => shape for (name, shape) in zip(data_names, data_shapes)),
                          Dict(name => shape for (name, shape) in zip(label_names, label_shapes)))
  arg_shapes, out_shapes, aux_shapes = infer_shape(symbol; provided_shapes...)
  @assert(!isa(arg_shapes, Void), "Information not enough to perform complete shape inference")

  grad_req, freeze_idx = get_grads(symbol, param_names, arg_names, data_names, inputs_need_grad, fixed_param_names, grad_req)

  arg_params = Dict{Symbol, NDArray}()
  aux_params = Dict{Symbol, NDArray}()

  for (name, shape) in filter(x -> in(x[1], param_names), zip(arg_names, arg_shapes))
    arg_params[name] = empty(shape)
  end

  for (name, shape) in zip(aux_names, aux_shapes)
    aux_params[name] = empty(shape)
  end

  for i = 1:num_dev
    data_shapes = Dict(k => tuple(v[1:end-1]...,length(slices[i])) for (k,v) in zip(data_names, data_shapes))
    label_shapes = Dict(k => tuple(v[1:end-1]...,length(slices[i])) for (k,v) in zip(label_names, label_shapes))
    arg_arrays = NDArray[zeros(shape, context[i]) for shape in arg_shapes]
    grad_arrays = Dict{Symbol,NDArray}()
    aux_arrays = NDArray[zeros(shape, context[i]) for shape in aux_shapes]

    shapes = zip(arg_names, arg_shapes)

    # if not in provided data, should be parameters
    if inputs_need_grad
      provided_data_names = label_names
    else
      provided_data_names = [data_names; label_names]
    end
    shapes = filter(x -> !in(x[1], provided_data_names), shapes)

    # Remove all gradients for nop params
    shapes = filter(x -> grad_req[x[1]] != GRAD_NOP, shapes)

    for (name, shape) in shapes
      grad_arrays[name] = zeros(shape, context[i])
    end

    execs[i] = bind(symbol, context[i], arg_arrays, args_grad=grad_arrays, grad_req=grad_req, aux_states=aux_arrays)
    #= dbg_str = mx.debug_str(train_execs[i]) =#
    #= info(string("TempSpace: ", split(dbg_str, ['\n'])[end-2]..., " on ", self.ctx[i])) =#
  end

  # TODO: perform type inference

  # set up input data structures
  data_arrays  = [SlicedNDArray[(slices[i], exec.arg_dict[name]) for (i,exec) in enumerate(execs)] for name in data_names]
  label_arrays = [SlicedNDArray[(slices[i], exec.arg_dict[name]) for (i,exec) in enumerate(execs)] for name in label_names]

  param_idx    = filter(i -> in(arg_names[i], param_names), 1:length(arg_names))
  name_idx     = filter(i -> in(arg_names[i], data_names), 1:length(arg_names))

  param_arrays = [NDArray[exec.arg_arrays[i] for exec in execs] for i in param_idx]
  grad_arrays  = [NDArray[exec.grad_arrays[i] for exec in execs] for i in param_idx]
  aux_arrays   = [NDArray[exec.aux_arrays[i] for exec in execs] for i = 1:length(aux_names)]

  if inputs_need_grad
    input_grad_arrays = [NDArray[exec.grad_arrays[i] for exec in execs] for i in name_idx]
  else
    input_grad_arrays = []
  end

  return DataParallelExecutorGroup(
    symbol, context, execs,
    data_shapes, label_shapes, for_training,
    shared_group, inputs_need_grad, fixed_param_names, grad_req, freeze_idx,
    data_arrays, label_arrays, param_arrays, grad_arrays, aux_arrays,
    input_grad_arrays, arg_params, aux_params)
end

"""
    forward(exec_group, data_batch, is_train)
Split `data_batch` according to workload and run forward on each devices.
# Arguments
* `data_batch` : AbstractDataBatch
* `is_train` : Nullable{Bool}
  The hint for the backend, indicating whether we are during training phase.
  Default is `nothing`, then the value `self.for_training` will be used.
"""
function forward(self:: DataParallelExecutorGroup, data_provider :: AbstractDataProvider, data_batch :: AbstractDataBatch, is_train = nothing)

  load_data!(data_provider, data_batch, self.data_arrays)
  is_train = get(is_train, self.for_training)
  
  if is_train && !isempty(get_label(data_provider, data_batch))
    load_label!(data_provider, data_batch, self.label_arrays)
  end

  for exec in self.execs
    forward(exec, is_train=is_train)
  end
   # TODO add callbacks here
end

"""
		set_params!(self::DataParallelExecutorGroup, arg_params, aux_params)

Assign, i.e. copy parameters to all the executors.
# Arguments
* `arg_params` : Dict{Symbol, NDArray}
  A dictionary of name to `NDArray` parameter mapping.
* `aux_params` : Dict{Symbol, NDArray}
  A dictionary of name to `NDArray` auxiliary variable mapping.
"""
function set_params!(self::DataParallelExecutorGroup,
                    arg_params, aux_params; allow_extra_params::Bool = false)
  for exec in self.execs
    copy_params_from(exec, arg_params, aux_params, allow_extra_params=allow_extra_params)
  end
end

##
# Internals
##


function output_shapes(self:: DataParallelExecutorGroup)
  #= outputs = [size(out) for out in self.execs[1].outputs] =#
  #= return [tuple(key, shape) for key, shape in zip(list_outputs(exec_group.symbol), outputs)] =#
end

function get_grads(symbol, param_names, arg_names, data_names, inputs_need_grad, fixed_param_names, grad_req)
  if isnull(fixed_param_names)
    # get grad attribute to allow for freezing
    fixed_param_names = Symbol[]
    for (attr, value) in list_all_attr(symbol)
      sattr = string(attr)
      if endswith(sattr, "grad") && value == "freeze"
        push!(fixed_param_names, Symbol(sattr[1:end-5]))
      end
    end
  else
    fixed_param_names = get(fixed_param_names)
  end

  # Needs to correspond to the correct id in the update loop layer idx=1:length(param_names).
  freeze_idx = filter(i -> in(param_names[i], fixed_param_names), 1:length(param_names))

  # Setup grad_req as a dictionary
  grad_req_dict = Dict{Symbol, GRAD_REQ}()
  for param in arg_names
    if param in param_names
      if in(param, fixed_param_names)
        grad_req_dict[param] = GRAD_NOP
      else
        grad_req_dict[param] = grad_req
      end
    elseif param in data_names
      if inputs_need_grad
        grad_req_dict[param] = grad_req
      else
        grad_req_dict[param] = GRAD_NOP
      end
    else
      grad_req_dict[param] = GRAD_NOP
    end
  end

  return grad_req_dict, freeze_idx
end
