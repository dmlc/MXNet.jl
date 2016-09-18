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

Based on the underlyin API a high-level API is implemented:
* [`fit`](@ref):
* [`predict`](@ref):
* [`score`](@ref):
"""
abstract AbstractModule

function isbinded(self::AbstractModule)
  throw(MethodError(isbinded, (self,)))
end

function allows_training(self::AbstractModule)
  throw(MethodError(allows_training, (self,)))
end

function isinitialized(self::AbstractModule)
  throw(MethodError(isinitialized, (self,)))
end

function hasoptimizer(self::AbstractModule)
  throw(MethodError(hasoptimizer, (self,)))
end


function forward_backward(self :: AbstractModule, data_batch)
  forward(self, is_train=true)
  backward(self)
end

function score(self :: AbstractModule, eval_data, eval_metric, num_batch=nothing, batch_end_callback=nothing, reset=true, epoch=0)
  @assert isbinded(self) && isinitialized(self)

  reset && reset!(eval_data)
  reset!(eval_metric)

  for (nbatch, eval_batch) in enumerate(eval_data)
    if num_batch !== nothing && nbatch == num_back
      break
    end

    forward(self, eval_batch, is_train=false)
    update_metric(self, eval_metric, label(eval_batch))

    if batch_end_callback !== nothing
      error("Not implemented yet!")
    end
  end
  get(eval_metric)
end

function iter_predict(self :: AbstractModule, eval_data, num_batch=nothing, reset=true)
  @assert isbinded(self) && isinitialized(self)

  reset && reset!(eval_data)

  for (nbatch, eval_batch) in enumerate(eval_data)
    if num_batch !== nothing && nbatch == num_back
      break
    end
    forward(self, eval_batch, is_train=false)
    samples = count_samples(eval_batch)
end
