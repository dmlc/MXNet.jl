abstract PipelineModule <: AbstractModule

"""
    SimplePipelineModule

Allows the pipelining of several modules.

# Arguments:
* `pipeline :: Vector{Module}`
  The elements that are called sequentially

# Functionality
*
"""
type SimplePipelineModule <: PipelineModule
  pipeline :: Vector{Module}
end

type ModuleDataProvider <: mx.AbstractDataProvider
  mod :: Module
end


function forward(self :: SimplePipelineModule)
  for mod in self.pipeline
    forward(mod)
  end
end

function backward(self :: SimplePipelineModule)
  for i in length(self.pipeline):-1:1
    mod = self.pipeline[i]
    backward(mod)
  end
end

function get_outputs(self :: SimplePipelineModule)
  return get_outputs(last(self.pipeline))
end
