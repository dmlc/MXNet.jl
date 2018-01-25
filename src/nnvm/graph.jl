struct Graph
  handle::NN_GraphHandle
end

function Graph(x::SymbolicNode)
  h = Ref{MX_handle}(C_NULL)
  @mxcall(:NNGraphCreate, (MX_handle, Ref{MX_handle}), x, h)
  Graph(NN_GraphHandle(h[]))
end

Base.unsafe_convert(::Type{MX_handle}, x::Graph) =
  Base.unsafe_convert(MX_handle, x.handle)

function getsymbol(x::Graph)
  s = Ref{MX_handle}(C_NULL)
  @mxcall(:NNGraphGetSymbol, (MX_handle, Ref{MX_handle}), x, s)
  SymbolicNode(MX_SymbolHandle(s[]))
end

function apply(x::Graph, pass::Symbol)
  y = Ref{MX_handle}(C_NULL)
  @mxcall(:NNGraphApplyPasses, (MX_handle, Cuint, char_pp, Ref{MX_handle}),
          x, 1, [dump_mx_param(pass)], y)
  Graph(NN_GraphHandle(y[]))
end

function Base.getindex(x::Graph, k::Symbol)
  json = Ref{char_p}(C_NULL)
  success = Ref{Cint}(0)
  @mxcall(:NNGraphGetJSONAttr, (MX_handle, char_p, Ref{char_p}, Ref{Cint}),
          x, dump_mx_param(k), json, success)
  success[] == 0 && throw(KeyError(k))
  typ, val = JSON.parse(unsafe_string(json[]))

  if typ == "str"
    val
  else
    warn("unkown type $typ")
    typ, val
  end
end

Base.setindex!(x::Graph, v, k::Symbol) = (x[:str, k] = v)

function Base.setindex!(x::Graph, v, typ::Symbol, k::Symbol)
  s = JSON.json([typ, v])
  @show s
  @mxcall(:NNGraphSetJSONAttr, (MX_handle, Cstring, Cstring),
          x, dump_mx_param(k), s)
end

setshape!(x::Graph, shape::Tuple) = setshape!(x, [shape])

function setshape!(x::Graph, shapes::Vector)
  x[:list_shape, :shape_inputs] = shapes
  nothing
end

setdtype!(x::Graph, t::Int) = setdtype!(x, [t])

function setdtype!(x::Graph, ts::Vector)
  x[:list_int, :dtype_inputs] = ts
  nothing
end

function _set_node_attr!(x::Graph, k::Symbol, s::SymbolicNode)
  @mxcall(:NNGraphSetNodeEntryListAttr_, (MX_handle, Cstring, MX_handle),
          x, String(k), s)
end

ir(x::SymbolicNode) = graphir(Graph(x))
ir(x::Graph) = apply(x, :PrintGraphIR)[:graphir]

function gradient(y::SymbolicNode, x::SymbolicNode)
  g = Graph(y)
  _set_node_attr!(g, :grad_ys, y)
  _set_node_attr!(g, :grad_xs, x)

  # y could have multiple output
  # ny = length(list_outputs(y))
  # ∇y = [ones_like(i) for i ∈ y]
  ∇y = ones_like(y)
  _set_node_attr!(g, :grad_ys_out_grad, ∇y)

  getsymbol(apply(g, :Gradient))
end
