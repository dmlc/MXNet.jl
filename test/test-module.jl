using MXNet
using Base.Test

# run test in the whole directory, latest modified files
# are run first, this makes waiting time shorter when writing
# or modifying unit-tests
function test_dir(dir)
  jl_files = sort(filter(x -> ismatch(r".*module\.jl$", x), readdir(dir)), by = fn -> stat(joinpath(dir,fn)).mtime)
  map(reverse(jl_files)) do file
    include("$dir/$file")
  end
end

include(joinpath(dirname(@__FILE__), "common.jl"))
@testset "Modules Test" begin
  test_dir(joinpath(dirname(@__FILE__), "unittest"))
end
