using Debugger

# Minimal package versions for the debugger to work:
#   Juno.jl 0.7
#
# My test environment for this experiment:
#    Atom v0.8.5+
#    CodeTools v0.6.4+
#    Debugger v0.4.0
#    JuliaInterpreter v0.5.0
#    Juno v0.7.0+
#    Rebugger v0.3.1
#    Revise v2.1.2

# Installation:
#  https://discourse.julialang.org/t/ann-juno-0-8/22157

# Blog post discussing introducing the new debugger:
#   https://julialang.org/blog/2019/03/debuggers

# Some problems/issues are discussed here:
#   https://discourse.julialang.org/t/new-julia-debugger-in-juno/22359/4

# Debugging howto:
#   http://docs.junolab.org/latest/man/debugging/

# To kick off the debugger, run `Juno.@enter func(params..)`
#   e.g. for this file run `Juno.@enter problem()`


function bar()
  x = 3
  y = 7
  z = x * y
  return z + 1
end

function foo()
  a = 2
  b = 4
  c = b + a
  d = bar()
  return c + d
end


function problem()
  u = foo()
  println(u)
end
