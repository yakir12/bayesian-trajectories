using IntervalArithmetic, IntervalArithmetic.Symbols, IntervalRootFinding
using Turing, Random, GLMakie, ProgressMeter, AlgebraOfGraphics

function next_step(brwθ, crwθ, w, prev_θ)
    brwyx = sincos(brwθ) # compass error
    crwyx = sincos(prev_θ + crwθ) # motor error
    y, x = @. w*brwyx + (1 - w)*crwyx
    return atan(y, x)
end

function inext_step(crwθ, w, prev_θ, θ)
    Δtanθ(brwθ) = ((1 - w)*sin(crwθ + prev_θ) + w*sin(brwθ)) / ((1 - w)*cos(crwθ + prev_θ) + w*cos(brwθ)) - tan(θ)
    rts = IntervalRootFinding.roots(Δtanθ, interval(-pi, pi))
    filter!(==(:unique) ∘ root_status, rts)
    rts = mid.(root_region.(rts))
    if isempty(rts)
        missing
    elseif length(rts) == 1
        rts[]
    else
        minimum(abs, rts)
    end
end

using Turing
brwθ = rand(VonMises(1))
crwθ = rand(VonMises(100))
w = 0.3
prev_θ = -0.22913086189431867
θ = next_step(brwθ, crwθ, w, prev_θ)

using ApproxFun
d = (-π..π)^2
brwθf, crwθf = Fun(d)
brwy = sin.(brwθf) # compass error
brwx = cos.(brwθf) # compass error
crwy = sin.(prev_θ + crwθf) # motor error
crwx = cos.(prev_θ + crwθf) # motor error
y = w*brwy + (1 - w)*crwy
x = w*brwx + (1 - w)*crwx
α = range(-pi, pi, 100)
θf = atan.(y.(α, α'), x.(α, α'))

using GLMakie
contour(α, α, θf, levels = [θ])
scatter!(brwθ, crwθ)

# y = tan(θ)
# x = 1
# y = w*sin(brwθ) + (1 - w)*sin(prev_θ + crwθ)
# x = w*cos(brwθ) + (1 - w)*cos(prev_θ + crwθ)
#
# sin(θ) = w*sin(brwθ) + (1 - w)*sin(prev_θ + crwθ)
# cos(θ) = w*cos(brwθ) + (1 - w)*cos(prev_θ + crwθ)

# @syms θ::real brwθ::real prev_θ::real crwθ::real
# w = symbols("w", real = true, positive = true)
# eqs = [Eq(sin(θ), w*sin(brwθ) + (1 - w)*sin(prev_θ + crwθ)), Eq(cos(θ), w*cos(brwθ) + (1 - w)*cos(prev_θ + crwθ))]
# solve(eqs, brwθ)
#
# eqs = [Eq(tan(θ), w*sin(brwθ) + (1 - w)*sin(prev_θ + crwθ)), Eq(1, w*cos(brwθ) + (1 - w)*cos(prev_θ + crwθ))]
# solve(eqs..., (brwθ, crwθ))
#
# subs(eqs, θ => 0.2, w => 0.5, prev_θ => 0.23) 
#
#
# eqs = [Eq(tan(θ), w*sin(brwθ) + (1 - w)*sin(prev_θ + crwθ)), Eq(1, w*cos(brwθ) + (1 - w)*cos(prev_θ + crwθ))]
# solve(eqs..., (brwθ, crwθ))
#
# subs(eqs, θ => 0.2, w => 0.5, prev_θ => 0.23) 
#
#
#
#
# @syms θ::real brwθ::real crwθ2::real
# w = symbols("w", real = true, positive = true)
# eqs = [Eq(sin(θ), w*sin(brwθ) + (1 - w)*sin(crwθ2)), Eq(cos(θ), w*cos(brwθ) + (1 - w)*cos(crwθ2))]
# solve(eqs, brwθ)
#
# @syms brwθ::real crwθ2::real tanθ::real
# w = symbols("w", real = true, positive = true)
# brwy = sin(brwθ)
# crwy = sin(crwθ2)
# y = w*brwy + (1 - w)*crwy
# brwx = cos(brwθ)
# crwx = cos(crwθ2)
# x = w*brwx + (1 - w)*crwx
# eq = Eq(0.6, subs(y/x, crwθ2 => 0.2, w => 0.3))
# solve(eq, brwθ)
#
# @variables θ brwθ prev_θ crwθ w
# brwy = sin(brwθ)
# crwy = sin(crwθ + prev_θ)
# y = w*brwy + (1 - w)*crwy
# brwx = cos(brwθ)
# crwx = cos(crwθ + prev_θ)
# x = w*brwx + (1 - w)*crwx
# eq = substitute(y/x, Dict(crwθ => 0.2, prev_θ => 0.5, w => 0.3))
#
# using ApproxFun, DataFrames, AlgebraOfGraphics
#
# function unwrap!(x, period = π)
#     y = convert(eltype(x), period)
#     v = first(x)
#     for k = eachindex(x)
#         x[k] = v = v + rem(x[k] - v,  y, RoundNearest)
#     end
#     return x
# end
# α = range(-pi, pi, 100)
# function fun(crwθ2, w)
#     brwθ = Fun(-π .. π)
#     brwy = sin(brwθ)
#     crwy = sin(crwθ2)
#     y = w*brwy + (1 - w)*crwy
#     brwx = cos(brwθ)
#     crwx = cos(crwθ2)
#     x = w*brwx + (1 - w)*crwx
#     unwrap!(atan.(y.(α) ./ x.(α)))
# end
#
# df = DataFrame(crwθ2 = range(-pi + 1e-6, pi, 10))
# df.w .= Ref(range(1e-6, 1, 10))
# df = flatten(df, :w)
# DataFrames.transform!(df, [:crwθ2, :w] => ByRow(fun) => :θ)
# df.brwθ .= Ref(α)
# df = flatten(df, Not(:crwθ2, :w))
#
# data(df) * mapping(:brwθ, :θ, row = :w => nonnumeric => "w", col = :crwθ2 => nonnumeric => "crwθ2") * visual(Lines) |> draw()
#
# ##
# α = range(-pi, pi, 100)
# function fun(w, prev_θ)
#     d = (-π..π)^2
#     brwθ, crwθ = Fun(d)
#     brwy = sin.(brwθ) # compass error
#     brwx = cos.(brwθ) # compass error
#     crwy = sin.(prev_θ + crwθ) # motor error
#     crwx = cos.(prev_θ + crwθ) # motor error
#     y = w*brwy + (1 - w)*crwy
#     x = w*brwx + (1 - w)*crwx
#     atan.(y.(α, α'), x.(α, α'))
# end
#
# prev_θ = 1.9
# contour(α, α, fun(0.2, prev_θ), levels = [0.1])
#
# # lines(α, x.(α), label = "x")
# # lines!(α, y.(α), label = "y")
# lines(α, atan.(y.(α) ./ x.(α)))
#
#
# # θ = atan(y/x)

TN(μ, σ, m, M) = Truncated(Normal(μ, σ), m, M)
TN(μ, σ) = TN(μ, σ, -π, π)
TN(σ) = TN(zero(σ), σ)

σ_b = 1.1
σ_c = 0.1
w = 0.3
n = 20
θs = zeros(n)
b = TN(σ_b)
c = TN(σ_c)
prev_θ = 0.0
for i in eachindex(θs)
    global prev_θ
    θs[i] = next_step(rand(b), rand(c), w, prev_θ)
    prev_θ = θs[i]
end
xy = cumsum([Point2f(reverse(sincos(θ))) for θ in θs])
fig = lines(xy, axis = (;autolimitaspect = 1))
text!(xy[1], text = "start")
display(fig)

# crwθ, w, prev_θ, θ

@model function bmodel(data)
    σ_b ~ InverseGamma(2)
    σ_c ~ InverseGamma(2)
    prev_θ = data[1:end-1]
    θ = data[2:end]
    n = length(data) - 1
    crwθ ~ filldist(TN(σ_c), n)
    brwθ = inext_step.(crwθ, w, prev_θ, θ)
    for i in 1:n
        brwθ[i] ~ TN(σ_b)
    end
end

model = bmodel(θs)
chain = Turing.sample(model, Gibbs(), 10000)
# chain = sample(model, NUTS(), 1000)
# chain = Turing.sample(model, NUTS(), MCMCThreads(), 10000, 4)

describe(chain)


fig = Figure()
for (i, var_name) in enumerate(filter(x -> !contains(string(x), "θ"), chain.name_map.parameters))
    draw!(
          fig[i, 1],
          data(chain) *
          mapping(var_name; color=:chain => nonnumeric) *
          AlgebraOfGraphics.density() *
          visual(fillalpha=0)
         )
end
fig

#
#
# @model function bmodel(data)
#     σ_ϵ ~ InverseGamma(2)
#     σ_b ~ InverseGamma(2)
#     σ_c ~ InverseGamma(2)
#     prev_θs = data[1:end-1]
#     θs = data[2:end]
#     n = length(data) - 1
#     θ_b ~ filldist(TN(σ_b), n)
#     θ_c ~ filldist(TN(σ_c), n)
#     θ_μs = next_step.(θ_b, θ_c, w, prev_θs)
#     for i in eachindex(θ_μs)
#         θs[i] ~ TN(θ_μs[i], σ_ϵ)
#     end
# end
#
#
# model = bmodel(θs)
#
# chain = sample(model, Gibbs(), 10000)
#
# # chain = sample(model, NUTS(), 100)
#
#
# @model function demo(x, g)
#     k = length(unique(g))
#     a ~ filldist(Exponential(), k) # = Product(fill(Exponential(), k))
#     mu = a[g]
#     for i in eachindex(x)
#         x[i] ~ Normal(mu[i])
#     end
#     return mu
# end
# df = DataFrame(g = rand(1:15, 100))
# transform!(groupby(df, :g), :g => (g -> rand(Normal(5rand()), length(g))) => :x)
# model = demo(df.x, df.g)
# chain = sample(model, NUTS(), 10000)
# combine(groupby(df, :g), :x => mean)
#
#
# using DynamicPPL.TestUtils.AD: run_ad
# adtype = AutoForwardDiff()
# run_ad(model, adtype; test=false, benchmark=false).grad_actual
