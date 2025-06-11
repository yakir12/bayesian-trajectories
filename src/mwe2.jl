using Turing, Random, GLMakie, ProgressMeter, AlgebraOfGraphics

function next_step(brwθ, crwθ, w, prev_θ)
    brwyx = sincos(brwθ) # compass error
    crwyx = sincos(prev_θ + crwθ) # motor error
    y, x = @. w*brwyx + (1 - w)*crwyx
    return atan(y, x)
end

TN(μ, σ, m, M) = Truncated(Normal(μ, σ), m, M)
TN(μ, σ) = TN(μ, σ, -π, π)
TN(σ) = TN(zero(σ), σ)

σ_ϵ = 0.01
σ_b = 1.1
σ_c = 0.1
w = 0.5
n = 20
θs = zeros(n)
b = TN(σ_b)
c = TN(σ_c)
prev_θ = 0.0
for i in eachindex(θs)
    θ_μ = next_step(rand(b), rand(c), w, prev_θ)
    θs[i] = rand(TN(θ_μ, σ_ϵ))
    prev_θ = θs[i]
end
xy = cumsum([Point2f(reverse(sincos(θ))) for θ in θs])
lines(xy, axis = (;autolimitaspect = 1))
text!(xy[1], text = "start")

@model function bmodel(data)
    σ_ϵ ~ InverseGamma(2)
    σ_b ~ InverseGamma(2)
    σ_c ~ InverseGamma(2)
    prev_θ = zero(eltype(data))
    n = length(data)
    θ_b ~ filldist(TN(σ_b), n)
    θ_c ~ filldist(TN(σ_c), n)
    for i in eachindex(data)
        θ_μ = next_step(θ_b[i], θ_c[i], w, prev_θ)
        prev_θ = data[i]
        data[i] ~ TN(θ_μ, σ_ϵ)
    end
end

model = bmodel(θs)
# chain = sample(model, Gibbs(), 10000)
# chain = sample(model, NUTS(), 1000)
chain = sample(model, NUTS(), MCMCThreads(), 10000, 4)

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



@model function bmodel(data)
    σ_ϵ ~ InverseGamma(2)
    σ_b ~ InverseGamma(2)
    σ_c ~ InverseGamma(2)
    prev_θs = data[1:end-1]
    θs = data[2:end]
    n = length(data) - 1
    θ_b ~ filldist(TN(σ_b), n)
    θ_c ~ filldist(TN(σ_c), n)
    θ_μs = next_step.(θ_b, θ_c, w, prev_θs)
    for i in eachindex(θ_μs)
        θs[i] ~ TN(θ_μs[i], σ_ϵ)
    end
end


model = bmodel(θs)

chain = sample(model, Gibbs(), 10000)

# chain = sample(model, NUTS(), 100)


@model function demo(x, g)
    k = length(unique(g))
    a ~ filldist(Exponential(), k) # = Product(fill(Exponential(), k))
    mu = a[g]
    for i in eachindex(x)
        x[i] ~ Normal(mu[i])
    end
    return mu
end
df = DataFrame(g = rand(1:15, 100))
transform!(groupby(df, :g), :g => (g -> rand(Normal(5rand()), length(g))) => :x)
model = demo(df.x, df.g)
chain = sample(model, NUTS(), 10000)
combine(groupby(df, :g), :x => mean)


using DynamicPPL.TestUtils.AD: run_ad
adtype = AutoForwardDiff()
run_ad(model, adtype; test=false, benchmark=false).grad_actual
