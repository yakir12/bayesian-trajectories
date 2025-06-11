using Turing, Random, Bijectors
using GLMakie, ProgressMeter, AlgebraOfGraphics

TN(μ, σ, m, M) = Truncated(Normal(μ, σ), m, M)
TN(μ, σ) = TN(μ, σ, -π, π)
TN(σ) = TN(zero(σ), σ)

struct CustomDistribution <: ContinuousUnivariateDistribution
    dist

    CustomDistribution(σ) = new(TN(σ))
end

Distributions.rand(rng::AbstractRNG, d::CustomDistribution) = rand(rng, d.dist)
# Distributions.rand(rng::AbstractRNG, d::CustomDistribution) = rand(rng, VonMises(d.a))

const nleds = 8
θ2index(θ) = mod(round(Int, nleds*θ/2π), nleds) + 1 

function Distributions.logpdf(d::CustomDistribution, x::T) where T <:Real
    target = θ2index(x)
    n = 1000
    p = Threads.Atomic{Int}(0)
    Threads.@threads for _ in 1:n
        θ = rand(Random.default_rng(), d)
        i = θ2index(θ)
        if i == target
            Threads.atomic_add!(p, 1)
        end
    end
    return log(p[]/n)
end

Bijectors.bijector(d::CustomDistribution) = Logit(-π, π)
Distributions.minimum(d::CustomDistribution) = -π
Distributions.maximum(d::CustomDistribution) = π

σ = 2
d = CustomDistribution(σ)
xs = [rand(Random.default_rng(), d) for _ in 1:100]

n = 81
σs = range(σ - 1.9, σ + 3, n)
p = zeros(n)
h = Progress(n)
Threads.@threads for i in 1:n
    di = CustomDistribution(σs[i])
    p[i] = sum(x -> logpdf(di, x), xs)
    next!(h)
end
finish!(h)

keep = findall(!isinf, p)

fig = lines(σs[keep], p[keep])
vlines!(σ)
display(fig)


@model function bmodel(xs)
    σ² ~ InverseGamma(2)
    for i in eachindex(xs)
        xs[i] ~ CustomDistribution(sqrt(σ²))
    end
end
model = bmodel(xs)
chain = sample(model, NUTS(), 100)

# chain = sample(model, Gibbs(), 1000)
describe(chain)

fig = Figure()
for (i, var_name) in enumerate(chain.name_map.parameters)
    draw!(
          fig[i, 1],
          data(chain) *
          mapping(var_name; color=:chain => nonnumeric) *
          AlgebraOfGraphics.density() *
          visual(fillalpha=0)
         )
end
fig
