using Distributions, StaticArrays, GLMakie, AlgebraOfGraphics, Turing

const SV = SVector{2, Float64}

d = VonMises(0.3, 5)
n = 10000

xys = Vector{SV}(undef, n)
xys[1] = zero(SV)
for i in 2:n
    xys[i] = xys[i-1] + SV(reverse(sincos(rand(d))))
end
lines(xys, axis = (; aspect = DataAspect()))

Δs = diff(xys)
αs = [atan(reverse(Δ)...) for Δ in Δs]
hist(αs)


@model function bmodel(αs)
    μ ~ VonMises(0.3, 1)
    κ ~ InverseGamma(10, 55)
    for i in eachindex(αs)
        αs[i] ~ VonMises(μ, κ)
    end
end

model = bmodel(αs)

chain = sample(model, Prior(), MCMCThreads(), 1000, 4)

chain = sample(model, NUTS(), MCMCThreads(), 1000, 4)

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

# MWE issue #2584
using Turing
@model function bmodel(αs)
    μ ~ Uniform(-π, π)
    κ ~ InverseGamma(2, 3)
    for i in eachindex(αs)
        αs[i] ~ VonMises(μ, κ) # doesn't work
        # αs[i] ~ Normal(μ, κ) # works
    end
end
x = rand(VonMises(0.3, 3), 100)
model = bmodel(x)
chain = sample(model, NUTS(), 1000)

# simplified MWE

using Turing
@model function bmodel(α)
    μ ~ Uniform(-π, π)
    κ ~ InverseGamma(2, 3)
    α ~ VonMises(μ, κ) # doesn't work
    # α ~ Normal(μ, κ) # works
end
x = rand(VonMises(0.3, 3))
model = bmodel(x)
chain = sample(model, NUTS(), 1000)

