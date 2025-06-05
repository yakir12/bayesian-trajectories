using Turing, GLMakie, AlgebraOfGraphics

h(t, h₀, g) = h₀ - g/2*t^2

n = 100
h₀ = 25
g = 9.8
ts = range(0, sqrt(2h₀/g), n)
σ² = 3
noise = Normal(0, sqrt(σ²))
hs = h.(ts, h₀, g) .+ rand(noise, n)
hs .= max.(hs, 0)

lines(ts, hs)


@model function bmodel(ts, hs)
    g ~ InverseGamma(2, 3)
    h₀ ~ Uniform(0, 100)
    σ² ~ InverseGamma(2, 3)
    μ = h₀ .- g/2*ts.^2 
    for i in eachindex(hs)
        hs[i] ~ Normal(μ[i], sqrt(σ²))
    end
end
model = bmodel(ts, hs)
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
