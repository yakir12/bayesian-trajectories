# Broken
using Turing, GLMakie, AlgebraOfGraphics, StaticArrays

const SV = SVector{2, Float64}

n = 100
crwκ = 2
crw = VonMises(crwκ)
steps = accumulate(1:n, init = (zero(SV), 0.0)) do (xy, θ), _
    crwθ = rand(crw)
    crwyx = sincos(θ - crwθ)
    y, x = crwyx
    return (SV(x, y), atan(y, x))
end

xy = cumsum(first.(steps))
θ = last.(steps)

lines(xy, axis = (;aspect = DataAspect()))



@model function bmodel(Δ)
    crwκ ~ InverseGamma(2, 3)
    for i in eachindex(Δ)
        Δ[i] ~ VonMises(crwκ)
    end
end

model = bmodel(diff(θ))
chain = sample(model, externalsampler(RandPermGibbs(SliceSteppingOut(2.0))), 1000)

chain = sample(model, NUTS(), 1000)

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

