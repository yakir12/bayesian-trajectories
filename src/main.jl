using Distributions, StaticArrays, GLMakie, AlgebraOfGraphics, Turing

const SV = SVector{2, Float64}

d = VonMises(0.3, 5)
n = 100

xys = Vector{SV}(undef, n)
xys[1] = zero(SV)
for i in 2:n
    xys[i] = xys[i-1] + SV(reverse(sincos(rand(d))))
end
lines(xys, axis = (; aspect = DataAspect()))

@model function bmodel(xs)
    μ ~ VonMises(0, 0.1)
    κ ~ InverseGamma(3, 5)
    Δs = diff(xs)
    αs = [atan(reverse(Δ)...) for Δ in Δs]
    for i in 1:length(αs)
        αs[i] ~ VonMises(μ, κ)
    end
end

chain = sample(bmodel(xys), NUTS(), 10_000, progress=false)

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

