# using Statistics, LinearAlgebra, Distributions, StaticArrays
#
# function next_step(θ, brwθ, crwθ, w)
#     brwyx = sincos(brwθ) # compass error
#     crwyx = sincos(θ + crwθ) # motor error
#     y, x = @. w*brwyx + (1 - w)*crwyx
#     return atan(y, x)
# end
#
# function get_sample_one(brwσ, crwσ, w) 
#     brw = VonMises(brwσ)
#     crw = VonMises(crwσ)
#     return θ -> next_step(θ, rand(brw), rand(crw), w)
# end
#
# sample_one = get_sample_one(0.1, 0.1, 0.9)
#
# @benchmark sample_one(0.0)
#
# n = 1_000_000
# θs = Vector{Float64}(undef, n)
# Threads.@threads for i in 1:n
#     θs[i] = sample_one(1.0)
# end
#
# hist(θs, bins = 1000, normalization = :pdf)#, axis = (; limits = ((-π, π), nothing)))
# lines!(VonMises(1.0, 1.1))
#
#
# function my_rand(θ, brwσ, crwσ, w)
#     brw = VonMises(brwσ)
#     crw = VonMises(crwσ)
#     brwθ = rand(Xoshiro(0), brw)
#     crwθ = rand(Xoshiro(0), crw)
#     brwyx = sincos(brwθ) # compass error
#     crwyx = sincos(θ + crwθ) # motor error
#     y, x = @. w*brwyx + (1 - w)*crwyx
#     return atan(y, x)
# end
#
# using GLMakie
#
# θ = range(-π, π, 100)
# a = my_rand.(θ, 0.1, 0.2, 0.4)
# lines(θ, a)
#
#
#
# # For truncated normal, can use similar analysis as wrapped normal
# # but without the wrapping complications
# function my_logpdf_truncnorm_exact(angle, θ, brwσ, crwσ, w)
#     # Similar to wrapped normal but simpler since no wrapping
#     # This gives you exact normal distribution math without periodicity issues
#
#     μ_y = w * 0 + (1-w) * sin(θ)
#     μ_x = w * 1 + (1-w) * cos(θ)
#     σ²_y = w^2 * brwσ^2 + (1-w)^2 * crwσ^2  # Simplified
#     σ²_x = w^2 * brwσ^2 + (1-w)^2 * crwσ^2
#
#     μ_result = atan(μ_y, μ_x)
#     denom = μ_x^2 + μ_y^2
#     σ²_result = (μ_x^2 * σ²_y + μ_y^2 * σ²_x) / denom^2
#     σ_result = sqrt(σ²_result)
#
#     # Simple normal logpdf (no wrapping needed if σ is small)
#     return -(angle - μ_result)^2 / (2σ_result^2) - 0.5*log(2π*σ_result^2)
# end
#
#
# ####
#
# using Turing, GLMakie, AlgebraOfGraphics
#
#
# @model function bmodel(rolls)
#     p ~ Beta(1, 1)
#     for i in eachindex(rolls)
#         rolls[i] ~ Bernoulli(p)
#     end
# end
# p = 0.2
# d = Bernoulli(p)
# rolls = rand(d, 100)
# model = bmodel(rolls)
# chain = sample(model, NUTS(), MCMCThreads(), 10000, 20)
#
# fig = Figure()
# for (i, var_name) in enumerate(chain.name_map.parameters)
#     draw!(
#         fig[i, 1],
#         data(chain) *
#         mapping(var_name; color=:chain => nonnumeric) *
#         AlgebraOfGraphics.density() *
#         visual(fillalpha=0)
#     )
# end
#
###

using Distributions, Random, KernelDensity, Turing, GLMakie, AlgebraOfGraphics, ProgressMeter

# struct BeetleAngle <: ContinuousMultivariateDistribution 
#     brw::Truncated{Normal{Float64}, Continuous, Float64, Float64, Float64}
#     crw::Truncated{Normal{Float64}, Continuous, Float64, Float64, Float64}
#     w::Float64
#     n::Int
#     BeetleAngle(brwσ::Float64, crwσ::Float64, w::Float64, n::Int) = new(Truncated(Normal(0.0, brwσ), -π, π), Truncated(Normal(0.0, brwσ), -π, π), w, n)
# end

struct BeetleAngle <: ContinuousMultivariateDistribution 
    brw
    # brw::VonMises{Float64}
    crw
    # crw::VonMises{Float64}
    w::Float64
    n::Int
    # BeetleAngle(brwκ, crwκ, w, n::Int) = new(Normal(0.0, brwκ), Normal(0.0, crwκ), w, n)
    # BeetleAngle(brwκ, crwκ, w, n::Int) = new(Truncated(Normal(0.0, brwκ), -π, π), Truncated(Normal(0.0, crwκ), -π, π), w, n)
    BeetleAngle(brwκ, crwκ, w, n::Int) = new(VonMises(brwκ), VonMises(crwκ), w, n)
end

struct Track <: ContinuousMultivariateDistribution 
    d::BeetleAngle
    wbrwyx
    # wbrwyx::Vector{Tuple{Float64, Float64}}
    crwθ
    # crwθ::Vector{Float64}
    xs
    # xs::Vector{Float64}
    n::Int
    function Track(d, n)
        wbrwyx = [d.w .* sincos(rand(d.brw)) for _ in 1:n]
        crwθ = rand(d.crw, n)
        xs = similar(crwθ)
        new(d, wbrwyx, crwθ, xs, n)
    end
end

function Distributions.logpdf(t::Track, θs::AbstractArray{T, 1}) where T <:Real
    logp = zero(T)
    prev_θ = zero(T)
    for θ in θs
        p = transition_density(t, prev_θ, θ) 
        if p < 0
            return -Inf
        end
        logp += log(p)
        prev_θ = θ
    end
    return logp
end

function transition_density(t::Track, prev_θ, θ)
    # Threads.@threads 
    for i in 1:t.n
        crwyx = sincos(prev_θ + t.crwθ[i])
        y, x = @. t.wbrwyx[i] + (1 - t.d.w)*crwyx
        t.xs[i] = atan.(y, x)
    end
    kded = kde(t.xs,  boundary = (-π, π))
    return pdf(kded, θ)
end

Distributions.rand(rng::AbstractRNG, t::Track) = rand(rng, t.d)

function Distributions.rand(rng::AbstractRNG, d::BeetleAngle)
    θs = Vector{Float64}(undef, d.n)
    prev_θ = 0.0
    for i in 1:d.n
        brwθ = rand(rng, d.brw)
        crwθ = rand(rng, d.crw)
        θs[i] = prev_θ = next_step(brwθ, crwθ, d.w, prev_θ)
    end
    return θs
end

function next_step(brwθ, crwθ, w, prev_θ)
    brwyx = sincos(brwθ) # compass error
    crwyx = sincos(prev_θ + crwθ) # motor error
    y, x = @. w*brwyx + (1 - w)*crwyx
    return atan(y, x)
end

brwκ = 100
crwκ = 3
w = 1
d = BeetleAngle(brwκ, crwκ, w, 100)
θs = rand(Xoshiro(0), d)

xy = cumsum([Point2f(reverse(sincos(θ))) for θ in θs])
lines(xy, axis = (;autolimitaspect = 1))

n = 11
brwκs = range(brwκ - 5, brwκ + 5, n)
crwκs = range(crwκ - 2, crwκ + 2, n)
p = zeros(n, n)
h = Progress(n*n)
Threads.@threads for ij in 1:n*n
    i, j = Tuple(CartesianIndices((n , n))[ij])
    d = BeetleAngle(brwκs[i], crwκs[j], w, length(θs))
    t = Track(d, 100000)
    p[j, i] = logpdf(t, θs)
    next!(h)
end
finish!(h)

heatmap(brwκs, crwκs, p)
scatter!(brwκ, crwκ)

#
# dshfkshfdshfsf

@model function bmodel(θs)
    brwκ ~ InverseGamma(2)
    crwκ ~ InverseGamma(2)
    d = BeetleAngle(brwκ, crwκ, 0.5, length(θs))
    t = Track(d, 10000)
    Turing.@addlogprob! logpdf(t, θs)
end

d = BeetleAngle(20, 40, 0.5, 10)
θs = rand(Xoshiro(0), d)
model = bmodel(θs)

using SliceSampling

# chain = sample(model, NUTS(), MCMCThreads(), 1000, 4)
chain = sample(model, Gibbs(), 10000)

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


# using GLMakie
#
# d = BeetleAngle(0.2, 0.9, 0.1, 100)
# xy = cumsum([Point2f(reverse(sincos(θ))) for θ in rand(Xoshiro(0), d)])
# lines(xy, axis = (;aspect = DataAspect()))
#
#
# using LinearAlgebra
# xy = rand(Point2f, 1000)
# lines(xy)
# sum(norm.(diff(xy)))
#


