using Statistics, LinearAlgebra, Distributions, StaticArrays

function next_step(θ, brwθ, crwθ, w)
    brwyx = sincos(brwθ) # compass error
    crwyx = sincos(θ + crwθ) # motor error
    y, x = @. w*brwyx + (1 - w)*crwyx
    return atan(y, x)
end

function get_sample_one(brwσ, crwσ, w) 
    brw = VonMises(brwσ)
    crw = VonMises(crwσ)
    return θ -> next_step(θ, rand(brw), rand(crw), w)
end

sample_one = get_sample_one(0.1, 0.1, 0.9)

@benchmark sample_one(0.0)

n = 1_000_000
θs = Vector{Float64}(undef, n)
Threads.@threads for i in 1:n
    θs[i] = sample_one(1.0)
end

hist(θs, bins = 1000, normalization = :pdf)#, axis = (; limits = ((-π, π), nothing)))
lines!(VonMises(1.0, 1.1))


function my_rand(θ, brwσ, crwσ, w)
    brw = VonMises(brwσ)
    crw = VonMises(crwσ)
    brwθ = rand(brw)
    crwθ = rand(crw)
    brwyx = sincos(brwθ) # compass error
    crwyx = sincos(θ + crwθ) # motor error
    y, x = @. w*brwyx + (1 - w)*crwyx
    return atan(y, x)
end

# For truncated normal, can use similar analysis as wrapped normal
# but without the wrapping complications
function my_logpdf_truncnorm_exact(angle, θ, brwσ, crwσ, w)
    # Similar to wrapped normal but simpler since no wrapping
    # This gives you exact normal distribution math without periodicity issues
    
    μ_y = w * 0 + (1-w) * sin(θ)
    μ_x = w * 1 + (1-w) * cos(θ)
    σ²_y = w^2 * brwσ^2 + (1-w)^2 * crwσ^2  # Simplified
    σ²_x = w^2 * brwσ^2 + (1-w)^2 * crwσ^2
    
    μ_result = atan(μ_y, μ_x)
    denom = μ_x^2 + μ_y^2
    σ²_result = (μ_x^2 * σ²_y + μ_y^2 * σ²_x) / denom^2
    σ_result = sqrt(σ²_result)
    
    # Simple normal logpdf (no wrapping needed if σ is small)
    return -(angle - μ_result)^2 / (2σ_result^2) - 0.5*log(2π*σ_result^2)
end
