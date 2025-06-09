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
    brwθ = rand(Xoshiro(0), brw)
    crwθ = rand(Xoshiro(0), crw)
    brwyx = sincos(brwθ) # compass error
    crwyx = sincos(θ + crwθ) # motor error
    y, x = @. w*brwyx + (1 - w)*crwyx
    return atan(y, x)
end

using GLMakie

θ = range(-π, π, 100)
a = my_rand.(θ, 0.1, 0.2, 0.4)
lines(θ, a)



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


####

using Turing, GLMakie, AlgebraOfGraphics


@model function bmodel(rolls)
    p ~ Beta(1, 1)
    for i in eachindex(rolls)
        rolls[i] ~ Bernoulli(p)
    end
end
p = 0.2
d = Bernoulli(p)
rolls = rand(d, 100)
model = bmodel(rolls)
chain = sample(model, NUTS(), MCMCThreads(), 10000, 20)

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

###

using Distributions, Random

struct BeetleAngle <: ContinuousMultivariateDistribution 
    brw::Truncated{Normal{Float64}, Continuous, Float64, Float64, Float64}
    crw::Truncated{Normal{Float64}, Continuous, Float64, Float64, Float64}
    w::Float64
    n::Int
    BeetleAngle(brwσ::Float64, crwσ::Float64, w::Float64, n::Int) = new(Truncated(Normal(0.0, brwσ), -π, π), Truncated(Normal(0.0, brwσ), -π, π), w, n)
end

function next_step(θ, brwθ, crwθ, w)
    brwyx = sincos(brwθ) # compass error
    crwyx = sincos(θ + crwθ) # motor error
    y, x = @. w*brwyx + (1 - w)*crwyx
    return atan(y, x)
end

function Distributions.rand(rng::AbstractRNG, d::BeetleAngle)
    θs = Vector{Float64}(undef, d.n)
    prev_θ = 0.0
    for i in 1:d.n
        brwθ = rand(rng, d.brw)
        crwθ = rand(rng, d.crw)
        θs[i] = prev_θ = next_step(prev_θ, brwθ, crwθ, d.w)
    end
    return θs
end


d = BeetleAngle(0.2, 0.4, 0.9, 10)
θs = rand(Xoshiro(0), d)

prev_θ = 0.0
n = 10000000
xs = Vector{Float64}(undef, n)
Threads.@threads for i in 1:n
    xs[i] = next_step(prev_θ, rand(d.brw), rand(d.crw), d.w)
end
hist(xs, bins = 1000)




function Distributions.logpdf(d::BeetleAngle, θs::Vector{Float64})
    if length(θs) != d.n
        return -Inf
    end
   
    logp = 0.0
    prev_θ = 0.0
   
    for i in 1:d.n
        θ_curr = θs[i]
       
        # Compute the log density for this transition step
        step_logp = log_transition_density(d, prev_θ, θ_curr)
       
        if step_logp == -Inf
            return -Inf
        end
       
        logp += step_logp
        prev_θ = θ_curr
    end
   
    return logp
end






function log_transition_density(d::BeetleAngle, θ_prev::Float64, θ_curr::Float64)
    # The transition density is the integral over all (brwθ, crwθ) that produce θ_curr
    # Given the constraint: θ_curr = atan(Y, X) where
    # Y = w*sin(brwθ) + (1-w)*sin(θ_prev + crwθ)
    # X = w*cos(brwθ) + (1-w)*cos(θ_prev + crwθ)
   
    # Transform to Cartesian coordinates for the target
    target_y = sin(θ_curr)
    target_x = cos(θ_curr)
   
    # Use adaptive Monte Carlo integration
    n_samples = 50000
    tolerance = 1e-4
   
    total_density = 0.0
    valid_samples = 0
   
    for _ in 1:n_samples
        # Sample from the noise distributions
        brwθ = rand(d.brw)
        crwθ = rand(d.crw)
       
        # Compute the predicted Cartesian coordinates
        brw_y, brw_x = sincos(brwθ)
        crw_y, crw_x = sincos(θ_prev + crwθ)
       
        pred_y = d.w * brw_y + (1 - d.w) * crw_y
        pred_x = d.w * brw_x + (1 - d.w) * crw_x
       
        # Check if this sample is close to the target (within tolerance)
        dist_sq = (pred_y - target_y)^2 + (pred_x - target_x)^2
       
        if dist_sq < tolerance^2
            # This sample contributes to the density
            brw_density = pdf(d.brw, brwθ)
            crw_density = pdf(d.crw, crwθ)
            joint_density = brw_density * crw_density
           
            # Weight by the Jacobian of the transformation
            jacobian = compute_transformation_jacobian(θ_prev, brwθ, crwθ, d.w)
           
            total_density += joint_density / jacobian
            valid_samples += 1
        end
    end
   
    if valid_samples == 0
        return -Inf
    end
   
    # Normalize by the area of the tolerance region and number of samples
    # The tolerance region has area π * tolerance^2
    area_factor = π * tolerance^2
    avg_density = total_density / valid_samples
   
    # The density is the average density times the probability of being in the tolerance region
    final_density = avg_density * (valid_samples / n_samples) / area_factor
   
    return final_density > 0 ? log(final_density) : -Inf
end

function compute_transformation_jacobian(θ_prev::Float64, brwθ::Float64, crwθ::Float64, w::Float64)
    # Compute the Jacobian of the transformation from (brwθ, crwθ) to (Y, X)
    # where Y = w*sin(brwθ) + (1-w)*sin(θ_prev + crwθ)
    #       X = w*cos(brwθ) + (1-w)*cos(θ_prev + crwθ)
   
    # Partial derivatives
    # ∂Y/∂brwθ = w*cos(brwθ)
    # ∂Y/∂crwθ = (1-w)*cos(θ_prev + crwθ)
    # ∂X/∂brwθ = -w*sin(brwθ)  
    # ∂X/∂crwθ = -(1-w)*sin(θ_prev + crwθ)
   
    dY_dbrw = w * cos(brwθ)
    dY_dcrw = (1 - w) * cos(θ_prev + crwθ)
    dX_dbrw = -w * sin(brwθ)
    dX_dcrw = -(1 - w) * sin(θ_prev + crwθ)
   
    # Jacobian determinant for the 2×2 matrix
    jacobian_det = abs(dY_dbrw * dX_dcrw - dY_dcrw * dX_dbrw)
   
    return max(jacobian_det, 1e-12)  # Avoid division by zero
end



@model function bmodel(xys)
    brwσ ~ InverseGamma(2)
    crwσ ~ InverseGamma(2)
    w ~ Beta(1, 1)
    xys ~ BeetleAngle(brwσ, crwσ, w, length(xys))
end



d = BeetleAngle(0.2, 0.4, 0.9, 10)
θs = rand(Xoshiro(0), d)

logpdf(d, θs)


using GLMakie

d = BeetleAngle(0.2, 0.9, 0.1, 100)
xy = cumsum([Point2f(reverse(sincos(θ))) for θ in rand(Xoshiro(0), d)])
lines(xy, axis = (;aspect = DataAspect()))


using LinearAlgebra
xy = rand(Point2f, 1000)
lines(xy)
sum(norm.(diff(xy)))

