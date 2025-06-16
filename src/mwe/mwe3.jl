using IntervalArithmetic, IntervalArithmetic.Symbols, IntervalRootFinding
using Turing, Random, GLMakie, ProgressMeter, AlgebraOfGraphics
# using ApproxFun
# import ApproxFun:(..)

function next_step(brwθ, crwθ, w, prev_θ)
    brwyx = sincos(brwθ) # compass error
    crwyx = sincos(prev_θ + crwθ) # motor error
    y, x = @. w*brwyx + (1 - w)*crwyx
    return atan(y, x)
end

function inext_step(crwθ, w, prev_θ, θ)
    Δθ(brwθ) = atan(((1 - w)*sin(crwθ + prev_θ) + w*sin(brwθ)) / ((1 - w)*cos(crwθ + prev_θ) + w*cos(brwθ))) - θ
    rts = IntervalRootFinding.roots(Δθ, interval(-pi, pi))
    filter!(==(:unique) ∘ root_status, rts)
    rts = mid.(root_region.(rts))
    if isempty(rts)
        # @show "missing"
        missing
    elseif length(rts) == 1
        rts[]
    else
        _, i = findmin(abs, rts)
        rts[i]
    end
end

# ###
#
# crwθ = 3.
# w = 0.5
# prev_θ = 0.2
# θ = 0.3
# Δθ(brwθ) = atan(((1 - w)*sin(crwθ + prev_θ) + w*sin(brwθ)) / ((1 - w)*cos(crwθ + prev_θ) + w*cos(brwθ))) - θ
# brwθ = range(-pi, pi, 100)
# y = Δθ.(brwθ)
# lines(brwθ, y)
# hlines!(0)
#
# ###

TN(μ, σ, m, M) = Truncated(Normal(μ, σ), m, M)
TN(μ, σ) = TN(μ, σ, -π, π)
TN(σ) = TN(zero(σ), σ)
σ_b = 1
σ_c = 0.1
w = 0.2
n = 1000
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


function eachsstep(crw, w, θs)
    n = length(θs)
    brwθs = Vector{Float64}(undef, n)
    crwθs = Vector{Float64}(undef, n)
    θs2 = pushfirst!(θs, 0.0)
    p = Progress(n)
    Threads.@threads for i in 2:n
        crwθ = rand(crw)
        brwθ = inext_step(crwθ, w, θs2[i - 1], θs2[i])
        while ismissing(brwθ)
            crwθ = rand(crw)
            brwθ = inext_step(crwθ, w, θs2[i - 1], θs2[i])
        end
        brwθs[i] = brwθ
        crwθs[i] = crwθ
        next!(p)
    end
    finish!(p)
    return (brwθs, crwθs)
end

σ_cf = 0.2
crw = TN(σ_cf)

brwθs, crwθs = eachsstep(crw, w, θs)

fig = Figure()
ax1 = Axis(fig[1,1], title = "brwθs")
hist!(ax1, brwθs)
ax2 = Axis(fig[1,2], title = "crwθs")
hist!(ax2, crwθs)
linkaxes!(ax1, ax2)

mean(brwθs)
std(brwθs)
mean(crwθs)
std(crwθs)
