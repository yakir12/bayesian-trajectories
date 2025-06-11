using Turing, Random, KernelDensity, GLMakie, ProgressMeter

struct CustomDistribution <: ContinuousUnivariateDistribution
    a
end

Distributions.rand(rng::AbstractRNG, d::CustomDistribution) = rand(rng, VonMises(d.a))

function Distributions.logpdf(d::CustomDistribution, x::T) where T <:Real

    n = 1000
    p = zeros(8)
    for _ in 1:n
        θ = rand(Random.default_rng(), d)

    kded = kde(xs, boundary = (-π, π))
    return log(pdf(kded, x))
end

a = 2
d = CustomDistribution(a)
xs = [rand(Random.default_rng(), d) for _ in 1:100]
n = 21
as = range(a - 1, a + 1, n)
p = zeros(n)
h = Progress(n)
Threads.@threads for i in 1:n
    di = CustomDistribution(as[i])
    p[i] = sum(x -> logpdf(di, x), xs)
    next!(h)
end
finish!(h)

keep = findall(!isinf, p)

fig = lines(as[keep], p[keep])
vlines!(a)
display(fig)


@model function bmodel(xs)
    a ~ InverseGamma(2)
    for i in eachindex(xs)
        xs[i] ~ CustomDistribution(a)
    end
end
model = bmodel(xs)
chain = sample(model, Gibbs(), 1000)
describe(chain)

