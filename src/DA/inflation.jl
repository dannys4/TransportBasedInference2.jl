
import Statistics: mean, cov


export InflationType, IdentityInflation, AdditiveInflation, RelativeAdditiveInflation,
    MultiplicativeInflation, MultiAddInflation,
    exactn, has_nonzero_mean, get_cov



"""
        exactn(N)

A function to create a 1D sample with exactly mean 0 and covariance 1
(The samples are no longer i.i.d but this can be usueful when the initialization of the problem is challenging.)
"""
function exactn(N)
    a = deepcopy(randn(N))
    return (a .- mean(a)) ./ std(a)
end


"""
#     InflationType
#
# An abstract type for Inflation.
"""
abstract type InflationType end

"""
    IdentityInflation <: InflationType


An type to store identity inflation :

Define additive inflation: x <- x
"""
struct IdentityInflation <: InflationType
end


"""
        (A::IdentityInflation)(X)

Apply an `IdentityInflation` `A` on an ensemble matrix `X`, i.e. xⁱ -> xⁱ
"""
function (A::IdentityInflation)(X)
    nothing
end

"""
        AdditiveInflation <: InflationType

An type to store additive inflation :

Define additive inflation: x <- x + ϵ with ϵ a random vector
drawn from the distribution α

## Fields:
$(TYPEDFIELDS)

## Constructors
- `AdditiveInflation(Nx::Int64, α::ContinuousMultivariateDistribution)`
- `AdditiveInflation(Nx::Int64)`
- `AdditiveInflation(Nx::Int64, m::Array{Float64,1}, Σ::Union{Array{Float64,2}, Diagonal{Float64}})`
- `AdditiveInflation(Nx::Int64, m::Array{Float64,1}, σ::Array{Float64,1})`
- `AdditiveInflation(Nx::Int64, m::Array{Float64,1}, σ::Float64)`
- `AdditiveInflation(m::Array{Float64,1}, σ::Float64)`
"""

struct AdditiveInflation{MeanT<:Union{AbstractVector{<:Real},Nothing},CovT,StdT} <: InflationType
    "Dimension of the state vector"
    Nx::Int64

    "Mean of the additive inflation"
    m::MeanT

    "Covariance of the additive inflation"
    Σ::CovT

    "Square-root of the covariance matrix"
    σ::StdT
end

# Some convenient constructors for multivariate Gaussian distributions
# By default, the distribution of the additive inflation α is a multivariate
# normal distribution with zero mean and identity as the covariance matrix
@inline AdditiveInflation(Nx::Int64) = AdditiveInflation(Nx, zeros(Nx), Diagonal(ones(Nx)), Diagonal(ones(Nx)))

function AdditiveInflation(Nx::Int64, m::Array{Float64,1}, Σ::Union{Array{Float64,2},Diagonal{Float64}})
    @assert Nx == size(m, 1) "Error dimension of the mean"
    @assert Nx == size(Σ, 1) == size(Σ, 2) "Error dimension of the covariance matrix"

    return AdditiveInflation(Nx, m, Σ, sqrt(Σ))

end

function AdditiveInflation(Nx::Int64, m::Array{Float64,1}, σ::Array{Float64,1})
    @assert Nx == size(m, 1) "Error dimension of the mean"
    @assert Nx == size(σ, 1) "Error dimension of the std vector"

    return AdditiveInflation(Nx, m, Diagonal(σ .^ 2), Diagonal(σ))

end

function AdditiveInflation(Nx::Int64, m::Array{Float64,1}, σ::Float64)
    @assert Nx == size(m, 1) "Error dimension of the mean"

    return AdditiveInflation(Nx, m, σ^2 * I, σ * I)

end

function AdditiveInflation(m::Array{Float64,1}, σ::Float64)
    Nx = size(m, 1)
    return AdditiveInflation(Nx, m, σ^2 * I, σ * I)
end

function AdditiveInflation(Nx::Int, σ::Float64)
    return AdditiveInflation(Nx, nothing, σ^2 * I, σ * I)
end

Base.size(A::AdditiveInflation) = A.Nx
mean(A::AdditiveInflation) = A.m
cov(A::AdditiveInflation) = A.Σ

"""
        (A::AdditiveInflation)(x::Array{Float64,1})

Apply the additive inflation `A` to the vector `x`,
i.e. x -> x + ϵ with ϵ ∼ `A.α`.
"""
function (A::AdditiveInflation)(x::T, t::Float64, perturb::Bool = false, rand_workspace::Union{Nothing,T} = nothing) where {T<:AbstractVecOrMat{Float64}}
    @assert size(x, 1) == A.Nx "Expected the first dimension of x to be of size $(A.Nx)"
    if isnothing(rand_workspace)
        rand_workspace = randn(size(x)...)
    else
        randn!(rand_workspace)
    end
    if has_nonzero_mean(A)
        if perturb
            x .-= A.m
        else
            x .+= A.m
        end
    end
    mul!(x, A.σ, rand_workspace, true, true)
    x
end

has_nonzero_mean(A::AdditiveInflation) = !(isnothing(A.m) || all(iszero, A.m))

get_cov(A::AdditiveInflation, t = nothing) = A.Σ

raw"""
    ``X_t \sim \mathcal{N}(m, (\sigma + diag(offsets))^2)``
"""
struct RelativeAdditiveInflation{MeanT<:Union{AbstractVector{<:Real},Nothing},StdT<:AbstractMatrix{<:Real}} <: InflationType
    "Dimension of the state vector"
    Nx::Int64

    "Mean of the additive inflation"
    m::MeanT

    "Times at which offset is recorded"
    times::Vector{Float64}

    "offset for which we create the inflation"
    offsets::Matrix{Float64}

    "Scaling of the offsets for the inflation"
    offsets_scale::Float64

    "Square-root of the covariance matrix"
    σ_base::StdT
    function RelativeAdditiveInflation(Nx, m::MT, T::Int, sigma_base::Union{<:Real, <:AbstractVector{<:Real}, <:AbstractMatrix{<:Real}}, offsets_scale::Float64) where {MT}
        times = fill(Float64(Inf), T)
        offsets = Matrix{Float64}(undef, Nx, T)
        σ_base = nothing
        if sigma_base isa Real
            σ_base = sigma_base * I(Nx)
        elseif sigma_base isa AbstractVector
            length(sigma_base) == Nx || throw(ArgumentError("Expected length(sigma_base) == $Nx, got $(length(sigma_base))"))
            σ_base = Diagonal(sigma_base)
        else
            σ_base = sigma_base
        end
        return new{MT, typeof(σ_base)}(Nx, m, times, offsets, offsets_scale, σ_base)
    end
end

has_nonzero_mean(A::RelativeAdditiveInflation) = !(isnothing(A.m) || all(iszero, A.m))

function (A::RelativeAdditiveInflation)(x::T, t::Float64, perturb::Bool = false, rand_workspace::Union{Nothing,T} = nothing) where {T<:AbstractVecOrMat{Float64}}
    @assert size(x, 1) == A.Nx "Expected first dimension of x to be $(A.Nx)"
    _,which_time = findmin(i-> abs(t - i), A.times)
    std_t = A.offsets_scale * Diagonal(@view(A.offsets[:,which_time])) + A.σ_base
    if has_nonzero_mean(A)
        if perturb
            x .-= A.m
        else
            x .+= A.m
        end
    end
    if isnothing(rand_workspace)
        rand_workspace = randn(size(x)...)
    else
        randn!(rand_workspace)
    end
    mul!(x, std_t, rand_workspace, true, true)
    return x
end

function get_cov(A::RelativeAdditiveInflation, t)
    _,which_time = findmin(i-> abs(t - i), A.times)
    std_t = A.offsets_scale * Diagonal(@view(A.offsets[:,which_time])) + A.σ_base
    std_t * std_t'
end

"""
    MultiplicativeInflation <: InflationType

An type to store multiplicative inflation :

xⁱ -> xⁱ + β*(xⁱ - x̄) with β a scalar

# Fields:
- 'β' : multiplicative inflation factor
"""
struct MultiplicativeInflation <: InflationType
    "Multiplicative inflation factor β"
    β::Real
end

"""
        (A::MultiplicativeInflation)(X, start::Int64, final::Int64)


Apply the multiplicative inflation `A` to the lines `start` to `final` of an ensemble matrix `X`,
i.e. xⁱ -> x̄ + β*(xⁱ - x̄)  with β scalar, usually ∼ 1.0.
"""
function (A::MultiplicativeInflation)(X, start::Int64, final::Int64, t::Float64, perturb::Bool)
    Ne = size(X, 2)
    X̂ = copy(mean(view(X, start:final, :), dims=2)[:, 1])
    @inbounds for i = 1:Ne
        col = view(X, start:final, i)
        col .= (1.0 - A.β) * X̂ + A.β * col
    end
end


"""
        (A::MultiplicativeInflation)(X)


Apply the multiplicative inflation `A` to an ensemble matrix `X`,
i.e. xⁱ -> x̄ + β*(xⁱ - x̄)  with β scalar, usually ∼ 1.0.
"""
(A::MultiplicativeInflation)(X, t, perturb=false) = A(X, 1, size(X, 1), t, perturb)


"""
    MultiAddInflation <: InflationType


An type to store multiplico-additive inflation :

Define multiplico-additive inflation: xⁱ -> x̄ + β*(xⁱ - x̄)  + ϵⁱ with ϵⁱ ∼ α and β a scalar

## Fields:
- `Nx` : dimension of the vector
- 'β' : Multiplicative inflation factor
- 'α' : Distribution of the additive inflation

## Constructors:
- `MultiAddInflation(Nx::Int64, β::Real, α::ContinuousMultivariateDistribution)`
- `MultiAddInflation(Nx::Int64, β::Float64, m::Array{Float64,1}, Σ)`
- `MultiAddInflation(Nx::Int64, β::Float64, m::Array{Float64,1}, σ::Array{Float64,1})`
- `MultiAddInflation(Nx::Int64, β::Float64, m::Array{Float64,1}, σ::Float64)`
"""
struct MultiAddInflation{Mu<:Union{Vector{Float64},Nothing}, T} <: InflationType
    "Dimension of the state vector"
    Nx::Int64

    "Multiplicative inflation factor β"
    β::Real

    "Mean of the additive inflation"
    m::Mu

    "Covariance of the additive inflation"
    Σ::T

    "Square-root of the covariance matrix"
    σ::T
end

# Some convenient constructors for multivariate Gaussian additive distributions
# By default, for a Multiplico-additive inflation, the multiplicative inflation
# factor β is set to 1.0, and  α is a  multivariate
# normal distribution with zero mean and identity as the covariance matrix
function MultiAddInflation(Nx::Int)
    return MultiAddInflation(Nx, 1.0, nothing, Diagonal(ones(Nx)), Diagonal(ones(Nx)))
end


function MultiAddInflation(Nx::Int64, β::Float64, m::Vector{Float64}, Σ)
    @assert β > 0.0 "The multiplicative inflation must be >0.0"
    @assert Nx == size(m, 1) "Error dimension of the mean"
    @assert Nx == size(Σ, 1) == size(Σ, 2) "Error dimension of the covariance matrix"

    return MultiAddInflation(Nx, β, m, Σ, sqrt(Σ))
end

function MultiAddInflation(Nx::Int64, β::Float64, m::Vector{Float64}, σ::Vector{Float64})
    @assert β > 0.0 "The multiplicative inflation must be >0.0"
    @assert Nx == size(m, 1) "Error dimension of the mean"
    @assert Nx == size(σ, 1) "Error dimension of the std vector"

    return MultiAddInflation(Nx, β, m, Diagonal(σ .^ 2), Diagonal(σ))
end

function MultiAddInflation(Nx::Int64, β::Float64, m::Vector{Float64}, σ::Float64)
    @assert β > 0.0 "The multiplicative inflation must be >0.0"
    @assert Nx == size(m, 1) "Error dimension of the mean"

    return MultiAddInflation(Nx, β, m, σ^2*I, σ*I)
end

"""
    size(A::MultiAddInflation)

Return the dimension of the additive inflation of `A`.
"""
Base.size(A::MultiAddInflation) = A.Nx
mean(A::MultiAddInflation) = isnothing(A.m) ? zeros(A.Nx) : A.m
cov(A::MultiAddInflation) = A.Σ

@inline function scale_noise!(A::MultiAddInflation, noise::AbstractArray)
    if A.σ isa UniformScaling
        noise .*= A.σ.λ
    else
        noise .= A.σ * noise.σ
    end
end

"""
        (A::MultiAddInflation)(X, start::Int64, final::Int64)


Apply the multiplicat inflation `A` to the lines `start` to `final` of an ensemble matrix `X`,
i.e. xⁱ -> x̄ + β*(xⁱ - x̄)  + ϵⁱ with ϵⁱ ∼ `A.α` and β a scalar, usually ∼ 1.0.
"""
function (A::MultiAddInflation)(X::AbstractMatrix{T}, start::Int64, final::Int64, t::Float64, perturb::Bool = false) where {T}
    # @assert A.Nx == final - start + 1 "Dimension does not match"
    Ne = size(X, 2)
    # X̂ = copy(mean(view(X, start:final,:), dims = 2)[:,1])
    μX = vec(mean(@view(X[start:final, :]), dims=2))
    rand_space = Vector{T}(undef, final-start + 1)
    @inbounds for i = 1:Ne
        randn!(rand_space)
        scale_noise!(A, rand_space)
        col = @view X[start:final, i]
        # col .= A.β * (col - μX) + μX + A.m
        for col_idx in eachindex(col)
            m_val = isnothing(A.m) ? 0. : A.m[col_idx]
            if perturb
                rmul!(m_val, -1)
            end
            col_val = col[col_idx]
            col[col_idx] = muladd(A.β, col_val - μX[col_idx], μX[col_idx] + m_val + A.σ*rand_space[col_idx])
        end
    end
end

"""
        (A::MultiAddInflation)(X, start::Int64, final::Int64)


Apply the multiplico-additive inflation `A` to the ensemble matrix `X`,
i.e. xⁱ -> x̄ + β*(xⁱ - x̄)  + ϵⁱ with ϵⁱ ∼ `A.α` and β a scalar, usually ∼ 1.0.
"""
(A::MultiAddInflation)(X, t) = A(X, 1, size(X, 1), t)
