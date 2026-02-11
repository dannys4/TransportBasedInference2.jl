export Localization, Locgaspari, Locgaspari_symm, PeriodicMetric, CartesianMetric, gaspari

using SparseArrays

struct Localization{T<:Union{AbstractMatrix{Float64},LinearMap{Float64}}}
    ρX::T
end

"""
    Localization(grid, radius, metric, scale; symm_kernel, )
"""
function Localization(grid::Vector{Float64}, radius::Float64, metric::Function, scale::Real=true; Nvar::Int=1, symm_kernel=false, is_sparse=false, herm_matrix=false)
    ρX_dense = scale * Locgaspari(grid, grid, radius, metric, symm_kernel)
    ρX_sparse = is_sparse ? sparse(ρX_dense) : ρX_dense
    if Nvar > 1
        ρX_sparse = kron(ρX_sparse, ones(Nvar, Nvar))
    end
    ρX = herm_matrix ? Hermitian(ρX_sparse, :U) : ρX_sparse
    Localization(ρX)
end

Localization(Nx::Int, args...; kwargs...) = Localization(1:Nx, args...; kwargs...)

# Some metric for the distance between variables
function periodicmetric(i, j, lower, upper)
    i, j = extrema((i, j))
    dist1 = j - i
    dist2 = (upper - j) + (i - lower)
    min(dist1, dist2)
end
PeriodicMetric(d) = (i, j) -> periodicmetric(i, j, 1, d)
PeriodicMetric(lower, upper) = (i, j) -> periodicmetric(i, j, lower, upper)

cartesianmetric(i, j) = abs(i - j)
function CartesianMetric(type::Type{T}=Float64; Nvar::Int=1) where {T}
    Nvar < 1 && throw(ArgumentError("Requires positive Nvar. Got Nvar=$Nvar."))
    if T <: Int
        if Nvar > 1
            (i, j) -> cartesianmetric((i - 1) ÷ Nvar + 1, (j - 1) ÷ Nvar + 1)
        else
            cartesianmetric
        end
    else
        cartesianmetric
    end
end

function Locgaspari(gridx::AbstractVector, gridy::AbstractVector, L, metric::Function, is_symm::Bool)
    (issorted(gridx) && issorted(gridy)) || throw(ArgumentError("Expected input vectors to be sorted"))

    dx, dy = length(gridx), length(gridy)
    G = zeros(dx, dy)
    @inbounds for idx in CartesianIndices(G)
        i, j = Tuple(idx)
        is_symm && i > j && continue
        xi, yj = gridx[i], gridy[j]
        eval_ij = gaspari(metric(xi, yj) / L)
        G[i, j] = eval_ij
        is_symm && (G[j, i] = eval_ij)
    end
    G
end

Locgaspari(dx::Int, dy::Int, args...) = Locgaspari(1:dx, 1:dy, args...)

# Gaspari-Cohn kernel, Formula found in doi.org/10.1002/wcc.535, page 19
@inline g1(r) = 1 - (5 / 3) * r^2 + (5 / 8) * r^3 + (1 / 2) * r^4 - 0.25 * r^5
@inline g2(r) = 4 - 5r + (5 / 3) * r^2 + (5 / 8) * r^3 - (1 / 2) * r^4 + (1 / 12) * r^5 - (2 / 3) * r^(-1)
@inline gaspari(r) = r > 2.0 ? 0.0 : r < 1.0 ? g1(r) : g2(r)
