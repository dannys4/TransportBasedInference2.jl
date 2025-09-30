export Localization, Locgaspari, Locgaspari_symm, periodicmetric, PeriodicMetric, cartesianmetric, gaspari

using SparseArrays

struct Localization{T<:Union{AbstractMatrix{Float64},LinearMap{Float64}}}
    ρX::T
end

function Localization(Nx::Int, L, metric::Function; kernel=nothing, is_sparse=false, is_herm=true)
    isnothing(kernel) && (kernel = is_herm ? Locgaspari_symm : Locgaspari)
    ρX_dense = kernel((Nx, Nx), L, metric)
    ρX_sparse = is_sparse ? sparse(ρX_dense) : ρX_dense
    ρX = is_herm ? Hermitian(ρX_sparse, :U) : ρX_sparse
    Localization(ρX)
end

# function Locgaspari(d, L, periodic::Bool)
#
#     G = zeros(d,d)
#     @inbounds for i=1:d
#         # Check if the domain is periodic
#         if periodic == true
#             @inbounds for j=i:d
#                 rdom = min(abs(j - i), abs(-d + (j-1) - i), abs(d + j - i))
#                 G[i,j] = gaspari(rdom/L)
#             end
#         else
#             @inbounds for j=i:d
#             rdom = abs(j - i)
#             G[i,j] = gaspari(rdom/L)
#             end
#         end
#
#     end
#     return Symmetric(G)
# end

# Some metric for the distance between variables
function periodicmetric(i, j, d)
    dist1 = abs(j - i)
    dist2 = abs((j - d) - i)
    dist3 = abs(j - (i - d))
    min(dist1, dist2, dist3)
end
PeriodicMetric(d) = (i, j) -> periodicmetric(i, j, d)

cartesianmetric(i, j) = abs(i - j)

# Construct a possibly non-symm localisation matrix using
# the Gaspari-Cohn kernel
function Locgaspari(d::NTuple{2,Int}, L, metric::Function)
    dx, dy = d
    G = zeros(dx, dy)
    @inbounds for i = 1:dx, j = 1:dy
        G[i, j] = gaspari(metric(i, j) / L)
    end
    return G
end

# Construct a possibly non-symm localisation matrix using
# the Gaspari-Cohn kernel
function Locgaspari_symm(dx::NTuple{2,Int}, L, metric::Function)
    Nx = dx[1]
    @assert dx[2] == Nx
    G = zeros(Nx, Nx)
    @inbounds for i = 1:Nx, j = i:Nx
        G[i, j] = gaspari(abs(metric(i, j) / L))
    end
    return G
end

# Caspari-Cohn kernel, Formula found in Data assimilation in the geosciences:
# An overview of methods, issues, and perspectives
@inline g1(r) = 1 - (5 / 3) * r^2 + (5 / 8) * r^3 + (1 / 2) * r^4 - 0.25 * r^5
@inline g2(r) = 4 - 5r + (5 / 3) * r^2 + (5 / 8) * r^3 - (1 / 2) * r^4 + (1 / 12) * r^5 - (2 / 3) * r^(-1)
@inline gaspari(r) = r > 2.0 ? 0.0 : r < 1.0 ? g1(r) : g2(r)
