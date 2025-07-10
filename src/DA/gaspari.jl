export Localization, Locgaspari, Locgaspari_symm, periodicmetric!, periodicmetric, cartesianmetric, gaspari

using SparseArrays

struct Localization{T<:AbstractMatrix{Float64}}
    ρX::T
end

function Localization(Nx::Int, L, metric::Function; kernel=Locgaspari_symm, is_sparse=false, is_herm=true)
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

periodicmetric!(i, j, d) = min(abs(j - i), abs(-d + j - i), abs(d + j - i))
periodicmetric(d) = (i, j) -> periodicmetric!(i, j, d)

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
