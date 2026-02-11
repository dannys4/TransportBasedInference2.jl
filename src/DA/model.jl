export Model, SyntheticData, partition

"""
        Model

A structure to perform the twin-experiment

## Fields
- `Nx` : Dimension of the state variable
- `Ny` : Dimension of the observation variable
- `Δtdyn` : Time-step for the dynamical model
- `Δtobs` : Time step between two observations of the state
- `ϵx` : Process noise
- `ϵy` : Observation noise
- `π0` : Multivariate distribution for the initial condition
- `Tburn` : Number of steps to burn from the end of the spin up to compute the metrics
- `Tstep` : Number of steps for which the filter to be tested is applied
- `Tspinup` : Number of steps of spin-up phase,
   i.e. number of steps to generate the initial ensemble for the filtering algorithms
- `F` : State-Space Model
"""
struct Model

    "Dimension of the state variable"
    Nx::Int64

    "Dimension of the observation variable"
    Ny::Int64

    "Time-step for the dynamical model"
    Δtdyn::Float64

    "Time step between two observations of the state"
    Δtobs::Float64

    "Process noise"
    ϵx::InflationType

    "Observation noise"
    ϵy::InflationType

    "Multivariate distribution for the initial condition"
    π0::ContinuousMultivariateDistribution

    "Number of steps to burn from the end of the spin up to compute the metrics"
    Tburn::Int64
    Tstep::Int64
    Tspinup::Int64

    " State-Space Model"
    F::StateSpace
end

"""
    SyntheticData

A structure to store the synthetic data in a twin-experiment


## Fields
- `tt` : time history
- `Δt` : time step
- `x0` : the initial condition
- `xt` : history of the state
- `yt` : history of the observations
"""
struct SyntheticData
    tt::Vector{Float64}
    Δt::Float64
    x0::Vector{Float64}
    xt::Matrix{Float64}
    yt::Matrix{Float64}
end

function partition(data::SyntheticData, lengths::NTuple{N,Int}) where {N}
    sum(lengths) <= length(data.tt) || throw(ArgumentError("Unexpected lengths vector"))
    idxs = (0, cumsum(lengths)...)
    nt = fcn -> ntuple(fcn, N)
    partitions = nt(j -> (idxs[j]+1):idxs[j+1])
    tts = nt(j -> data.tt[partitions[j]])
    x0s = nt(j -> j == 1 ? data.x0 : @view(data.xt[:, idxs[j]]))
    xts = nt(j -> @view(data.xt[:, partitions[j]]))
    yts = nt(j -> @view(data.yt[:, partitions[j]]))
    SyntheticData.(tts, (data.Δt,), x0s, xts, yts)
end