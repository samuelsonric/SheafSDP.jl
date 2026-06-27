const HistoryRow{T} = @NamedTuple{μ::T, pstep::T, dstep::T, pres::T, dres::T, niter::Int}

struct History{T} <: AbstractVector{HistoryRow{T}}
    μ::Vector{T}
    pstep::Vector{T}
    dstep::Vector{T}
    pres::Vector{T}
    dres::Vector{T}
    niter::Vector{Int}
end

function History{T}() where {T}
    return History{T}(T[], T[], T[], T[], T[], Int[])
end

function Base.size(h::History)
    n = length(h.μ)
    return (n,)
end

function Base.getindex(h::History, i::Int)
    return (μ=h.μ[i], pstep=h.pstep[i], dstep=h.dstep[i], pres=h.pres[i], dres=h.dres[i], niter=h.niter[i])
end

function Base.push!(h::History, row::NamedTuple)
    push!(h.μ,     row.μ)
    push!(h.pstep, row.pstep)
    push!(h.dstep, row.dstep)
    push!(h.pres,  row.pres)
    push!(h.dres,  row.dres)
    push!(h.niter, row.niter)
    return h
end

function printrow(i::Integer, row::HistoryRow)
    println("Iter $i: μ = $(row.μ), ||rp|| = $(row.pres), ||rd|| = $(row.dres), kkt = $(row.niter)")
end
