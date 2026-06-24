const HistoryRow{T} = @NamedTuple{μ::T, τp::T, τd::T, rp::T, rd::T, kkt_iters::Int}

struct History{T} <: AbstractVector{HistoryRow{T}}
    μ::Vector{T}
    τp::Vector{T}
    τd::Vector{T}
    rp::Vector{T}
    rd::Vector{T}
    kkt_iters::Vector{Int}
end

function History{T}() where {T}
    return History{T}(T[], T[], T[], T[], T[], Int[])
end

Base.size(h::History) = (length(h.μ),)

function Base.getindex(h::History, i::Int)
    return (μ=h.μ[i], τp=h.τp[i], τd=h.τd[i], rp=h.rp[i], rd=h.rd[i], kkt_iters=h.kkt_iters[i])
end

function Base.push!(h::History, row::NamedTuple)
    push!(h.μ, row.μ)
    push!(h.τp, row.τp)
    push!(h.τd, row.τd)
    push!(h.rp, row.rp)
    push!(h.rd, row.rd)
    push!(h.kkt_iters, row.kkt_iters)
    return h
end

function printrow(i::Int, row::HistoryRow)
    println("Iter $i: μ = $(row.μ), ||rp|| = $(row.rp), ||rd|| = $(row.rd), kkt = $(row.kkt_iters)")
end
