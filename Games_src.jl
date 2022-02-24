using Pkg
Pkg.activate(".")
Pkg.instantiate()
# using Revise
using LazySets, Optim, Plots
import LinearAlgebra: I

# STRUCTURE AND CONSTRUCTORS:

struct NormalForm
    nPlayers::Int64
    nMoves::Tuple{Vararg{Int64}}    
    payoffMatList::Tuple{ Vararg{ Array{<:Real} } }

    function NormalForm(players, nmoves, payofflist)
        if size(payofflist[1])==nmoves && players==ndims(payofflist[1])
            new(players, nmoves, payofflist)
        else
            error("Violates dimension requirements for NormalForm.")
        end
    end

end

function NormalForm(normalFormTable::Array{<:Tuple})
    nplayers=ndims(normalFormTable)
    nmoves=size(normalFormTable)

    payoffmatlist=ntuple(
        i -> [normalTable[k][i] for k in CartesianIndices(normalFormTable)],
        nplayers
    )

    return NormalForm(nplayers, nmoves, payoffmatlist)
end

#CALCULATING VALUES

function pureMinMax2(normalform::NormalForm)
    pure_mm1= minimum(
        maximum(
            normalform.payoffMatList[1];
            dims=1
        ),
        dims=2
    )
    pure_mm2= minimum(
        maximum(
            normalform.payoffMatList[2];
            dims=2
        ),
        dims=1
    )
    return [pure_mm1[1], pure_mm2[1]]
end

function brCorrelatMix(normalform::NormalForm,player::Int64,corrMix::Vector)
    payoff=normalform.payoffMatList[player]
    permutation=Tuple(
        unique( [player; 1:normalform.nPlayers] )
    )
    permutedPayoff=permutedims(payoff,permutation)
    permutedPayoff= reshape(permutedPayoff, ( size(permutedPayoff,1) , :))
    return maximum(permutedPayoff*corrMix)
end

function miniMax_i(normalform::NormalForm, player::Int64)
    uBR_i(μ) = brCorrelatMix(normalform, player, μ)
    parametrize_mix(z)=exp.(z)/ sum(exp.(z))
    obj(z) = uBR_i(parametrize_mix(z))

    z0=zeros( prod(normalform.nMoves) ÷ normalform.nMoves[player])
    solution=optimize(obj, z0)
    return solution.minimum
end

function miniMaxProfile(normalform::NormalForm)
    return [miniMax_i(normalform, i) for i=1:normalform.nPlayers]
end

function randomNormalForm(nMovesList::Tuple{Vararg{<:Real}})
    nplayers=length(nMovesList)
    payoffmatlist= Tuple(rand(Float64, nMovesList) for i in 1:nplayers)
    NormalForm(nplayers, nMovesList, payoffmatlist)
end

function greaterThanSet(minPoint::Vector{<:Real})::HPolyhedron
    HPolyhedron(
        -Matrix{Float64}(I,length(minPoint), length(minPoint)),
        -minPoint
    )
end

#PLOTTING

function plotFeasible2!(p,normalform::NormalForm)
    payoff1=normalform.payoffMatList[1]
    payoff2=normalform.payoffMatList[2]
    
    payoff_points=[
        Vector{Float64}( [payoff1[i], payoff2[i]] )
        for i in eachindex(payoff1)
    ]
    feasible = VPolygon(payoff_points)
    Plots.plot!(p,
        feasible,
        xlabel="Payoff player 1",
        ylabel="Payoff player 2",
        label="Feasible set"
    );
    
    Plots.scatter!(p,
        [payoff_points[i][1] for i in eachindex(payoff_points)],
        [payoff_points[i][2] for i in eachindex(payoff_points)],
        label="Pure payoffs"
    );
end

function plotFeasible2(normalform::NormalForm)
    p=Plots.plot(title="Normal Form Payoffs")
    plotFeasible2!(p,normalform)
    display(p)
    return p
end

function plotIR_Set2!(p::Plots.Plot, normalform::NormalForm)
    plot!(
        p,
        greaterThanSet( miniMaxProfile(normalform) ),
        label="IR Set",
        legend=:bottomleft
    )
end

function plotMinimax2!(p::Plots.Plot, normalform::NormalForm)
    mm=miniMaxProfile(normalform)
    Plots.scatter!(p,
        [mm[1]],
        [mm[2]],
        label="Minimax"
    );
end

function plotall(normalform::NormalForm)
    p=plot(
        title="Normal form payoffs"
        )
    plotFeasible2!(p,normalform)
    plotIR_Set2!(p,normalform)
    plotMinimax2!(p,normalform)
    display(p)
    return p
end

#= TO DO:
- generate functions to plot 3D
=#

#= CODE TO PLOT 3D FEASIBLE SET
Using LazySets, GLMakie, Polyhedra

test=CH(
    Singleton([1.,1.,3.]), 
    Singleton([2.,1.,4.])
)

test=CH(test,
    Singleton([2., 1.5, 3.])    
)

figureplot3d(
    overapproximate(test)
) =#
