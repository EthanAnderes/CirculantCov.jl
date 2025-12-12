module CirculantCov

using Dierckx: Spline1D 
using ApproxFun: Fun, Jacobi
using FastTransforms: jac2cheb
using FFTW: plan_fft
using LinearAlgebra: mul!
using AssociatedLegendrePolynomials: λlm

# J
# ==================================================

"""
Jperm(ℓ::Int, n::Int) return the column number in the J matrix U^2
where U is unitary FFT. The J matrix looks like this:

|1   0|
|  / 1|
| / / |
|0 1  |

"""
function Jperm end


function Jperm(ℓ::Int, n::Int)
    @assert 1 <= ℓ <= n
    ℓ==1 ? 1 : n - ℓ + 2
end

# Quadrature pixel area element for GCP grids
# ==================================================

function ΔΩ(θvec::AbstractVector{T}, nφ; lmax=(length(θvec)+1)÷2) where T

    Λ    = λlm(0:lmax, 0, cos.(θvec)) # ℓ indexes the columns
    b    = zeros(T, lmax+1)
    b[1] = √(4π) / nφ

    return Λ' \ b

end


# geometric methods with angles and geodesics and periodize
# ==================================================

function periodize(f::Vector{T}, freq_mult::Int) where {T}
    n = length(f)
    nfm = n÷freq_mult
    @assert nfm == n//freq_mult
    f′ = sum( circshift(f, k*nfm) for k=0:freq_mult-1)
    f′[1:nfm]
end

function in_0_2π(φ::Real)
    pφ, p2π = promote(φ, 2*π) # handles the case φ=0 for in_0_2π
    mod(pφ, p2π)
end
# special case so that in_0_2π(Float32(2π)) ≈ Float32(2π)
in_0_2π(φ::Float32) = mod(φ, Float32(2π)) 
# 
# ... equiv to ...
# 
# The if/else is introduced so that we always return a value in [0,2π)
# function in_0_2π(φ::T) where T 
#     # rtn = rem2pi(1*φ, RoundDown) # the 1* fixes the error that rem2pi doesn't take irrational arguments
#     rtn = mod2pi(1*φ) # the 1* fixes the error that rem2pi doesn't take irrational arguments
#     if rtn ≈ 2π
#         return zero(T)
#     else
#         return rtn
#     end
# end

# The if/else is introduced so that we always return a value in [-π,π)
function in_negπ_π(φ::T) where T 
    rtn = rem2pi(1*φ, RoundNearest) # the 1* fixes the error that rem2pi doesn't take irrational arguments
    if rtn ≈ π
        return -rtn
    else
        return rtn
    end
end


counterclock_Δφ(φstart, φstop) = in_0_2π(φstop - φstart)
#
# ... equiv to ...
#
# function counterclock_Δφ(φ_start, φ_stop)
#     Δφ  = in_negπ_π(φ_stop) - in_negπ_π(φ_start)
#     Δφ′ =  Δφ < 0 ? 2π + Δφ : Δφ
#     return in_0_2π(Δφ′)
# end



function φ2φspan(φ)
    Δφpix  = counterclock_Δφ(φ[1], φ[2])
    Δφspan = counterclock_Δφ(φ[1], φ[end]) + Δφpix
    # The extra Δφpix makes Δφspan measure the (angular) distance between the 
    # left boundary of the starting pixel and the right boundary of the ending pixel 
    φspan = (φ[1], φ[1] + Δφspan)
    return φspan
end

function fullcircle(φ::AbstractVector)
    φspan    = φ2φspan(φ)
    φs₁, φs₂ = in_0_2π.(φspan)
    Δφspan   = φs₁ ≈ φs₂ ? 2π : counterclock_Δφ(φs₁, φs₂)    
    
    freq_mult_float = 2π / Δφspan 
    freq_mult       = round(Int, freq_mult_float) # div(2π, Δφspan, RoundNearest)
    @assert isapprox(freq_mult, freq_mult_float, rtol=1e-6)

    nφ2π = length(φ)*freq_mult
    φ2π  = @. in_0_2π(φ[1] + 2π * (0:nφ2π-1) / nφ2π) 
    return φ2π, freq_mult
end


# This formula for angle_separation is the geodesic formula simplified 
# for two points (θ₁, φ₁) & (θ₂, φ₂) on the sphere at the equator
# where Δφ = φ₂ - φ₁, θ₁ = θ₂ = π/2
angle_separation(Δφ)  = 2asin(abs(sin(Δφ/2)))


"""
`geoβ(θ₁, θ₂, φ₁, φ₂)` -> Geodesic between two spherical points at 
(θ₁, φ₁) and (θ₂, φ₂).
"""
function geoβ(θ₁, θ₂, φ₁, φ₂)
    sΔθ½, sΔφ½ = sin((θ₁ - θ₂)/2), sin((φ₁ - φ₂)/2)
    2asin(√(sΔθ½^2 + sin(θ₁)*sin(θ₂) * sΔφ½^2))    
end

"""
`cosgeoβ(θ₁, θ₂, φ₁, φ₂)` -> Cosine of the geodesic between two spherical points 
at (θ₁, φ₁) and (θ₂, φ₂).
"""
function cosgeoβ(θ₁, θ₂, φ₁, φ₂)
    cos(θ₁-θ₂) - sin(θ₁)*sin(θ₂)*(1-cos(φ₁-φ₂))/2
end


"""
`βsingrid(ngrid, period)` creats a grid of points on [0,period) ⊂ [0,π)
with extra density near the endpoints
"""
βsingrid(ngrid, period)   = shift_scale_sin.(range(0,π,length=ngrid), period)


"""
`shift_scale_sin(β,period)` internal function used in `βsingrid`
"""
shift_scale_sin(β,period) = period * (sin(π*β/period - π/2) + 1) / 2




# Methods for generating some useful polar grids
# ================================================

function θ_healpix_j_Nside(j_Nside) 
    0 < j_Nside < 1  ? acos(1-abs2(j_Nside)/3)      :
    1 ≤ j_Nside ≤ 3  ? acos(2*(2-j_Nside)/3)        :
    3 < j_Nside < 4  ? acos(-(1-abs2(4-j_Nside)/3)) : 
    error("argument ∉ (0,4)")
end

θ_healpix(Nside) = θ_healpix_j_Nside.((1:4Nside-1)/Nside)

function θ_grid(;θspan::NTuple{2,Real}, N::Int, type=:equiθ)
    @assert N > 0
    @assert 0 < θspan[1] < θspan[2] < π

    if type==:equiθ
        θgrid″ = range(θspan[1], θspan[2], length=N+2)
    elseif type==:equicosθ
        θgrid″ = acos.(range(cos(θspan[2]), cos(θspan[1]), length=N+2)[end:-1:1])
    elseif type==:healpix
        @warn """
            When `type` argument is set to `:healpix` the parameter `N` corresponds 
            to Healpix `Nside`, _not_ the number of θ grid points within the interval 
            specified by `θspan` as it does when `type ∈ {:equiθ, :equicosθ}`.
            """
        θgrid′ = θ_healpix(N)
        θgrid″ = θgrid′[θspan[1] .≤ θgrid′ .≤ θspan[2]]
    else
        error("`type` argument variable is not a valid option. Choose from `type ∈ {:equiθ, :equicosθ, :healpix}`")
    end 

    # θgrid″ subsets θgrid′ to be within θspan
    # δ½south″ and δ½north″ are the arclength midpoints to the adjacent pixel
    δ½south″ = (circshift(θgrid″,-1)  .- θgrid″) ./ 2
    δ½north″ = (θgrid″ .- circshift(θgrid″,1)) ./ 2   
    
    # now restrict to the interior of the range of θgrid″
    θ       = θgrid″[2:end-1]
    δ½south = δ½south″[2:end-1]
    δ½north = δ½north″[2:end-1]

    # These are the pixel boundaries along polar
    # so length(θ∂) == length(θ)+1
    θ∂ = vcat(θ[1] .- δ½north[1], θ .+ δ½south)

    return θ, θ∂
end 

"""
`φ_grid(;φspan::Tuple{Real,Real}, N::Int)` specifies a uniform grid on a 
contiguous interval of azumimuth. `∂φstart` begins the interval. Moving counter 
clockwise (looking down from the north pole) to `∂φstop`. 
Only integer fractions are allowed and both `∂φstart`, `∂φstop` must be `≥ 0`.

Note: `(∂φstart, ∂φstop) = (5.3, 1.0) ≡ (5.3, 1.0 + 2π)`
"""
function φ_grid(;φspan::Tuple{T1, T2}, N::Int) where {T1<:Real, T2<:Real}
    ∂φstart′, ∂φstop′ = promote(in_0_2π(φspan[1]), in_0_2π(φspan[2]))
    T12    = promote_type(T1, T2)
    Δφspan = ∂φstart′ == ∂φstop′ ? T12(2π) : counterclock_Δφ(∂φstart′, ∂φstop′)    
    φ∂  = @. in_0_2π(∂φstart′ + Δφspan * (0:N) / N) 
    ## Δφ  = Δφspan / N
    ## φ   = φ∂[1:end-1] .+ Δφ / 2
    φ   = φ∂[1:end-1] 
    return φ, φ∂
end





# methods for formatting Z(θ,k) to be operated on by the block diagonals
# which are produced by Γ2cov_blks and ΓC2cov_blks (given below).
#  - θ is a pixel argument a pixel field Z(θ₁,φ₁)
#  - k is a fourier frequency of a azimuthal pixel coordinate φ
# ==================================================


"""
Real map fields have an implicit pairing with primal and dual frequency
so we instead construct nφ÷2+1 vectors of length nθ 
"""
function ℝfθk2▪(Uf::AbstractArray)
    return [copy(v) for v ∈ eachcol(Uf)]
end
 
function ▪2ℝfθk(w::Vector{V}) where {To, V<:AbstractVector{To}} 
    nθ, nφ½₊1 = length(w[1]), length(w)
    fθk = zeros(To, nθ, nφ½₊1)
    for i in 1:nφ½₊1 
        fθk[:,i] = w[i]
    end
    fθk
end

"""
Complex map fields get frequency paired with dual frequency ... to make nφ÷2+1 vectors of length 2nθ 
"""
function ℂfθk2▪(Up::AbstractArray{To}) where To
    nθ, nφ = size(Up)
    w  = Vector{To}[zeros(To,2nθ) for ℓ = Base.OneTo(nφ÷2+1)]
    Up_col = collect(eachcol(Up))
    for ℓ = 1:nφ÷2+1
        if (ℓ==1) | ((ℓ==nφ÷2+1) & iseven(nφ))
            w[ℓ][1:nθ]     .= Up_col[ℓ]
            w[ℓ][nθ+1:2nθ] .= conj.(Up_col[ℓ])
        else 
            Jℓ = nφ - ℓ + 2
            w[ℓ][1:nθ]     .= Up_col[ℓ]
            w[ℓ][nθ+1:2nθ] .= conj.(Up_col[Jℓ])
        end
    end
    w
end

function ▪2ℂfθk(w::Vector{V}, nφ::Int) where {To, V<:AbstractVector{To}} 
    nθₓ2, nφ½₊1   = length(w[1]), length(w)
    @assert nφ½₊1 == nφ÷2+1
    @assert iseven(nθₓ2)
    nθ  = nθₓ2÷2

    pθk = zeros(To, nθ, nφ)
    for ℓ = 1:nφ½₊1
        if (ℓ==1) | ((ℓ==nφ½₊1) & iseven(nφ))
            pθk[:,ℓ] .= w[ℓ][1:nθ] 
        else 
            Jℓ = nφ - ℓ + 2
            pθk[:,ℓ]  .= w[ℓ][1:nθ]      
            pθk[:,Jℓ] .= conj.(w[ℓ][nθ+1:2nθ])
        end
    end 
    pθk
end


# Hi level methods for converting functions
# Γ(θ₁, θ₂, φ₁, φ⃗) = E(Z(θ₁,φ₁)* conj(Z(θ₂,φ⃗)))
# C(θ₁, θ₂, φ₁, φ⃗) = E(Z(θ₁,φ₁)* Z(θ₂,φ⃗))
# to the block diagonals that operate on the columns of the output 
# of ℂfθk2▪ and ℝfθk2▪
# ==================================================

# Γ::Function
# Γ(θ₁, θ₂, φ₁, φ⃗) = E(Z(θ₁,φ₁)* conj(Z(θ₂,φ⃗))) as a function of θ₁, θ₂, φ₁ .- φ⃗
# Note: hard coding Float64 and CompleF64 for now
function Γ2cov_blks(Γ; θ, φ, ℓrange=1:length(φ)÷2+1)
    nθ, nφ = length(θ), length(φ)
    ptmW   = [plan_fft(Vector{ComplexF64}(undef, nφ), num_threads=1) for i=1:Threads.nthreads()]
    M▫     = Matrix{Float64}[zeros(Float64,nθ,nθ) for ℓ′ in ℓrange]
    Threads.@threads for (k,j) in collect(Iterators.product(1:nθ, 1:nθ))
        Mγⱼₖℓ⃗  = γθ₁θ₂ℓ⃗(θ[j], θ[k], φ, Γ, ptmW[Threads.threadid()])
        for (i,ℓ′) in enumerate(ℓrange)
            M▫[i][j,k] = real(Mγⱼₖℓ⃗[ℓ′])
        end
    end
    return M▫
end

# Γ::Function, C::Function
# Γ(θ₁, θ₂, φ₁, φ⃗) = E(Z(θ₁,φ₁)* conj(Z(θ₂,φ⃗))) as a function of θ₁, θ₂, φ₁ .- φ⃗
# C(θ₁, θ₂, φ₁, φ⃗) = E(Z(θ₁,φ₁)* Z(θ₂,φ⃗)) as a function of θ₁, θ₂, φ₁ .- φ⃗
# Note: hard coding Float64 and CompleF64 for now
function ΓC2cov_blks(Γ, C; θ, φ, ℓrange=1:length(φ)÷2+1)
    nθ, nφ = length(θ), length(φ)
    ptmW   = [plan_fft(Vector{ComplexF64}(undef, nφ), num_threads=1) for i=1:Threads.nthreads()]
    M▫     = Matrix{ComplexF64}[zeros(ComplexF64,2nθ,2nθ) for ℓ′ in ℓrange]
    Threads.@threads for (k,j) in collect(Iterators.product(1:nθ, 1:nθ))
        Mγⱼₖℓ⃗, Mξⱼₖℓ⃗ = γθ₁θ₂ℓ⃗_ξθ₁θ₂ℓ⃗(θ[j], θ[k], φ, Γ, C, ptmW[Threads.threadid()])
        for (i,ℓ′) in enumerate(ℓrange)
            Jℓ′ = Jperm(ℓ′, nφ)
            M▫[i][j,   k   ] = Mγⱼₖℓ⃗[ℓ′]
            M▫[i][j,   k+nθ] = Mξⱼₖℓ⃗[ℓ′]
            M▫[i][j+nθ,k   ] = conj(Mξⱼₖℓ⃗[Jℓ′])
            M▫[i][j+nθ,k+nθ] = conj(Mγⱼₖℓ⃗[Jℓ′])
        end
    end
    return M▫
end


# lower level methods for the internals of ΓC2cov_blks and Γ2cov_blks
# ==================================================


function γθ₁θ₂φ⃗(θ₁::Real, θ₂::Real, φ::AbstractVector, Γθ₁θ₂φ₁φ⃗)
    φ2π, freq_mult = fullcircle(φ)
    covPP̄  = Γθ₁θ₂φ₁φ⃗(θ₁, θ₂, φ2π[1], φ2π)
    covPP̄′ = periodize(covPP̄, freq_mult)       
    return covPP̄′
end
function γθ₁θ₂ℓ⃗(
    θ₁::Real, θ₂::Real, φ::AbstractVector, Γθ₁θ₂φ₁φ⃗,
    planFFT  = plan_fft(Vector{ComplexF64}(undef,length(φ))),
    γstorage = Vector{ComplexF64}(undef,length(φ)), 
    )
    γ₁₂φ⃗ = γθ₁θ₂φ⃗(θ₁, θ₂, φ, Γθ₁θ₂φ₁φ⃗)      
    mul!(γstorage, planFFT, γ₁₂φ⃗)
    return γstorage
end




function γθ₁θ₂φ⃗_ξθ₁θ₂φ⃗(θ₁::Real, θ₂::Real, φ::AbstractVector, Γθ₁θ₂φ₁φ⃗, Cθ₁θ₂φ₁φ⃗)
    φ2π, freq_mult = fullcircle(φ)
    covPP̄  = Γθ₁θ₂φ₁φ⃗(θ₁, θ₂, φ2π[1], φ2π)
    covPP  = Cθ₁θ₂φ₁φ⃗(θ₁, θ₂, φ2π[1], φ2π)
    covPP̄′ = periodize(covPP̄, freq_mult)       
    covPP′ = periodize(covPP, freq_mult)       
    return covPP̄′, covPP′
end
function γθ₁θ₂ℓ⃗_ξθ₁θ₂ℓ⃗(
    θ₁::Real, θ₂::Real, φ::AbstractVector, Γθ₁θ₂φ₁φ⃗, Cθ₁θ₂φ₁φ⃗,
    planFFT = plan_fft(Vector{ComplexF64}(undef,length(φ))),
    γstorage = Vector{ComplexF64}(undef,length(φ)), 
    ξstorage = Vector{ComplexF64}(undef,length(φ)), 
    )
    γ₁₂φ⃗, ξ₁₂φ⃗ =  γθ₁θ₂φ⃗_ξθ₁θ₂φ⃗(θ₁, θ₂, φ, Γθ₁θ₂φ₁φ⃗, Cθ₁θ₂φ₁φ⃗)      
    mul!(γstorage, planFFT, γ₁₂φ⃗)
    mul!(ξstorage, planFFT, ξ₁₂φ⃗)
    return γstorage, ξstorage
end





# Some custom types for Γθ₁θ₂φ₁φ⃗, Cθ₁θ₂φ₁φ⃗ for dispatching to a 
# custom method for γθ₁θ₂φ⃗_ξθ₁θ₂φ⃗ (defined above) for CMB temp
# =====================================================

struct Γθ₁θ₂φ₁φ⃗_Iso
    spln::Spline1D
end 

# constructor
function Γθ₁θ₂φ₁φ⃗_Iso(
        ℓ, ttℓ;
        ngrid::Int = 100_000, 
        βgrid = βsingrid(ngrid,π),
    )
    @assert ℓ[1] == 0
    @assert ℓ[2] == 1
    nℓ = @. (2ℓ+1)/(4π)
    j0⁺0tℓ = @. ttℓ * nℓ
    f0⁺0t = ((a,b,jℓ)=(0,0,j0⁺0tℓ); Fun(Jacobi(b,a),jℓ))
    covtt = x-> f0⁺0t(cos(x))
    β2covtt = Spline1D(βgrid, covtt.(βgrid), k=3)
    Γθ₁θ₂φ₁φ⃗_Iso(β2covtt)
end 

# Hook into method γθ₁θ₂φ⃗
function γθ₁θ₂φ⃗(
    θ₁::Real, θ₂::Real, φ::AbstractVector, 
    Γθ₁θ₂φ₁φ⃗::Γθ₁θ₂φ₁φ⃗_Iso, 
    )
    
    φ2π, freq_mult = fullcircle(φ)
    β      = geoβ.(θ₁, θ₂, φ2π[1], φ2π)
    covPP̄  = Γθ₁θ₂φ₁φ⃗.spln(β)
    covPP̄′ = periodize(covPP̄, freq_mult)       

    return complex(covPP̄′)
end





# Some custom types for Γθ₁θ₂φ₁φ⃗, Cθ₁θ₂φ₁φ⃗ for dispatching to a 
# custom method for γθ₁θ₂φ⃗_ξθ₁θ₂φ⃗ (defined above) for CMB polarization
# =====================================================

"""
IAU uses rotation around outward normal to the sphere => Q + iU is spin (+2)
"""
struct Γθ₁θ₂φ₁φ⃗_CMBpol
    IAU::Bool
    premult_spln::Spline1D
end 

struct Cθ₁θ₂φ₁φ⃗_CMBpol
    IAU::Bool
    premult_spln::Spline1D
end 

# Constructor for both Γ and C
function ΓCθ₁θ₂φ₁φ⃗_CMBpol(
        ℓ, eeℓ, bbℓ;
        IAU = false, 
        ngrid::Int = 100_000, 
        βgrid = βsingrid(ngrid, π),
    )
    @assert ℓ[1] == 0
    @assert ℓ[2] == 1
    @assert IAU == false # TODO remove this an impliment the spin(+2) version
    nℓ = @. (2ℓ+1)/(4π)
    ## ↓ starts at ℓ=2 since the Jacobi expansion goes like J^(a,b)_{ℓ-2}
    j2⁺2ℓ = (@. (eeℓ + bbℓ) * nℓ)[3:end]
    j2⁻2ℓ = (@. (eeℓ - bbℓ) * nℓ)[3:end]
    ## ↓  TODO: check the a,b swap
    f2⁺2  = ((a,b,jℓ)=(0,4,j2⁺2ℓ);  Fun(Jacobi(b,a),jℓ))
    f2⁻2  = ((a,b,jℓ)=(4,0,j2⁻2ℓ);  Fun(Jacobi(b,a),jℓ))
    # !! pre-canceled out cos β½ and sin β½ in the denom
    covPP̄ = x-> f2⁺2(cos(x))
    covPP = x-> f2⁻2(cos(x))
    β2covPP̄ = Spline1D(βgrid, covPP̄.(βgrid), k=3)
    β2covPP = Spline1D(βgrid, covPP.(βgrid), k=3)
    Γθ₁θ₂φ₁φ⃗_CMBpol(IAU, β2covPP̄), Cθ₁θ₂φ₁φ⃗_CMBpol(IAU, β2covPP)
end 


function γθ₁θ₂φ⃗_ξθ₁θ₂φ⃗(
    θ₁::Real, θ₂::Real, φ::AbstractVector, 
    Γθ₁θ₂φ₁φ⃗::Γθ₁θ₂φ₁φ⃗_CMBpol, 
    Cθ₁θ₂φ₁φ⃗::Cθ₁θ₂φ₁φ⃗_CMBpol,
    )
    
    φ2π, freq_mult = fullcircle(φ)
    β      = geoβ.(θ₁, θ₂, φ2π[1], φ2π)
    covPP̄  = Γθ₁θ₂φ₁φ⃗.premult_spln(β) .* multPP̄.(θ₁, θ₂, φ2π[1], φ2π)
    covPP  = Cθ₁θ₂φ₁φ⃗.premult_spln(β) .* multPP.(θ₁, θ₂, φ2π[1], φ2π)
    covPP̄′ = periodize(covPP̄, freq_mult)       
    covPP′ = periodize(covPP, freq_mult)       

    return covPP̄′, covPP′
end


function sincosΔθpθΔφ(θ₁, θ₂, φ₁, φ₂)
    𝓅θ½ = (θ₁ + θ₂)/2
    Δθ½ = (θ₁ - θ₂)/2
    Δφ½ = (φ₁ - φ₂)/2
    s𝓅θ½, c𝓅θ½ = sincos(𝓅θ½)
    sΔθ½, cΔθ½ = sincos(Δθ½)
    sΔφ½, cΔφ½ = sincos(Δφ½)
    return sΔθ½, sΔφ½, cΔθ½, cΔφ½, s𝓅θ½, c𝓅θ½
end

function multPP̄(θ₁, θ₂, φ₁, φ₂)
    sΔθ½, sΔφ½, cΔθ½, cΔφ½, s𝓅θ½, c𝓅θ½ = sincosΔθpθΔφ(θ₁, θ₂, φ₁, φ₂)
    return complex(sΔφ½ * c𝓅θ½,   cΔφ½ * cΔθ½)^4
end

function multPP(θ₁, θ₂, φ₁, φ₂)
    sΔθ½, sΔφ½, cΔθ½, cΔφ½, s𝓅θ½, c𝓅θ½ = sincosΔθpθΔφ(θ₁, θ₂, φ₁, φ₂)
    return complex(sΔφ½ * s𝓅θ½, - cΔφ½ * sΔθ½)^4
end

## multII(θ₁, θ₂, φ₁, φ₂) = 1

Q1Q2(covPP̄, covPP) = ( real(covPP̄) + real(covPP) ) / 2

U1U2(covPP̄, covPP) = ( real(covPP̄) - real(covPP) ) / 2

Q1U2(covPP̄, covPP) = ( imag(covPP̄) + imag(covPP) ) / 2

U1Q2(covPP̄, covPP) = (- imag(covPP̄) + imag(covPP) ) / 2




# misc
# ==================================================

"""
`cheb2spherecov(β, θs)` compute auto-covariance function at 
angular distances `θs` from chebyshev expansion.
"""
function cheb2spherecov(β, θs)
    cθ = zero(θs)
    n  = length(β)
    for i ∈ eachindex(cθ)
        for (k,kp1) ∈ zip(0:n-1, 1:n)
            @inbounds cθ[i] += cos(k * θs[i]) * β[kp1]
        end
    end 
    return cθ
end


"""
`spec2spherecov(cl, θs)` compute auto-covariance function at 
angular distances `θs` from spherical spectral density `cl`. 
Note: `cl` is assumed to be a vector whose values correspond to 
multipoles `l = [0,1,2 ... ,lmax]`
"""
function spec2spherecov(cl, θs)
    lmax = length(cl)-1
    l    = 0:lmax
    j00l = @. cl * (2l + 1) / (4π)
    β    = jac2cheb(j00l, 0, 0) 
    cheb2spherecov(β, θs)
end



end
