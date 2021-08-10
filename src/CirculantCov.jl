module CirculantCov

using Dierckx: Spline1D 
using ApproxFun: Fun, Jacobi
using FastTransforms: jac2cheb
using FFTW: plan_fft
using LinearAlgebra: mul!

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


# geometric methods with angles and geodesics and periodize
# ==================================================

function periodize(f::Vector{T}, freq_mult::Int) where {T}
    n = length(f)
    nfm = n÷freq_mult
    @assert nfm == n//freq_mult
    f′ = sum( circshift(f, k*nfm) for k=0:freq_mult-1)
    f′[1:nfm]
end

in_0_2π(φ) = rem2pi(φ, RoundDown)

counterclock_Δφ(φstart, φstop) = in_0_2π(φstop - φstart)

function fullcircle(φ::AbstractVector)
    Δφpix  = counterclock_Δφ(φ[1], φ[2])
    Δφspan = counterclock_Δφ(φ[1], φ[end]) + Δφpix
    # The extra Δφpix makes Δφspan measure the (angular) distance between the 
    # left boundary of the starting pixel and the right boundary of the ending pixel 
    
    @assert div(2π, Δφspan, RoundNearest) ≈ 2π / Δφspan
    freq_mult = Int(div(2π, Δφspan, RoundNearest))
    
    nφ2π = length(φ)*freq_mult
    φ2π  = @. in_0_2π(φ[1] + 2π * (0:nφ2π-1) / nφ2π) 

    return φ2π, freq_mult
end


"""
# slated for removal ...
`fraccircle(∂φstart, ∂φstop, nφ)` specifies a uniform grid on a 
contiguous interval of azumimuth. `∂φstart` begins the interval. Moving counter 
clockwise (looking down from the north pole) to `∂φstop`. 
Only integer fractions are allowed and both `∂φstart`, `∂φstop` must be `≥ 0`.

Note: `(∂φstart, ∂φstop) = (5.3, 1.0) ≡ (5.3, 1.0 + 2π)` 
"""
# function fraccircle(∂φstart, ∂φstop, nφ)
#     ∂φstart′, ∂φstop′ = in_0_2π(∂φstart), in_0_2π(∂φstop)
#     Δφspan = counterclock_Δφ(∂φstart′, ∂φstop′)    
#     φ  = @. in_0_2π(∂φstart′ + Δφspan * (0:nφ-1) / nφ) 
#     return φ
# end



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

function θ_grid(;θspan::Tuple{T,T}, N::Int, type=:equiθ) where T<:Real
    @assert N > 0
    @assert 0 < θspan[1] < θspan[2] < π

    if type==:equiθ
        θgrid″ = range(θspan[1], θspan[2], length=N+2)
    elseif type==:equicosθ
        θgrid″ = range(cos(θspan[2]), cos(θspan[1]), length=N+2)[end:-1:1]
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
`φ_grid(;φspan::Tuple{T,T}, N::Int)` specifies a uniform grid on a 
contiguous interval of azumimuth. `∂φstart` begins the interval. Moving counter 
clockwise (looking down from the north pole) to `∂φstop`. 
Only integer fractions are allowed and both `∂φstart`, `∂φstop` must be `≥ 0`.

Note: `(∂φstart, ∂φstop) = (5.3, 1.0) ≡ (5.3, 1.0 + 2π)`
"""
function φ_grid(;φspan::Tuple{T,T}, N::Int) where T<:Real
    ∂φstart′, ∂φstop′ = in_0_2π(φspan[1]), in_0_2π(φspan[2])
    Δφspan = counterclock_Δφ(∂φstart′, ∂φstop′)    
    φ∂  = @. in_0_2π(∂φstart′ + Δφspan * (0:N) / N) 
    Δφ  = Δφspan / N
    φ   = φ∂[1:end-1] .+ Δφ / 2
    return φ, φ∂
end


# these are the generic versions ...
# you just need to define types for Γθ₁θ₂φ₁φ⃗, Cθ₁θ₂φ₁φ⃗
# for dispatch
# ==================================================

# Note: the reason we restrict to φ::AbstractVector is that Spline1D's are optimized for 
# Union{Vector, Number} so in general it is better to broadcast 
# via via whole columns


# overload this
function γθ₁θ₂φ⃗_ξθ₁θ₂φ⃗(θ₁::Real, θ₂::Real, φ::AbstractVector, Γθ₁θ₂φ₁φ⃗, Cθ₁θ₂φ₁φ⃗)
    
    φ2π, freq_mult = fullcircle(φ)
    covPP̄  = Γθ₁θ₂φ₁φ⃗(θ₁, θ₂, φ2π[1], φ2π)
    covPP  = Cθ₁θ₂φ₁φ⃗(θ₁, θ₂, φ2π[1], φ2π)
    covPP̄′ = periodize(covPP̄, freq_mult)       
    covPP′ = periodize(covPP, freq_mult)       

    return covPP̄′, covPP′
end

# overload this
function γθ₁θ₂φ⃗(θ₁::Real, θ₂::Real, φ::AbstractVector, Γθ₁θ₂φ₁φ⃗)
    
    φ2π, freq_mult = fullcircle(φ)
    covPP̄  = Γθ₁θ₂φ₁φ⃗(θ₁, θ₂, φ2π[1], φ2π)
    covPP̄′ = periodize(covPP̄, freq_mult)       

    return covPP̄′
end

# behavior comes directly from γθ₁θ₂φ⃗_ξθ₁θ₂φ⃗
function γθ₁θ₂ℓ⃗_ξθ₁θ₂ℓ⃗(
    θ₁::Real, θ₂::Real, φ::AbstractVector, Γθ₁θ₂φ₁φ⃗, Cθ₁θ₂φ₁φ⃗,
    planFFT = FFTW.plan_fft(Vector{ComplexF64}(undef,length(φ))),
    γstorage = Vector{ComplexF64}(undef,length(φ)), 
    ξstorage = Vector{ComplexF64}(undef,length(φ)), 
    )
    
    γ₁₂φ⃗, ξ₁₂φ⃗ =  γθ₁θ₂φ⃗_ξθ₁θ₂φ⃗(θ₁, θ₂, φ, Γθ₁θ₂φ₁φ⃗, Cθ₁θ₂φ₁φ⃗)      

    mul!(γstorage, planFFT, γ₁₂φ⃗)
    mul!(ξstorage, planFFT, ξ₁₂φ⃗)

    return γstorage, ξstorage
end

# behavior comes directly from γθ₁θ₂φ⃗_ξθ₁θ₂φ⃗
function γθ₁θ₂ℓ⃗(
    θ₁::Real, θ₂::Real, φ::AbstractVector, Γθ₁θ₂φ₁φ⃗,
    planFFT  = FFTW.plan_fft(Vector{ComplexF64}(undef,length(φ))),
    γstorage = Vector{ComplexF64}(undef,length(φ)), 
    )
    
    γ₁₂φ⃗ = γθ₁θ₂φ⃗(θ₁, θ₂, φ, Γθ₁θ₂φ₁φ⃗)      
    mul!(γstorage, planFFT, γ₁₂φ⃗)

    return γstorage
end



# structs Γθ₁θ₂φ₁φ⃗_CMBpol and Cθ₁θ₂φ₁φ⃗_CMBpol
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
    ## ↓ starts at 2 since the Jacobi expansion goes like J^(a,b)_{ℓ-2}
    j2⁺2ℓ = (@. (eeℓ + bbℓ) * nℓ)[2:end]
    j2⁻2ℓ = (@. (eeℓ - bbℓ) * nℓ)[2:end]
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


# Hook into method γθ₁θ₂φ⃗_ξθ₁θ₂φ⃗
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




# for isotropic spin 0
# =======

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

# Multipliers needed to convert the isotropic parts to full polarization cov 
# =====================================================

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






# 
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





# These are slated for removal ...
# ==================================================



struct βcovSpin2
    covPP̄_premult_spln::Spline1D
    covPP_premult_spln::Spline1D
end

struct βcovSpin0 
    covII_premult_spln::Spline1D
end

function βcovSpin2(
        ℓ, eeℓ, bbℓ;
        ngrid::Int = 100_000, 
        βgrid = βsingrid(ngrid, π),
    )
    @assert ℓ[1] == 0
    @assert ℓ[2] == 1
    nℓ = @. (2ℓ+1)/(4π)
    ## ↓ starts at 2 since the Jacobi expansion goes like J^(a,b)_{ℓ-2}
    j2⁺2ℓ = (@. (eeℓ + bbℓ) * nℓ)[2:end]
    j2⁻2ℓ = (@. (eeℓ - bbℓ) * nℓ)[2:end]
    ## ↓  TODO: check the a,b swap
    f2⁺2  = ((a,b,jℓ)=(0,4,j2⁺2ℓ);  Fun(Jacobi(b,a),jℓ))
    f2⁻2  = ((a,b,jℓ)=(4,0,j2⁻2ℓ);  Fun(Jacobi(b,a),jℓ))
    # !! pre-canceled out cos β½ and sin β½ in the denom
    covPP̄ = x-> f2⁺2(cos(x))
    covPP = x-> f2⁻2(cos(x))
    β2covPP̄ = Spline1D(βgrid, covPP̄.(βgrid), k=3)
    β2covPP = Spline1D(βgrid, covPP.(βgrid), k=3)
    βcovSpin2(β2covPP̄, β2covPP)
end 

function βcovSpin0(
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
    βcovSpin0(β2covtt)
end 

function (covP::βcovSpin2)(β::Matrix)
    rtnPP̄ = similar(β)
    rtnPP = similar(β)
    for (col, cβ) ∈ enumerate(eachcol(β))
        rtnPP̄[:,col] = covP.covPP̄_premult_spln(cβ)
        rtnPP[:,col] = covP.covPP_premult_spln(cβ)
    end
    return complex(rtnPP̄), complex(rtnPP)
end
function (covP::βcovSpin2)(β::Union{Vector, Number})
    rtnPP̄ = covP.covPP̄_premult_spln(β)
    rtnPP = covP.covPP_premult_spln(β)
    return complex(rtnPP̄), complex(rtnPP)
end

# Note: the reason we have different methods for Matrix vrs 
# Union{Vector, Number} is that Spline1D's are optimized for 
# Union{Vector, Number} so in general it is better to broadcast 
# via via whole columns

# Also note: the only reason we make the return argument complex 
# is that is the eltype the planned FFT will expect. 

function (covP::βcovSpin0)(β::Matrix)
    rtn = similar(β)
    for (col, cβ) ∈ enumerate(eachcol(β))
        rtn[:,col] = covP.covII_premult_spln(cβ)
    end
    return complex(rtn)  
end
function (covP::βcovSpin0)(β::Union{Vector, Number})
    return complex(covP.covII_premult_spln(β))
end






end
