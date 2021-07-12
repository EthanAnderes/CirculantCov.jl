module CirculantCov

using Dierckx: Spline1D 
using ApproxFun: Fun, Jacobi
using FastTransforms: jac2cheb


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


# periodize
# ==================================================

function periodize(f::Vector{T}, freq_mult::Int) where {T}
    n = length(f)
    nfm = n÷freq_mult
    @assert nfm == n//freq_mult
    f′ = sum( circshift(f, k*nfm) for k=0:freq_mult-1)
    f′[1:nfm]
end


# Types that compute the isotropic part of 
# Spin2 and Spin0 CMBfields
# ==================================================


struct βcovSpin2
    covPP̄_premult_spln::Spline1D
    covPP_premult_spln::Spline1D
end

struct βcovSpin0 
    covII_premult_spln::Spline1D
end


# constructors
# ==================================================

function βcovSpin2(
        ℓ, eeℓ, bbℓ;
        n_grid::Int = 100_000, 
        β_grid = range(0, π^(1/3), length=n_grid).^3,
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
    # pre-canceled out cos β½ and sin β½ in the denom
    covPP̄ = x-> f2⁺2(cos(x))
    covPP = x-> f2⁻2(cos(x))
    β2covPP̄ = Spline1D(β_grid, covPP̄.(β_grid), k=3)
    β2covPP = Spline1D(β_grid, covPP.(β_grid), k=3)

    return βcovSpin2(β2covPP̄, β2covPP)

end 

function βcovSpin0(
        ℓ, ttℓ;
        n_grid::Int = 100_000, 
        β_grid = range(0, π^(1/3), length=n_grid).^3,
    )

    @assert ℓ[1] == 0
    @assert ℓ[2] == 1
    nℓ = @. (2ℓ+1)/(4π)
    ## ↓ starts at 2 since the Jacobi expansion goes like J^(a,b)_{ℓ-2}
    j0⁺0tℓ = @. ttℓ * nℓ
    ## ↓  TODO: check the a,b swap
    f0⁺0t = ((a,b,jℓ)=(0,0,j0⁺0tℓ); Fun(Jacobi(b,a),jℓ))
    ## leaving out the outer factors witch cancel with the sphere rotation
    covtt = x-> f0⁺0t(cos(x))
    β2covtt = Spline1D(β_grid, covtt.(β_grid), k=3)

    return βcovSpin0(β2covtt)

end 


# the types operate ... this is pre-vectorized since Spline1D is on vectors
# TODO: include the spin2 mult factors
# ==================================================

function (covP::βcovSpin2)(β::Matrix)
    rtnPP̄ = similar(β)
    rtnPP = similar(β)
    for (col, cβ) ∈ enumerate(eachcol(β))
        rtnPP̄[:,col] = covP.covPP̄_premult_spln(cβ)
        rtnPP[:,col] = covP.covPP_premult_spln(cβ)
    end
    return complex.(rtnPP̄,0), complex.(rtnPP,0) 
end
function (covP::βcovSpin2)(β::Union{Vector, Number})
    rtnPP̄ = covP.covPP̄_premult_spln(β)
    rtnPP = covP.covPP_premult_spln(β)
    return complex.(rtnPP̄,0), complex.(rtnPP,0)     
end


function (covP::βcovSpin0)(β::Matrix)
    rtn = similar(β)
    for (col, cβ) ∈ enumerate(eachcol(β))
        rtn[:,col] = covP.covII_premult_spln(cβ)
    end
    return rtn  
end
function (covP::βcovSpin0)(β::Union{Vector, Number})
    return covP.covII_premult_spln(β)
end


# necessary geometric methods with angles and geodesics
# ==================================================

function sincosΔθpθΔφ(θ1, θ2, φ1, φ2)
    𝓅θ½ = (θ1 + θ2)/2
    Δθ½ = (θ1 - θ2)/2
    Δφ½ = (φ1 - φ2)/2
    s𝓅θ½, c𝓅θ½ = sincos(𝓅θ½)
    sΔθ½, cΔθ½ = sincos(Δθ½)
    sΔφ½, cΔφ½ = sincos(Δφ½)
    return sΔθ½, sΔφ½, cΔθ½, cΔφ½, s𝓅θ½, c𝓅θ½
end

function geoβ(θ1, θ2, φ1, φ2)
    sθ1, sθ2 = sin(θ1), sin(θ2)
    sΔθ½, sΔφ½, = sincosΔθpθΔφ(θ1, θ2, φ1, φ2)
    return 2asin(√(sΔθ½^2 + sθ1 * sθ2 * sΔφ½^2))    
end


# Multipliers needed to convert the isotropic parts to full polarization cov 
# =====================================================

function multPP̄(θ1, θ2, φ1, φ2)
    sΔθ½, sΔφ½, cΔθ½, cΔφ½, s𝓅θ½, c𝓅θ½ = sincosΔθpθΔφ(θ1, θ2, φ1, φ2)
    return complex(sΔφ½ * c𝓅θ½,   cΔφ½ * cΔθ½)^4
end

function multPP(θ1, θ2, φ1, φ2)
    sΔθ½, sΔφ½, cΔθ½, cΔφ½, s𝓅θ½, c𝓅θ½ = sincosΔθpθΔφ(θ1, θ2, φ1, φ2)
    return complex(sΔφ½ * s𝓅θ½, - cΔφ½ * sΔθ½)^4
end

## multII(θ1, θ2, φ1, φ2) = 1

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







end
