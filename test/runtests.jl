using CirculantCov
import CirculantCov as CC
using Test


@testset "φ_grid, θ_grid" begin
    
    θspan = (2.3,2.7)
    N     = 250
    type  = :equicosθ
    θ, θ∂ = CC.θ_grid(; θspan, N, type)
    @test all(θspan[1] .<  θ  .<  θspan[2])
    @test all(θspan[1] .<= θ∂ .<= θspan[2])
    @test length(θ)  == N
    @test length(θ∂) == N+1

    θspan = (2.3,2.7)
    N     = 250
    type  = :equiθ
    θ, θ∂ = CC.θ_grid(; θspan, N, type)
    @test all(θspan[1] .<  θ  .<  θspan[2])
    @test all(θspan[1] .<= θ∂ .<= θspan[2])
    @test length(θ)  == N
    @test length(θ∂) == N+1

end


@testset "counterclock_Δφ and fullcircle" begin
    
    φstart, φstop = 0, 1.0        ; @test CC.counterclock_Δφ(φstart, φstop) ≈ φstop - φstart
    φstart, φstop = 0, π          ; @test CC.counterclock_Δφ(φstart, φstop) ≈ φstop - φstart
    φstart, φstop = 0, 3π/2       ; @test CC.counterclock_Δφ(φstart, φstop) ≈ φstop - φstart
    φstart, φstop = 0, 2pi - .001 ; @test CC.counterclock_Δφ(φstart, φstop) ≈ φstop - φstart

    φstart, φstop = 2π - 0.5, 1.5 ; @test CC.counterclock_Δφ(φstart, φstop) ≈ 2.0

    φstart, φstop =  1.0        , 0 ; @test isapprox(CC.counterclock_Δφ(φstart, φstop), 2pi - 1.0, rtol=1e-5)
    φstart, φstop =  π          , 0 ; @test isapprox(CC.counterclock_Δφ(φstart, φstop), π, rtol=1e-5)
    φstart, φstop =  3π/2       , 0 ; @test isapprox(CC.counterclock_Δφ(φstart, φstop), π/2, rtol=1e-5)
    φstart, φstop =  2pi - .001 , 0 ; @test isapprox(CC.counterclock_Δφ(φstart, φstop), .001, rtol=1e-5)

    @test CC.fullcircle(2pi * (0:9) / 10)[2]   == 1
    @test CC.fullcircle(pi/2 * (0:9) / 10)[2]  == 4
    @test CC.fullcircle(2pi/3 * (0:9) / 10)[2] == 3
    @test all(CC.fullcircle(2pi * (0:9) / 10)[1]  .≈ 2pi * (0:9) / 10)
    @test all(CC.fullcircle(pi/2 * (0:9) / 10)[1]  .≈ 2pi * (0:39) / 40)
    @test all(CC.fullcircle(2pi/3 * (0:9) / 10)[1]  .≈ 2pi * (0:29) / 30)

end


@testset "test in_0_2π and in_negπ_π" begin

    # in_0_2π

    @test CC.in_0_2π(0.0) ≈ 0.0
    @test CC.in_0_2π(2π)  ≈ 0.0
    @test CC.in_0_2π(-2π) ≈ 0.0 atol=1e-10

    φ = range(0,2π,20)[1:end-1] # don't want 2π in this test range
    @test CC.in_0_2π.(φ) ≈ φ
    @test CC.in_0_2π.(φ .+ 2π) ≈ φ
    @test CC.in_0_2π.(φ .+ 2*2π) ≈ φ
    @test CC.in_0_2π.(φ .+ 3*2π) ≈ φ
    @test CC.in_0_2π.(φ .- 3*2π) ≈ φ
    @test CC.in_0_2π.(φ .- 2*2π) ≈ φ
    @test CC.in_0_2π.(φ .- 2π) ≈ φ

    @test CC.in_0_2π.(deg2rad.([0, 360]))   ≈ [0.0,0.0]
    @test CC.in_0_2π.(deg2rad.([-180, 180])) ≈ [π, π]
    @test CC.in_0_2π.(deg2rad.([-60, 60])) ≈ [deg2rad(300), deg2rad(60)]

    # in_negπ_π
    
    @test CC.in_negπ_π(0.0)  ≈ 0.0
    @test CC.in_negπ_π(Float64(-π))   ≈ -π
    @test CC.in_negπ_π(Float64(π))    ≈ -π

    φ = range(-π,π,20)[1:end-1] # don't want π in this test range
    @test CC.in_negπ_π.(φ) ≈ φ
    @test CC.in_negπ_π.(φ .+ 2π) ≈ φ
    @test CC.in_negπ_π.(φ .+ 2*2π) ≈ φ
    @test CC.in_negπ_π.(φ .+ 3*2π) ≈ φ
    @test CC.in_negπ_π.(φ .- 3*2π) ≈ φ
    @test CC.in_negπ_π.(φ .- 2*2π) ≈ φ
    @test CC.in_negπ_π.(φ .- 2π) ≈ φ

    @test CC.in_negπ_π.(deg2rad.([-180, 180])) ≈ [-π, -π]
    @test CC.in_negπ_π.(deg2rad.([-60, 60])) ≈ [deg2rad(-60), deg2rad(60)]

end 

# @testset "CirculantCov: periodize" begin


    #= 3 ways to test out periodizing

    n = 1024*3    # even
    #n = (768-1)*3 # odd
    f = 2π*(0:n-1)/n .|> cos .|> cφ -> exp(-(1-cφ)*100) * (1 - ((1-cφ)*100)^2/2) .|> complex

    freq_mult=3
    nfm = n÷freq_mult
    @assert nfm == n//freq_mult

    # -----
    f′ = sum( circshift(f, k*nfm) for k=0:freq_mult-1)
    f  .|> real  |> plot
    f′ .|> real |> plot
    f′[1:nfm] .|> real |> plot


    # -----
    fk = zeros(eltype(f), size(f))
    fk[1:freq_mult:end] = fft(f)[1:freq_mult:end]
    f′′ = freq_mult.* ifft(fk)


    f .|> real |> plot
    f′′ .|> real |> plot
    f′′[1:nfm] .|> real |> plot

    # -----
    f′′′ = ifft(fft(f)[1:freq_mult:end])
    f    .|> real |> plot
    f′′′ .|> real |> plot
    =#


# end
