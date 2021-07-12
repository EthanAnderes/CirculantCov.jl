using CirculantCov
using Test

@testset "CirculantCov" begin
    # Write your tests here.
end




@testset "CirculantCov: periodize" begin


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


end
