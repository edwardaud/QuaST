# G0: GLL nodes, weights, Lagrange basis and Gauss-Legendre quadrature — exactness and interpolation identities to round-off.
using Test
using QuaST

@testset "G0: GLL nodes, weights, Lagrange basis, Gauss-Legendre" begin
    for p in 1:10
        r = gll_rule(p)
        @test length(r.nodes) == p + 1
        @test isapprox(r.nodes[1], -1.0, atol=1e-14)
        @test isapprox(r.nodes[end], 1.0, atol=1e-14)
        @test isapprox(sum(r.weights), 2.0, atol=1e-13)
        for i in 1:p+1
            @test isapprox(r.nodes[i], -r.nodes[p+2-i], atol=1e-13)
        end
        # GLL quadrature is exact for degree <= 2p-1
        for k in 0:2p-1
            val = sum(r.weights .* r.nodes .^ k)
            exact = isodd(k) ? 0.0 : 2.0 / (k + 1)
            @test isapprox(val, exact, atol=1e-12)
        end
        # Lagrange partition of unity and zero derivative sum
        for x in (-1.0, -0.3, 0.0, 0.7, 1.0)
            L, dL = lagrange_eval(r, x)
            @test isapprox(sum(L), 1.0, atol=1e-13)
            @test isapprox(sum(dL), 0.0, atol=1e-12)
        end
        # nodal identity and derivative matrix row sums
        for i in 1:p+1
            L, dL = lagrange_eval(r, r.nodes[i])
            @test isapprox(L[i], 1.0, atol=1e-14)
            @test isapprox(sum(L), 1.0, atol=1e-13)
            @test isapprox(sum(dL), 0.0, atol=1e-12)
        end
    end
    for n in 1:10
        x, w = gauss_legendre(n)
        @test isapprox(sum(w), 2.0, atol=1e-13)
        for k in 0:2n-1
            val = sum(w .* x .^ k)
            exact = isodd(k) ? 0.0 : 2.0 / (k + 1)
            @test isapprox(val, exact, atol=1e-12)
        end
    end
end
