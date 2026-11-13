# Build-once h-multigrid preconditioner for the Stokes saddle system.
# Geometry/transfers are cached per Mesh; the coarse factorisation and pressure
# mass are refreshed per theta.

struct MGPreconditioner
    Pblk::SparseMatrixCSC{Float64,Int}
    R::SparseMatrixCSC{Float64,Int}
    Acl::Any
    Df::Vector{Float64}
    nvel::Int
    Av::SparseMatrixCSC{Float64,Int}
end

const MG_GEOM = IdDict{Any,Any}()

function build_prolongation(mf::Mesh, mc::Mesh, rule, xs, zs)
    p = mf.p
    I = Int[]; J = Int[]; V = Float64[]
    for iz in 1:mf.nvz, ix in 1:mf.nvx
        g = (iz - 1) * mf.nvx + ix
        x = xs[ix]; z = zs[iz]
        ex = min(floor(Int, x / mc.hx), mc.nxe - 1)
        ez = clamp(searchsortedlast(mc.zedges, z) - 1, 0, mc.nze - 1)
        hz = z_size(mc, ez)
        xi = 2.0 * (x - ex * mc.hx) / mc.hx - 1.0
        eta = 2.0 * (z - z_left(mc, ez)) / hz - 1.0
        Lx, _ = lagrange_eval(rule, xi)
        Lz, _ = lagrange_eval(rule, eta)
        gids = element_vel_gids(mc, ex, ez)
        for j in 1:p+1, i in 1:p+1
            k = (j - 1) * (p + 1) + i
            push!(I, g); push!(J, gids[k]); push!(V, Lx[i] * Lz[j])
        end
    end
    return sparse(I, J, V, mf.Nscalar, mc.Nscalar)
end

function mg_geometry(m::Mesh, dm::DofMap)
    haskey(MG_GEOM, m) && return MG_GEOM[m]
    n = m.nxe
    (n % 2 == 0 && m.nze % 2 == 0) || return nothing
    mc = make_mesh(m.Lx, n ÷ 2, m.nze ÷ 2, m.p; zedges=m.zedges[1:2:end])
    fdc = make_fedata(m.p, mc.q)
    dmc = make_dofmap(mc, :free)
    xs, zs = node_coords(m, gll_rule(m.p))
    P = build_prolongation(m, mc, gll_rule(m.p), xs, zs)
    gfx = findall(!iszero, dm.red_ux); gcx = findall(!iszero, dmc.red_ux)
    gfz = findall(!iszero, dm.red_uz); gcz = findall(!iszero, dmc.red_uz)
    Pblk = blockdiag(P[gfx, gcx], P[gfz, gcz])
    geo = (mc, fdc, dmc, P, Pblk, sparse(Pblk'))
    MG_GEOM[m] = geo
    return geo
end

function build_mg(m::Mesh, fd::FEData, pr::Params, theta::Vector{Float64}, dm::DofMap,
                  sys::StokesSystem; refstate=nothing, tala=true, p_prev=nothing,
                  mass_form=:rho, buoyancy_form=:anomaly, phases=PhaseTransition[],
                  rheology=nothing, tosi=nothing)
    geo = mg_geometry(m, dm)
    geo === nothing && return nothing
    mc, fdc, dmc, P, Pblk, R = geo
    theta_c = sparse(P') * theta
    sysc = assemble_stokes(mc, fdc, pr, theta_c, dmc; refstate=refstate, tala=tala,
                           p_prev=nothing, mass_form=mass_form,
                           buoyancy_form=buoyancy_form, phases=phases,
                           rheology=rheology, tosi=tosi)
    nvelc = dmc.nux + dmc.nuz
    Avc = sysc.A[1:nvelc, 1:nvelc]
    nvel = dm.nux + dm.nuz
    Av = sys.A[1:nvel, 1:nvel]
    return MGPreconditioner(Pblk, R, lu(Avc), diag(Av), nvel, Av)
end

function apply_mg!(z, r, mg::MGPreconditioner, A, Ml, Bm)
    nvel = mg.nvel
    x = zeros(nvel)
    for _ in 1:6
        x .+= (2.0 / 3.0) .* ((r[1:nvel] - mg.Av * x) ./ mg.Df)
    end
    x .+= mg.Pblk * (mg.Acl \ (mg.R * (r[1:nvel] - mg.Av * x)))
    for _ in 1:6
        x .+= (2.0 / 3.0) .* ((r[1:nvel] - mg.Av * x) ./ mg.Df)
    end
    z[1:nvel] = x
    rp = r[nvel+1:end] .- Bm * x
    z[nvel+1:end] = Ml \ rp
    return z
end


