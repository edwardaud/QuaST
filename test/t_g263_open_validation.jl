# G263: open walls and normal traction -- validation matrix, linear-path
# agreement, and the restart guard (boundary-condition feature, stage 3).
#
# Stage 3 makes `wall_bc` accept `:open` and adds `Params.wall_tn`.  That is a
# *constraint-set* change (an open wall writes no Dirichlet dof), so the stage
# needs more than the two analytic gates G261/G262:
#
#   (a) the legacy default path is unchanged at the library level: `fixed`/u/p/
#       `fv`/`fr`/`K`/`Cu`/`Cp` for the stage-1/2 configurations (all-`:free`
#       with a volume drive, the mixed per-wall `wall_u` lid, a per-wall
#       `wall_un`, a tangential `wall_t`, and the all-`:same` `bc=:rigid` path)
#       were recorded on a *clean export of a3b507e* (stage 2, `git archive`,
#       out_julia/agent/bc3_snapshot_base.log) before this stage's edits and are
#       compared here -- at 1e-12 relative against the portable fingerprints
#       (batch P3; see the portability block below), with the exact raw-byte
#       recording under `QUAST_BITWISE_PINS=1`.  The driver level is pinned by
#       the G150 state dump and G258 (a);
#   (b) the validation matrix: every documented accepted/refused combination of
#       `wall_bc`/`wall_u`/`wall_un`/`wall_t`/`wall_tn`, in `Params` and again in
#       the solver (a direct library call bypasses `Params`), with the message
#       content -- including the refusals of the two ill-posed open
#       configurations and of `:schur_cg`/`:lagged` on an open wall;
#   (c) the linear paths still work without the pressure pin: `:direct` (its
#       shift ladder), explicit `:mg`, and `:auto` agree on an open-wall case,
#       `:auto` reproduces the explicit answer, and the selector's decision is
#       unchanged (an `:open` wall does not move the load/operator decision);
#   (d) the restart guard records the open-wall data: a checkpoint written with
#       an open wall and no `wall_tn` is refused when resumed with a `wall_tn`
#       (the marker format is `_wall_value_marker`, one `none`/`custom` token per
#       wall), and the same-config resume is legal -- so the refusal is the data
#       change, not the open wall;
#   (e) the per-step Newton path refuses a per-wall configuration instead of
#       silently solving the single-`bc` (closed) problem with an `:open` wall
# (a pre-existing gap that stage 3 makes load-bearing).
#
# The pre-change fingerprints are not tautological: they were produced by the
# stage-2 tree in a separate export and this file only compares (the exact
# values are the env-gated byte pins, the default values the fingerprint bands
# measured in the portability block below).  The tolerances in (c)
# are the G260 conventions: `<= 1e-6` relative for direct-vs-MG (the design's
# G-path goal on this small tree), `<= 1e-10` for auto-vs-explicit-MG (G191).

using Test, QuaST, SHA, LinearAlgebra, SparseArrays

function g263_tree()
    qt = Quadtree(2)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

# Uniform `2^n x 2^n`-leaf tree (each leaf is one Q2 element), the G260 helper.
function g263_uniform(n::Int)
    qt = Quadtree(n)
    for _ in 1:n, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

g263_zero(x, y) = 0.0
g263_h(a) = isempty(a) ? "empty" :
            bytes2hex(sha256(reinterpret(UInt8, vec(collect(Float64, a)))))
function g263_hfixed(d::Dict{Int,Float64})
    b = UInt8[]
    for (k, v) in sort(collect(d))
        append!(b, reinterpret(UInt8, [Float64(k), v]))
    end
    return bytes2hex(sha256(b))
end

# ---------------------------------------------------------------------------
# P3 portability (batch P3, task 1).  Block (a) used to pin 22 raw-byte SHA-256
# digests of `fixed`/u/p/K/fv/fr/Cu/Cp recorded on a clean export of a3b507e.
# Raw Float64 bytes are only reproducible on the recording stack (libm/BLAS/
# Julia build; docs/dev/AUDIT_GATES.md §5.2), so the default path is portable:
#
#   * **in process**, each legacy configuration is solved twice: with the serial
#     assembly and with `assemble_threads=4` (the production S4 discipline, 4
#     contiguous leaf blocks).  The wall data must be bitwise equal, the
#     solution and the assembly equal to 1e-12 relative (measured on this tree:
#     `fixed`/u/p identical; K/Cu/Cp exact and fv/fr at 1.9e-17 round-off);
#   * the pre-restructure **values** are kept as portable fingerprints: the
#     `fixed`-dict semantics are *exact* (count, signed sum, sum of |values| --
#     integer/half-integer wall data, machine independent), and the l2 norms of
#     u, p and of the assembled blocks are compared at 1e-12 relative.
#     The l2 norms are magnitude locks: a pure sign/ownership change of a field is
#     caught by the exact `fixed`-dict functionals (the mutation campaign's
#     lid-sign flip fires only that assertion) and by the byte pins, not by an
#     l2 band -- stated, not hidden.
#     The fingerprints are not vacuous: a dropped lid moves B's `sum(fixed)` from 9
#     to 0, a dropped `wall_un` C's `sumabs` from 2.5 to 0, and a real assembly
#     defect is >=O(1e-2) (ten orders above the band; the 1e-6 perturbation
#     control in (a) is rejected).
#
# The exact legacy lock stays available on the recording stack:
# `QUAST_BITWISE_PINS=1 julia --project=. --threads=8 test/run_gates.jl
#  t_g263_open_validation.jl` runs the original 22 byte pins (plus the exact
# last-bit residual `== 5.457785285107014e-16`; the default path asserts the
# *semantic* residual band `res < 1e-12`, which is what the pin documented --
# the value is a round-off-level quantity, not portable in the last bit).
const QUAST_BITWISE_PINS = get(ENV, "QUAST_BITWISE_PINS", "") == "1"

g263_rel(a, b) = norm(vec(a) - vec(b)) / max(norm(vec(b)), 1.0e-300)
g263_rel_norm(n, r) = abs(n - r) / max(abs(r), 1.0e-300)
g263_fixed_ok(f::Dict{Int,Float64}, ref) =
    length(f) == ref[1] && sum(values(f)) == ref[2] &&
    sum(abs, values(f)) == ref[3]

# Pre-restructure fingerprints; the tree reproduces the recorded byte pins bit
# for bit, so these ARE the a3b507e values.
const G263_REF = (
    A=(ufixed=(36, 0.0, 0.0), u=0.03505731840523087,
       p=1.1964387235952676, K=76.82373765098032, fv=0.06858527503723187,
       fr=0.06858516672725049, Cu=11.224972160321824,
       Cp=4.898979485566356),
    B=(ufixed=(43, 9.0, 9.0), u=3.6685255500576854, p=11.78687780694598),
    C=(ufixed=(36, 0.0, 2.5), u=1.8691278434334009, p=19.72930244041604),
    D=(ufixed=(57, 0.0, 0.0), u=0.34027039433026157, p=4.555829660008786,
       fv=0.3679900360969937, fr=0.36324157862838957),
    E=(ufixed=(64, 0.0, 0.0), u=0.0, p=0.0))

@testset "G263 open-wall validation, paths and restart" begin
    qt = g263_tree()
    ss = make_semstokes(qt, 2)
    nu = ss.su.N
    np = ss.sp.N
    z0 = g263_zero
    eta1 = (x, y) -> 1.0
    lid = (x, y) -> 1.0
    tn = (x, y) -> 0.5
    tt = (x, y) -> 0.5

    # ---- (a) the legacy default path, portably ---------------------------
    # Configurations copied verbatim from out_julia/agent/bc3_snapshot.jl, whose
    # pre-change recording is bc3_snapshot_base.log.  (Config D's `wall_t` is the
    # constant 1.0 of the recording, so `tt` above is not used there.)
    # Each configuration is solved twice in this process (serial assembly and
    # `assemble_threads=4`) and the pre-change values are the G263_REF portable
    # fingerprints; the raw-byte pins are the QUAST_BITWISE_PINS branch below.
    t1 = (x, y) -> 1.0
    rA = solve_semquad_stokes_var(ss, qt, eta1, z0,
                                  (x, y) -> sinpi(x) * sinpi(y), z0, z0;
                                  bc=:free, linear=:direct)
    res_A = LAST_SEMSTOKES_RESIDUAL[]   # the pinned solve's own residual
    rA4 = solve_semquad_stokes_var(ss, qt, eta1, z0,
                                   (x, y) -> sinpi(x) * sinpi(y), z0, z0;
                                   bc=:free, linear=:direct,
                                   assemble_threads=4)
    @test rA4[3] == rA[3]                     # wall data: exact, in process
    @test g263_rel(rA4[1], rA[1]) <= 1.0e-12  # measured identical
    @test g263_rel(rA4[2], rA[2]) <= 1.0e-12  # measured identical
    @test g263_fixed_ok(rA[3], G263_REF.A.ufixed)
    @test g263_rel_norm(norm(rA[1]), G263_REF.A.u) <= 1.0e-12
    @test g263_rel_norm(norm(rA[2]), G263_REF.A.p) <= 1.0e-12
    # the *semantic* residual band: a last-bit quantity is not a portable pin,
    # so the default asserts the documentation-level fact -- the direct solve is
    # exact to <= 1e-12 (measured 5.46e-16 on the serial solve; a real solver
    # defect leaves >= 1e-6, cf. G209's measured 1e-8 residual of a 1e-14-shift
    # rung).  The exact value is asserted under QUAST_BITWISE_PINS below.
    @test res_A < 1.0e-12
    @test LAST_SEMSTOKES_PRESSURE_GAUGE[] === :pin
    rA2 = solve_semquad_stokes_var(ss, qt, eta1, z0,
                                   (x, y) -> sinpi(x) * sinpi(y), z0, z0;
                                   bc=:free, linear=:direct, assemble_only=true)
    rA2_4 = solve_semquad_stokes_var(ss, qt, eta1, z0,
                                     (x, y) -> sinpi(x) * sinpi(y), z0, z0;
                                     bc=:free, linear=:direct,
                                     assemble_only=true, assemble_threads=4)
    @test g263_rel(rA2_4.K.nzval, rA2.K.nzval) <= 1.0e-12    # measured 0.0
    @test g263_rel(rA2_4.fv, rA2.fv) <= 1.0e-12              # measured 1.9e-17
    @test g263_rel(rA2_4.fr, rA2.fr) <= 1.0e-12              # measured 1.9e-17
    @test g263_rel(rA2_4.Cu.nzval, rA2.Cu.nzval) <= 1.0e-12  # measured 0.0
    @test g263_rel(rA2_4.Cp.nzval, rA2.Cp.nzval) <= 1.0e-12  # measured 0.0
    for (k, ref) in ((:K, G263_REF.A.K), (:fv, G263_REF.A.fv),
                     (:fr, G263_REF.A.fr), (:Cu, G263_REF.A.Cu),
                     (:Cp, G263_REF.A.Cp))
        v = getfield(rA2, k)
        @test g263_rel_norm(v isa SparseMatrixCSC ? norm(v.nzval) : norm(v),
                            ref) <= 1.0e-12
    end
    @test size(rA2.Cp, 2) == np - 1            # the historical pin
    # non-vacuity: a 1e-6 relative change of the solution is rejected by the
    # 1e-12 band (a real wall/path defect is >=O(1e-2); the exact `fixed`-dict
    # semantics above reject a dropped wall row outright).
    @test g263_rel(rA[1] .* (1.0 + 1.0e-6), rA[1]) > 1.0e-12

    rB = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                  bc=:free,
                                  wall_bc=(:free, :free, :free, :rigid),
                                  wall_u=(nothing, nothing, nothing, lid),
                                  linear=:direct)
    rB4 = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                   bc=:free,
                                   wall_bc=(:free, :free, :free, :rigid),
                                   wall_u=(nothing, nothing, nothing, lid),
                                   linear=:direct, assemble_threads=4)
    @test rB4[3] == rB[3]
    @test g263_rel(rB4[1], rB[1]) <= 1.0e-12
    @test g263_rel(rB4[2], rB[2]) <= 1.0e-12
    @test g263_fixed_ok(rB[3], G263_REF.B.ufixed)
    @test g263_rel_norm(norm(rB[1]), G263_REF.B.u) <= 1.0e-12
    @test g263_rel_norm(norm(rB[2]), G263_REF.B.p) <= 1.0e-12

    rC = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                  bc=:free,
                                  wall_un=((x, y) -> y - 0.5, nothing, nothing,
                                           nothing),
                                  linear=:direct)
    rC4 = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                   bc=:free,
                                   wall_un=((x, y) -> y - 0.5, nothing, nothing,
                                            nothing),
                                   linear=:direct, assemble_threads=4)
    @test rC4[3] == rC[3]
    @test g263_rel(rC4[1], rC[1]) <= 1.0e-12
    @test g263_rel(rC4[2], rC[2]) <= 1.0e-12
    @test g263_fixed_ok(rC[3], G263_REF.C.ufixed)
    @test g263_rel_norm(norm(rC[1]), G263_REF.C.u) <= 1.0e-12
    @test g263_rel_norm(norm(rC[2]), G263_REF.C.p) <= 1.0e-12

    rD = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                  bc=:free,
                                  wall_bc=(:rigid, :rigid, :rigid, :free),
                                  wall_t=(nothing, nothing, nothing, t1),
                                  linear=:direct)
    rD4 = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                   bc=:free,
                                   wall_bc=(:rigid, :rigid, :rigid, :free),
                                   wall_t=(nothing, nothing, nothing, t1),
                                   linear=:direct, assemble_threads=4)
    @test rD4[3] == rD[3]
    @test g263_rel(rD4[1], rD[1]) <= 1.0e-12
    @test g263_rel(rD4[2], rD[2]) <= 1.0e-12
    @test g263_fixed_ok(rD[3], G263_REF.D.ufixed)
    @test g263_rel_norm(norm(rD[1]), G263_REF.D.u) <= 1.0e-12
    @test g263_rel_norm(norm(rD[2]), G263_REF.D.p) <= 1.0e-12
    rD2 = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                   bc=:free,
                                   wall_bc=(:rigid, :rigid, :rigid, :free),
                                   wall_t=(nothing, nothing, nothing, t1),
                                   linear=:direct, assemble_only=true)
    rD2_4 = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                     bc=:free,
                                     wall_bc=(:rigid, :rigid, :rigid, :free),
                                     wall_t=(nothing, nothing, nothing, t1),
                                     linear=:direct, assemble_only=true,
                                     assemble_threads=4)
    @test g263_rel(rD2_4.fv, rD2.fv) <= 1.0e-12              # measured 0.0
    @test g263_rel(rD2_4.fr, rD2.fr) <= 1.0e-12              # measured 0.0
    @test g263_rel_norm(norm(rD2.fv), G263_REF.D.fv) <= 1.0e-12
    @test g263_rel_norm(norm(rD2.fr), G263_REF.D.fr) <= 1.0e-12
    @test LAST_SEMSTOKES_PRESSURE_GAUGE[] === :pin

    rE = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                  bc=:rigid, linear=:direct)
    rE4 = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                   bc=:rigid, linear=:direct,
                                   assemble_threads=4)
    @test rE4[3] == rE[3]
    @test g263_rel(rE4[1], rE[1]) <= 1.0e-12
    @test g263_rel(rE4[2], rE[2]) <= 1.0e-12
    @test g263_fixed_ok(rE[3], G263_REF.E.ufixed)
    @test all(iszero, rE[1]) && all(iszero, rE[2])    # zero load, rigid walls

    # (a2) the exact pre-change recording: raw bytes and the last-bit residual,
    # only on the recording stack (QUAST_BITWISE_PINS=1).
    if QUAST_BITWISE_PINS
        @test g263_hfixed(rA[3]) ==
              "4cd640f81a1ec08ade30f5898fde6e3690c50e68d6ed4f48a3a8174a0a4f64a7"
        @test g263_h(rA[1]) ==
              "ef8ea05bbfaef60b419bcb621f05cc866d89f65a1ae0339bcc1f722a81bc1ef6"
        @test g263_h(rA[2]) ==
              "c997b4fcb7e902ab7c6d33f500aa355018f252b7418328c6559ddd6ce288514e"
        @test res_A == 5.457785285107014e-16
        @test g263_h(rA2.K) ==
              "0588961cf88da671bbb8b4b01d1ff7ea30964cebb28e1ef1821a26932141e839"
        @test g263_h(rA2.fv) ==
              "efd5e57e04c51af8c4448b368fa0ae8e9e43d2615f5adbdfced84d6bf85d1f1c"
        @test g263_h(rA2.fr) ==
              "e8ea8d09075658f0d9fcebf34c3c959254df7f72e49ba2fa79f50395c73ccbd3"
        @test g263_h(rA2.Cu) ==
              "cc50de131b2b80953400d61f1c106e2d21879ec46bada1d2d7b7ae570e821311"
        @test g263_h(rA2.Cp) ==
              "7e494aa54afe4274532b27f5578f22266fda8be22227ae8466ac72af8ef88686"
        @test g263_hfixed(rB[3]) ==
              "056afc4889fdf670135d8cfde8e737f95895e676ce1cd575ec16ea1775f6207f"
        @test g263_h(rB[1]) ==
              "204ddea0f3645fde1090892109a6b9e6e0148edfee2c54b3eacceca53390449b"
        @test g263_h(rB[2]) ==
              "9f4ce2ed910126679a57ad7a982d50a2e07d22d8864a20d0cb515ad337ef8808"
        @test g263_hfixed(rC[3]) ==
              "9b1e6d5126a8e6abe14d52e7bf7e07fe482f6c4874151a5da8a31903695e396d"
        @test g263_h(rC[1]) ==
              "7a18c19807f1802ad5bdcb65c60f895d7e704c8938ba7b84aeb9a8bfc77b61d7"
        @test g263_h(rC[2]) ==
              "562881d5452a7da04df4c0e4bb87a28e0b5ea633dcf0fd5ed48b51e0beec3202"
        @test g263_hfixed(rD[3]) ==
              "1573f29f41bbf67d1e0776bd85485accfa178de88a10b509f353bad4f7cef7ae"
        @test g263_h(rD[1]) ==
              "c7b47006fb1f9d2db9f1a86576d0336d20a6068edc584db2f5083b0c84c01bbd"
        @test g263_h(rD[2]) ==
              "117a56c60a48a399b1ce792643c1f4e34ae7034b2cfb4fb8be445247ba6339ee"
        @test g263_h(rD2.fv) ==
              "d07847431136c6cb276e906ecb18d32e56a0ac373511dcd8a42584a1cc3eff84"
        @test g263_h(rD2.fr) ==
              "6f6b3736a51a5c254428ec559b7a89c27bf5a7895a02a1710a9c0ab635ca945c"
        @test g263_hfixed(rE[3]) ==
              "e692807d78e4cb7959fdcbae510f3f81bd8b736f38260d1d2526db74968bb1cc"
        @test g263_h(rE[1]) ==
              "94637c6efefbdcc3d3bb74d61732b22250552654c8c11f0fa9c3b3ed11d38373"
        @test g263_h(rE[2]) ==
              "6d9c54dee5660c46886f32d80e57e9dd0ffa57ee0cd2a762b036d9c8e0c3a33a"
    end

    # ---- (b) the validation matrix ---------------------------------------
    # Accepted: every legal data value on the wall that carries it.
    accept = [
        (name="open wall, no data",
         pr=(wall_bc=(:open, :rigid, :rigid, :rigid),),
         solve=(wall_bc=(:open, :rigid, :rigid, :rigid),)),
        (name="open wall + normal traction",
         pr=(wall_bc=(:open, :rigid, :rigid, :rigid),
             wall_tn=(tn, nothing, nothing, nothing)),
         solve=(wall_bc=(:open, :rigid, :rigid, :rigid),
                wall_tn=(tn, nothing, nothing, nothing))),
        (name="open wall + tangential traction",
         pr=(wall_bc=(:open, :rigid, :rigid, :rigid),
             wall_t=(tt, nothing, nothing, nothing)),
         solve=(wall_bc=(:open, :rigid, :rigid, :rigid),
                wall_t=(tt, nothing, nothing, nothing))),
        (name="open wall + both tractions",
         pr=(wall_bc=(:free, :free, :free, :open),
             wall_t=(nothing, nothing, nothing, tt),
             wall_tn=(nothing, nothing, nothing, tn)),
         solve=(wall_bc=(:free, :free, :free, :open),
                wall_t=(nothing, nothing, nothing, tt),
                wall_tn=(nothing, nothing, nothing, tn))),
        (name="free wall + tangential traction",
         pr=(wall_t=(nothing, nothing, nothing, tt),),
         solve=(wall_t=(nothing, nothing, nothing, tt),)),
        (name="free wall + normal velocity",
         pr=(wall_un=((x, y) -> y - 0.5, nothing, nothing, nothing),),
         solve=(wall_un=((x, y) -> y - 0.5, nothing, nothing, nothing),)),
        (name="rigid wall + tangential velocity",
         pr=(bc=:rigid, wall_u=(nothing, nothing, nothing, lid)),
         solve=(bc=:rigid, wall_u=(nothing, nothing, nothing, lid))),
        (name="mixed closed free/rigid",
         pr=(wall_bc=(:free, :free, :free, :rigid),),
         solve=(wall_bc=(:free, :free, :free, :rigid),)),
    ]
    for c in accept
        @test Params(; c.pr...) isa Params
        asm = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                       bc=get(c.solve, :bc, :free),
                                       c.solve..., linear=:direct,
                                       assemble_only=true)
        @test asm isa NamedTuple
        @test LAST_SEMSTOKES_PRESSURE_GAUGE[] ===
              (any(w -> w === :open, get(c.solve, :wall_bc,
                                         (:same, :same, :same, :same))) ?
               :none : :pin)
    end

    # Refused: the data on a wall whose component is Dirichlet is a silent no-op,
    # so it is an error in both layers, with an explanatory message.
    reject = [
        (name="wall_u on :open", pr=(wall_bc=(:free, :free, :free, :open),
                                     wall_u=(nothing, nothing, nothing, lid)),
         solve=(wall_bc=(:free, :free, :free, :open),
                wall_u=(nothing, nothing, nothing, lid)),
         pats=["tangential", "ignored"]),
        (name="wall_un on :open", pr=(wall_bc=(:free, :free, :free, :open),
                                      wall_un=(nothing, nothing, nothing, lid)),
         solve=(wall_bc=(:free, :free, :free, :open),
                wall_un=(nothing, nothing, nothing, lid)),
         pats=["normal", "ignored"]),
        (name="wall_tn on :free", pr=(wall_tn=(nothing, nothing, nothing, tn),),
         solve=(wall_tn=(nothing, nothing, nothing, tn),),
         pats=["normal traction", ":open"]),
        (name="wall_tn on :rigid",
         pr=(bc=:rigid, wall_tn=(tn, nothing, nothing, nothing)),
         solve=(bc=:rigid, wall_tn=(tn, nothing, nothing, nothing)),
         pats=["normal traction", ":open"]),
        (name="wall_t on :rigid",
         pr=(bc=:rigid, wall_t=(tt, nothing, nothing, nothing)),
         solve=(bc=:rigid, wall_t=(tt, nothing, nothing, nothing)),
         pats=["tangential", "Dirichlet", "silently dropped"]),
        (name="wall_u on :free",
         pr=(bc=:free, wall_u=(nothing, nothing, nothing, lid)),
         solve=(bc=:free, wall_u=(nothing, nothing, nothing, lid)),
         pats=["tangential", "ignored"]),
        (name="all four walls :open",
         pr=(wall_bc=(:open, :open, :open, :open),),
         solve=(wall_bc=(:open, :open, :open, :open),),
         pats=["pure traction", "rigid motion"]),
        (name="both x-walls :open",
         pr=(wall_bc=(:open, :open, :free, :free),),
         solve=(wall_bc=(:open, :open, :free, :free),),
         pats=["x-translation", "Gram"]),
        (name="both y-walls :open",
         pr=(wall_bc=(:free, :free, :open, :open),),
         solve=(wall_bc=(:free, :free, :open, :open),),
         pats=["y-translation", "Gram"]),
        (name="unknown wall_bc value",
         pr=(wall_bc=(:open, :bogus, :free, :free),),
         solve=(wall_bc=(:open, :bogus, :free, :free),),
         pats=["wall_bc entries"]),
    ]
    for c in reject
        @test_throws ArgumentError Params(; c.pr...)
        err = try
            solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                     bc=get(c.solve, :bc, :free),
                                     c.solve..., linear=:direct)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = err isa ArgumentError ? sprint(showerror, err) : ""
        for p in c.pats
            @test occursin(p, msg)
        end
    end
    # the linear paths with their own gauge/scaling assumptions are refused on
    # an open-wall configuration, and accepted on a closed one (G44/G207 run the
    # closed paths; this pins only the refusal boundary).
    open_cfg = (bc=:free, wall_bc=(:rigid, :rigid, :rigid, :open))
    for lin in (:schur_cg, :lagged)
        errl = try
            solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                     open_cfg..., linear=lin)
            nothing
        catch e
            e
        end
        @test errl isa ArgumentError
        msgl = errl isa ArgumentError ? sprint(showerror, errl) : ""
        @test occursin("not supported with an :open", msgl)
        @test occursin(string(lin), msgl)
    end
    # ... and the three supported paths are accepted (they are compared below)
    for lin in (:direct, :mg, :auto)
        @test solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                       open_cfg..., linear=lin) isa Tuple
        @test LAST_SEMSTOKES_PRESSURE_GAUGE[] === :none
    end

    # ---- (c) direct / mg / auto agree without the pressure pin -----------
    # G260's L4 tree: 256 leaves, 2 467 saddle dofs, so `:auto` resolves to MG.
    qt4 = g263_uniform(4)
    ss4 = make_semstokes(qt4, 2)
    t4 = (x, y) -> 1.0 + 0.25 * x
    cfg4 = (bc=:free, wall_bc=(:rigid, :rigid, :rigid, :open),
            wall_tn=(nothing, nothing, nothing, t4))
    rd = solve_semquad_stokes_var(ss4, qt4, eta1, z0, z0, z0, z0;
                                  cfg4..., linear=:direct)
    @test LAST_SEMSTOKES_PRESSURE_GAUGE[] === :none
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-9
    nfb = LAST_AUTO_FALLBACK[].count
    ra = solve_semquad_stokes_var(ss4, qt4, eta1, z0, z0, z0, z0;
                                  cfg4..., linear=:auto)
    ch = LAST_SEMSTOKES_AUTO_CHOICE[]
    @test ch.choice === :mg
    @test ch.saddle_dofs == 2 * ss4.su.N + ss4.sp.N
    @test LAST_AUTO_FALLBACK[].count == nfb        # no fallback happened
    @test LAST_SEMSTOKES_PRESSURE_GAUGE[] === :none
    rmg = solve_semquad_stokes_var(ss4, qt4, eta1, z0, z0, z0, z0;
                                  cfg4..., linear=:mg,
                                  mg_operator=:matrixfree, mg_threads=1,
                                  mg_tol=1.0e-10)
    @test LAST_SEMSTOKES_RESIDUAL[] <= 1.0e-9
    @test norm(rd[1] - rmg[1]) / max(norm(rd[1]), 1.0) <= 1.0e-6
    @test norm(rd[2] - rmg[2]) / max(norm(rd[2]), 1.0) <= 1.0e-6
    @test norm(ra[1] - rmg[1]) / max(norm(rmg[1]), 1.0) <= 1.0e-10
    @test norm(ra[2] - rmg[2]) / max(norm(rmg[2]), 1.0) <= 1.0e-10
    @test maximum(abs, rd[1]) > 1.0e-3            # the load drives the flow
    @test maximum(abs, rd[2]) > 1.0e-3
    # the assembled system of the open case: unpinned pressure, loaded RHS only
    a4 = solve_semquad_stokes_var(ss4, qt4, eta1, z0, z0, z0, z0;
                                  cfg4..., linear=:direct, assemble_only=true)
    @test size(a4.Cp, 2) == ss4.sp.N
    @test size(a4.Bs, 1) == ss4.sp.N
    @test all(iszero, a4.fv[1:ss4.su.N])         # top y-wall load is u_z only
    @test sum(a4.fv[ss4.su.N + 1:end]) ≈ 1.125 rtol = 1.0e-14
    # On the small L3 tree `:auto` resolves to direct and is bitwise the direct
    # answer (the load and the open wall do not enter the selector).
    qt3 = g263_uniform(3)
    ss3 = make_semstokes(qt3, 2)
    ad = solve_semquad_stokes_var(ss3, qt3, eta1, z0, z0, z0, z0;
                                  cfg4..., linear=:auto)
    @test LAST_SEMSTOKES_AUTO_CHOICE[].choice === :direct
    dd = solve_semquad_stokes_var(ss3, qt3, eta1, z0, z0, z0, z0;
                                  cfg4..., linear=:direct)
    @test ad[1] == dd[1] && ad[2] == dd[2] && ad[3] == dd[3]

    # ---- (d) the restart guard records the open-wall data -----------------
    ckpath = joinpath(tempdir(), "g263_open_wall_restart.h5")
    qtck = g263_uniform(3)
    ssck = make_semstokes(qtck, 2)
    thck = zeros(Float64, ssck.su.N)
    ckkw = (p=2, tstop=2.0e-3, dt0=1.0e-3, dtmax=1.0e-3, dtmin=1.0e-16,
            amr_every=0, ev=0.0, linear=:direct, maxpic=1, verbose=false,
            mass_form=:boussinesq)
    prck = Params(Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0, bc=:free,
                  wall_bc=(:free, :free, :free, :open))
    rck = run_semquad_simulation(qtck, prck, copy(thck); ckkw...,
                                 checkpoint_path=ckpath, checkpoint_every=1)
    @test all(isfinite, rck.ux) && all(isfinite, rck.pv)
    # the same configuration resumes: the open wall itself is not a refusal
    rok = run_semquad_simulation(qtck, prck, copy(thck); ckkw...,
                                 restart_from=ckpath)
    @test all(isfinite, rok.ux)
    # a checkpoint written without wall_tn cannot be resumed with one
    prck2 = Params(Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0, bc=:free,
                   wall_bc=(:free, :free, :free, :open),
                   wall_tn=(nothing, nothing, nothing, tn))
    errck = try
        run_semquad_simulation(qtck, prck2, copy(thck); ckkw...,
                              restart_from=ckpath)
        nothing
    catch e
        e
    end
    @test errck isa ArgumentError
    msgck = errck isa ArgumentError ? sprint(showerror, errck) : ""
    @test occursin("restart configuration mismatch", msgck)
    @test occursin("wall_tn", msgck)
    @test occursin("checkpoint=none,none,none,none", msgck)
    @test occursin("current=none,none,none,custom", msgck)
    # ... and the guard records it (source assertion: the marker is part of the
    # trajectory, the same convention as wall_u/wall_un/wall_t; G245 pins the
    # marker helper itself and stays untouched by this stage)
    guardsrc = read(normpath(joinpath(@__DIR__, "..", "src", "driver",
                                      "restart_guard.jl")), String)
    @test occursin("run_config[\"wall_tn\"] = _wall_value_marker(pr.wall_tn)",
                   guardsrc)
    @test QuaST._wall_value_marker(prck2.wall_tn) == "none,none,none,custom"
    rm(ckpath; force=true)

    # ---- (e) the per-step Newton path refuses per-wall data --------------
    errn = try
        run_semquad_simulation(qtck, prck, copy(thck); ckkw...,
                               pic_mode=:newton)
        nothing
    catch e
        e
    end
    @test errn isa ArgumentError
    msgn = errn isa ArgumentError ? sprint(showerror, errn) : ""
    @test occursin("pic_mode=:newton ignores", msgn)
    @test occursin(":open", msgn)
    @test occursin("wall_bc", msgn)
    # the default per-wall configuration is still legal on the Newton path (the
    # refusal is about the silently ignored data, not about Newton itself)
    prn0 = Params(Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0, bc=:free)
    rn0 = run_semquad_simulation(qtck, prn0, copy(thck); ckkw...,
                                 pic_mode=:newton)
    @test all(isfinite, rn0.ux)
end

