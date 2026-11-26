# blankenbach2b: QuaST `Nu` and `Vrms` on the Blankenbach 2b card, three meshes

- **Quantity.** The three rows of the public 2b ladder in `docs/en/validation.md`
  (section "Steady benchmarks"), re-traced to the runs that produced them.  Case 2b is
  the 2.5 x 1 box, `Ra = 1e4`, no heating, `eta = exp(-b T + c(1-y))` with
  `b = ln 16384`, `c = ln 64`; the wall rule holds every leaf within the wall strip and
  the wall rows at a minimum level.

  | mesh | leaves | steps | wall clock | `Nu_top` | vs published 6.9299 | `Nu_bot` | `Vrms` | vs published 171.755 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | level 3 (40 x 16) | 640 | 5 492 | 20 530 s | **7.248805** | +4.60 % | 7.250974 | **170.213063** | -0.90 % |
  | level 3 + walls 4 | 1 120 | 5 674 | 6 930 s | **7.212885** | +4.08 % | 7.052840 | **170.633086** | -0.65 % |
  | level 4 + walls 5 | 4 000 | 7 550 | (no summary) | **6.992620** | +0.905 % | 6.965590 | **171.633560** | -0.070 % |

  The level-3 rows end with the case's closing summary.  The level-4 + walls-5 run does
  **not**: the host's out-of-memory killer stopped it after the last printed step
  (step 7 550, `t = 0.673`), so its evidence is that last step, where the top and bottom
  fluxes agree to 0.39 % -- the steady-state acceptance test reported on the page.  This
  is why the page calls the 2b rows the least certain ones in the table.
- **Files.** `logs/blankenbach2b_quaST_L3.log` (sha256 `20742d1cb0736221091f97df70eff1715980801c199682716bfb99750c89584a`, 6609 B),
  `logs/blankenbach2b_quaST_L3-walls4.log` (sha256 `1061a6809eb4754b919a24f502e56d2481eff9bd573721a5d7c108e4457d92dd`, 6846 B),
  `logs/blankenbach2b_quaST_L4-walls5.log` (sha256 `8dafdf937daf9afbbc1460a0edbd384950b4dcc53c04564290e09545bffdc6fb`, 8937 B).
  Each is a trimmed record: every 200th step line, the closing summary, the final steps; the
  full logs are in the development tree, and the quoted values are the closing-summary or
  last-step values, not the sampled ones.
- **Cross-code context.**  The pinned ASPECT image was run on its own shipped 2b card at the
  same 640-cell mesh (`aspc-blankenbach2b_nu-vrms_ref3.md`): ASPECT reads `Nu = 6.445128`
  (-6.995 % from the published value) where QuaST reads 7.248805 (+4.60 %), while `Vrms`
  agrees to 0.97 %.  Neither code is converged in the wall flux at 40 x 16, and the two
  read the wall flux differently (QuaST: one-sided GLL wall gradient, family A; ASPECT: its
  consistent boundary-flux statistic, family C), which is why the same-mesh `Nu` difference
  is 12.5 % while the velocity field agrees to about 1 %.  At the finer meshes both codes
  approach the published value from their own side; the 2b comparison is therefore a
  resolution-difference demonstration, not a physical disagreement, and the +12.5 % must not
  be quoted as a cross-code physics error.
- **Status.** Archived from the development campaign of 2026-11-14/15 (the same logs the
  public table cites); not re-run for this record.
