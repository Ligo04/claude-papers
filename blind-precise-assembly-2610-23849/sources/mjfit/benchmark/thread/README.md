# Bolt and nut benchmark: M10×1.5 thread

The comparison of the [peg benchmark](../README.md), run on a threaded
fastener, where both parts are concave and every flank is a contact surface.
Five representations of the same exact parts:

- **ours:** mjfit's convex wedges (`mjfit.thread`), with a primitive core
  cylinder in the bolt; the knob is segments per turn;
- **CoACD** and **V-HACD:** convex decompositions of exact watertight meshes
  of the bolt and nut;
- **SDF, mesh:** the same meshes as MuJoCo mesh SDFs; the knob is octree depth;
- **SDF, analytic:** the exact shape as a MuJoCo SDF plugin
  ([`plugin/`](plugin/)), so its only error is MuJoCo's SDF contact search,
  whose effort (initial points × gradient iterations) is the knob.

Every probe is passive and has an answer computed from the exact helical
geometry. Tables: [`../report/thread_summary.md`](../report/thread_summary.md);
figures: `../report/pareto_thread_cpu.png`, `pareto_thread_memory.png`,
`thread_start.png`, `thread_partial.png`, `thread_torque.png`.

## Task

An ISO 68-1 M10×1.5 thread: a 16 mm threaded rod, fixed, and an 8 mm steel
nut (Ø16 mm, about five engaged turns). The clearance c is the gap between
facing flanks with the nut centred (`mjfit.thread`'s `clearance`), 100 and
20 µm: the axial play is ±2c/√3 (±115 and ±23 µm) and the radial play c/2.
Every representation's nut gets the exact nut's mass and inertia, so they
differ only in contact. Contact parameters are the peg benchmark's (1 ms step,
solref 2 ms, zero margin). The exact geometry is in [`geometry.py`](geometry.py):
a point is inside the thread when its meridian coordinates
(r, z − P·φ/2π mod P) lie in the tooth profile; distances are measured in that
plane, within 0.1 % of the 3D distance at the 3° helix angle.

| probe | setup | exact answer |
|---|---|---|
| T1 axial | axial slide, frictionless, ±0.1 N | play 2c/√3 each way |
| T2 radial | lateral slides, frictionless, 0.1 N in 12 directions | play c/2 |
| T3 tilt | hinge through the nut centre, 1.5 mN·m, two axes | the computed maximum tilt, error at the nut faces |
| T4 screw | axial slide + spin hinge, frictionless, 0.12 mN·m against 0.1 N | follows the helix z = P·θ/2π, seated 2c/√3 low |
| T5 spin | free nut, gravity, frictionless; and at μ = 0.3 | descends at g / (1 + I(2π/P)²/m); at μ = 0.3 it holds |
| T6 start | nut resting on the bolt's top face under T4's load, turned by 5× (and 2×) the exact back-off torque, unscrewing (`back`) or screwing on (`on`) | the quasi-static path: `back` clicks down by 4c/√3 and rides the flank back up; `on` enters and follows the helix |
| T7 partial | T2 and T3 with the nut threaded ¼, ½, 1 and 2 turns on, hinge half-way along the engaged length | radial reach and first-contact tilt on the sampled exact surfaces, end faces included |
| T8 torque | engaged nut, no joint damping, axial load 0.1 or 1 N, μ = 0.1, 0.3, 1, torque ramped until it turns 5° | the power-screw formula (`geometry.screw_torque`, textbook rather than exact) |

Errors are in µm of wall position, negative when phantom material stops the
nut early. T1–T4 also record the true signed distance at every reported
contact. Settings: [`../configs/thread.yaml`](../configs/thread.yaml).
Every setting runs T1–T5; T6 and T7 run ours and both SDFs, and T8 runs ours
and the analytic SDF at c = 100 µm.

## Key results

Worst settled error over T1–T4 and CPU time per step (one thread, the nut
screwed up the thread under load), MULTICCD off, from
`../report/thread_summary_c{100,20}.csv`; "fills" means the parts overlap at
the centred start, so the nut is held by phantom material.

| setting | c = 100 µm | CPU | c = 20 µm | CPU |
|---|---:|---:|---:|---:|
| ours, 12 segments per turn | 87 µm | 0.46 ms | 86 µm | 0.49 ms |
| ours, 24 | 20 µm | 0.93 ms | 20 µm | 1.0 ms |
| ours, 48 | 4.4 µm | 1.9 ms | 4.5 µm | 2.1 ms |
| ours, 96 | 0.92 µm | 4.1 ms | 2.5 µm | 4.3 ms |
| CoACD, threshold 0.1 … 0.02 (up to 586 hulls) | fills | 0.95–9.6 ms | fills | 2.0–17 ms |
| V-HACD, 64 … 4096 hulls | fills | 0.9–30 ms | fills | 1.0–48 ms |
| SDF, mesh, depth 6 … 9 | fills | 0.74–0.98 ms | fills | 0.72–0.91 ms |
| SDF, analytic, 40 × 10 (MuJoCo's default) | 91 µm | 1.8 ms | 53 µm | 1.6 ms |
| SDF, analytic, 300 × 60 | 22 µm | 13 ms | 21 µm | 12 ms |

- **Only ours keeps the clearance open, and it converges.** Its error is the
  discretization, one-sided by construction (extra play, never
  interference): the axial error (T1) falls from 87 µm at 12 segments to
  4.4 µm at 48 and 0.03 µm at 192. From 96 segments the worst error sits at
  0.9–2.5 µm, set by the loaded screwing probe.
- **At equal cost, ours is far more accurate than the analytic SDF.** At
  about 4 ms per step ours (96 segments) is within 1 µm at c = 100 µm and the
  analytic SDF (100 × 30) 52 µm off; ours at 24 segments (0.93 ms, 20 µm)
  already beats the analytic SDF's finest search (13 ms, 22 µm). With the
  exact shape, the SDF's remaining error is its contact search: it finds
  only a few contacts along five turns of thread.
- **Generic decomposition fails on threads.** Every CoACD and V-HACD setting
  overlaps the parts at the start, and all but one (CoACD 0.1 at c = 20 µm)
  jam or skip when screwed. The mesh SDF's octree rounds the 60° tooth, so
  even at depth 9 (1.7 GB) the parts overlap, and the torqued nut skips
  through the thread or jams.
- **Frictionless spin-down (T5) is ours' weak spot.** The analytic SDF
  descends at 100 % of the exact rate; ours, MULTICCD off, at 31–80 % with
  24–64 segments, 14–28 % finer, and 2–5 % at 12–16 segments. Sliding
  across the seams between wedges produces internal-edge contacts that point
  around the axis and snag the nut. At μ = 0.3 (a self-locking thread) ours
  holds, with only its discretization play as error.
- **Turn MULTICCD off for thread wedges.** With it on, MuJoCo sometimes
  reports a spurious deep contact between thin wedges, which jams the screw
  probe (64–128 segments at c = 100 µm) and holds the nut on the top face in
  T6; where it does not misfire it costs 15–40 % more time.

**Starting a nut (T6, T7).** The exact back-spin click is the axial play:
4c/√3, 231 µm at c = 100 µm. With MULTICCD off, ours clicks and screws on at
every resolution (one chaotic exception: 128 segments at c = 20 µm jams after
the click at 5× the torque, yet clicks at 2×). From 64 segments the click is
within 2 µm of axial phase; coarser wedges click early by their extra axial
play. Off the drops, ours' height stays within about 9–10 µm at fine
resolutions. At 2× the exact torque, 12–24 segments jam right after the click
on the facets' seams. The analytic SDF clicks too, up to 150 µm early at
MuJoCo's default search effort; the mesh SDF overlaps the parts at the start
at every depth and never engages. A nut threaded on only ¼–½ turn is loose:
radial play up to 2c (100 µm against 50 µm fully engaged) and tilt up to
44 mrad (12.4 mrad fully engaged). Ours' radial error there is about 1 µm or
less from one turn on (24 segments and finer), and its tilt error falls to
0.5–11 µm from 64 segments up (the largest where the nut chatters without
settling); the analytic SDF is 12–56 µm off at its finest search
(c = 100 µm).

**Turning torque (T8).** Breakaway torque over the power-screw formula,
turning against / with the load, ours at 48 segments, MULTICCD off,
c = 100 µm ("> 3": never turned within 3×; "0": back-drove):

| setup | μ = 0.1 | μ = 0.3 | μ = 1 |
|---|---|---|---|
| 0.1 N, condim 3 | > 3 / 1.49 | 1.22 / 1.28 | 1.20 / 0.97 |
| 0.1 N, condim 3, elliptic cone | 2.79 / 1.48 | 1.20 / 1.22 | 1.16 / 1.18 |
| 0.1 N, condim 4 | > 3 / > 3 | 2.41 / 3.40 | 1.28 / 1.53 |
| 1 N, condim 3 | > 3 / 0 | 1.63 / 1.81 | 1.28 / 1.32 |
| 1 N, condim 3, elliptic cone | 2.54 / 5.71 | 1.60 / 1.92 | 1.28 / 1.35 |
| 1 N, condim 3, elliptic, 0.25 ms step | 1.48 / 1.50 | 1.24 / 1.27 | 1.14 / 1.22 |

At μ ≥ 0.3, 0.1 N and condim 3, ours (MULTICCD off) needs 1.0–1.7× the
formula at every resolution, and the analytic SDF 1.0–1.3× at every μ; at
μ = 0.1 ours' facets dominate. Condim 4 adds torsional friction at every
flank contact, so use condim 3 on threads (mjfit.thread's default). A 1 N
load on the 8 g nut is 120 m/s², which at a 1 ms step lets the nut fall about
its whole axial play between contacts; a 0.25 ms step restores the
light-load behaviour.

## Raw results

```
../results/thread/probes_cpu/  memory  axial  radial  tilt  screw  spin   (T1–T5, model size)
                               start  partial                         (T6, T7)
                               torque                                 (T8)
../results/thread/speed_cpu    CPU µs per step, contacts, geoms, per setting and clearance
```

Rows are keyed by `label` (method and parameters), `multiccd`, `clearance_um`
and the probe's own variables (`direction`, `theta`, `phi_deg`, `turns`,
`kind`, `angle_deg`, `mu`, `load_N`, `condim`, `cone`, `timestep_ms`). Two
caveats: ours' timings were measured on wedges without the partial turn at
each part's end, about 9 % fewer pieces than the probed ones (timing the
current wedges costs 0–5 % more, 24–96 segments), and one timing is left out
of the time plots because the nut was ejected (mesh SDF depth 9, c = 20 µm).
CoACD at threshold 0.01 (about 3,800 hulls) is not in the results; its
decompositions are in the release cache and `--stage thread-coacd` runs it.

## Running

```bash
python -m benchmark.reproduce --stage thread --dry-run    # the four commands below
python -m benchmark.thread.run --parts probes --probes memory axial radial tilt screw spin
python -m benchmark.thread.run --parts probes --probes start partial --methods ours sdf sdf_analytic
python -m benchmark.thread.run --parts probes --probes torque --methods ours sdf_analytic --clearance 100
python -m benchmark.thread.run --parts speed               # one process: run on an idle machine
python -m benchmark.thread.report --traces                 # tables + figures -> ../report/
```

Each setting is decomposed (or loaded from `../cache/`) in its own process,
and its probe jobs start as soon as it is ready; `--resume` runs only the jobs
missing from the stored tables. With the release cache
(`python -m benchmark.fetch cache`) the four commands take roughly 20, 20, 25
and 20 minutes on 32 threads; without it, add about 30 minutes of
decompositions (CoACD 0.02: 20–30 min per clearance) and 10–20 minutes of
exact T7 answers.

**The analytic SDF plugin** ([`plugin/iso_thread_sdf.cc`](plugin/iso_thread_sdf.cc))
evaluates the exact distance, a C++ port of `geometry.py`. It is compiled on
first use with g++ (C++17) against the `libmujoco.so` inside the installed
`mujoco` wheel and stored in `../cache/`; this needs Linux and a C++
compiler, but no separate MuJoCo install. ISO threads use its built-in tooth;
any other tooth, start count, hand or chamfer is passed as attributes.

## Videos

`python -m benchmark.reproduce --stage videos` renders them (MUJOCO_GL=egl),
or `python -m benchmark.fetch videos` downloads them into `../results/render/`:
screwing (T4) and spin-down (T5), a coarse and a fine setting per method;
a nut-runner run-down at μ = 0.3 (ours at 24 and 96 segments with 5 µm seam
relief against both SDFs) at both clearances; side-by-side run-downs with
each tile's real-time factor (`thread_compare_*`); the same run-down on other
thread forms ([`forms.py`](forms.py): trapezoidal, square, buttress, round
wire, and the FurnitureBench leg); and the generator's range on unusual
designs ([`wacky.py`](wacky.py): multi-start, left-hand, custom and undercut
teeth). Ours follows the runner on every form and design; CoACD jams on every
M10 form. Each tile draws the collision geometry the simulation uses: hulls,
the mesh an SDF was built from, or MuJoCo's surface mesh of the analytic SDF.

## Platform notes

- CoACD's `preprocess_mode='auto'` aborts on the threaded bolt and `'on'`
  re-voxelizes it far coarser than the thread; the input meshes are
  watertight, so preprocessing is off.
- `manifold3d` booleans are not reproducible run to run, so each exact mesh is
  built once per clearance and cached, and every method starts from the same
  file.
- Fine wedges with MULTICCD produce 5,000–22,000 contacts, so the scenes raise
  MuJoCo's arena to 512 MB.
- Most of the analytic plugin's model memory (about 48 MB) is the surface
  mesh MuJoCo extracts for every SDF; collision uses the exact distance.
