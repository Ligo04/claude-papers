# Collision-representation benchmark

MuJoCo collides convex shapes natively, so a concave part such as a plate with
a hole has to be represented some other way. This benchmark asks how much that
choice costs, in accuracy and in speed, when the clearance is small. It
compares mjfit's exact convex pieces ("ours") with the three automatic
alternatives: convex decomposition by CoACD and by V-HACD, and MuJoCo's mesh
SDF. Every method is swept over its granularity setting, and every probe has a
correct answer computed from the analytic geometry, not from a simulator.

The peg-in-hole benchmark is described here. The same comparison on an M10
bolt and nut, with an analytic SDF as a fifth method, is in
[`thread/README.md`](thread/README.md).

## Task and representations

A 16 mm round or square hole goes through a 48 × 48 × 15 mm plate; the peg is
40 mm long. Clearance c is radial (hole radius minus peg radius) and runs from
+1 mm down to −500 µm; at c = 20 µm, (D − d)/D = 0.0025. The peg is always an
exact MuJoCo primitive (cylinder or box), so every difference comes from the
plate. The plate depends only on the hole, so one decomposition serves a whole
clearance sweep.

| method | plate collision geometry | settings swept |
|---|---|---|
| `ours` | mjfit's convex prisms (`mjfit.peg`); a circle becomes a circumscribed polygon | chord error 1000 … 0.5 µm (7 … 281 sides); a square is exact |
| `coacd` | CoACD hulls of an exact watertight plate mesh | threshold 0.2, 0.1, 0.05 (default), 0.02, 0.01; circle also 0.0075, 0.005 |
| `vhacd` | V-HACD hulls (`vhacdx`) of the same mesh | 16, 64 (default), 256 hulls at 4M voxels; circle also 1024 and 4096 hulls at up to 120M voxels |
| `sdf` | MuJoCo mesh SDF (octree) of the same mesh | octree depth 5, 6 (default), 7, 8, and 8 with 100 initial points × 30 iterations; circle also 9 and 10 |

The baselines get the circle tessellated at 1,024 segments (0.04 µm sagitta),
and ground truth uses the analytic circle, so ours is charged for its own
chord error. The settings are in [`configs/probes.yaml`](configs/probes.yaml);
the circle-only extensions, which push each baseline until it stops improving,
are in [`configs/circle_cpu.yaml`](configs/circle_cpu.yaml).

Held fixed: 1 ms time step, `implicitfast`, Newton solver, mjfit's contact
parameters (solref 2 ms / 1, solimp 0.99 0.999 0.001, µ = 1, condim 4, or
condim 1 in the frictionless probes), and zero margin. Multi-point collision
(MULTICCD, on by default in MuJoCo 3.14) is run on and off for every mesh
method; SDF collision ignores it.

## What is measured

All probes are passive: the peg moves only under gravity, a constant applied
wrench, or kinematic placement, on plain joints.

| probe | setup | exact answer |
|---|---|---|
| wall offset | rays from the hole axis, 720 angles × 11 heights; no solver | the wall lies on the analytic outline |
| S0 contact onset | aligned peg placed at known gaps (+1 mm … −200 µm, 24 directions); collision only | contact appears at gap 0 |
| S1 drop | centred, aligned free peg dropped from 1 mm; c from +1 mm to −500 µm; friction on and off | passes with no wall contact if c > 0, stays on top if c < 0 |
| S2 lateral play | x/y slides, frictionless, 0.1 N push in 24 directions | support of the feasible translation set |
| S3 tilt play | slides and one hinge, 1.5 mN·m torque | largest tilt at which both plate faces fit |
| S4 yaw play | slides and a vertical hinge, 0.8 mN·m (square only) | asin(a / (s√2)) − π/4 |

Errors are in µm of wall position (S3 at the plate faces, S4 at the peg
corner). Negative means phantom material stopped the peg early; positive means
the peg went where there is material. S2–S4 report the settled value over the
second half of a 1 s run. Whenever the simulator reports contact, the true
signed distance is also recorded from sampled exact surfaces: a positive
value is contact across a gap, a negative one is real penetration.

Cost is measured three ways, in three passive scenes with gravity and friction
on: `wall` (peg pushed sideways inside the hole, c = +100 µm), `rim` (peg
200 µm oversize resting on the rim) and `drop` (the S1 drop).

- **CPU:** one environment on one thread, best of 5 runs of 1,000 steps.
- **GPU:** MuJoCo Warp, 256 or 512 worlds, one CUDA graph per 100 steps,
  reported per environment step. Contact and constraint buffers are sized
  per representation from a calibration run (broadphase pairs included),
  every run is checked for buffer overflow, which would silently drop
  contacts, and each configuration runs in a fresh process.
- **Memory:** the compiled model's size.

The Pareto figures plot cost against the worst settled error over S2–S4, one
front per method over its settings (granularity × MULTICCD); on the GPU axis
the error is measured on MuJoCo Warp too.

## Key results

Real-time factor at the 1 ms step in the `wall` scene, and the worst settled
error over S2–S4 on CPU, from [`report/speed_table.csv`](report/speed_table.csv)
(each column is the better of MULTICCD on and off; "fills": the centred peg
already overlaps the plate, so there is no clearance left; "–": not timed at
512 worlds, see the 256-world column there).

| circle, Ø16 mm | pieces | CPU, 1 thread | GPU, 512 worlds | error, c = 100 µm | error, c = 20 µm |
|---|---:|---:|---:|---:|---:|
| ours, 7 sides | 7 | 82× | 1120× | 11 µm | 2.6 µm |
| ours, 32 sides | 32 | 29× | 1070× | 1.1 µm | 0.72 µm |
| ours, 281 sides | 281 | 4.9× | 442× | 0.29 µm | 0.17 µm |
| CoACD 0.05 (default) | 10 | 34× | 1310× | fills | fills |
| CoACD 0.005 (finest) | 4168 | 0.49× | – | 9.7 µm | 10 µm |
| V-HACD, 64 hulls (default) | 64 | 10× | 996× | fills | fills |
| V-HACD, 4096 hulls, 120M voxels | 2391 | 0.55× | – | 27 µm | fills |
| mesh SDF, depth 6 (default) | – | 2.5× | 761× | 52 µm | fills |
| mesh SDF, depth 7 | – | 2.4× | 842× | 98 µm | 33 µm |
| mesh SDF, depth 10, 100 × 30 search | – | 0.78× | – | 52 µm | 23 µm |
| square: ours (exact) | 4 | 93× | 970× | 0.31 µm | 0.17 µm |

- **On CPU, ours is an order of magnitude faster than mesh SDF at equal or
  better accuracy.** Every ours setting stays within 2.6 µm at c = 20 µm,
  while the best SDF setting is 23 µm off, and the fastest one that keeps the
  clearance open (depth 7) runs at 2.4× real time against ours' 29–82× for
  7–32 sides.
- **Ours only ever adds play.** Its circle is a circumscribed polygon, so its
  wall is never phantom; its error is the extra play toward the corners,
  c·(1/cos(π/n) − 1), and it raises no false contacts.
- **Refining a decomposition narrows its error but does not close it.** CoACD
  and V-HACD fill a 20 µm clearance at every setting but CoACD's finest
  (4,168 hulls, 8 min to build, 10 µm off).
- **Mesh SDF's remaining error is its contact search, not its octree.** At
  depth 10 the wall is within 7 µm of exact (0.1 µm mid-wall), yet S2–S4 stay
  23–68 µm off at c = 20 µm.
- **On the GPU, ours has the best front at 256 worlds**
  ([`report/pareto_gpu_circle.png`](report/pareto_gpu_circle.png)). Batching
  squeezes the fronts into 1.3–17 µs per environment step, and MuJoCo Warp
  adds an error floor of its own: ours' best is 3.2 µm at c = 100 µm and
  2.2 µm at 20 µm on Warp, against 0.29 and 0.17 µm on CPU.

The drop test (S1, c = +20 µm, friction on, MULTICCD on,
[`report/drop_circle.png`](report/drop_circle.png)): on every ours setting
the peg falls through; on each baseline's default plate it stops on the rim
or sticks partway (CoACD 0.05: blocked at every clearance up to +500 µm;
V-HACD 64 hulls: blocked up to +100 µm; mesh SDF depth 6: stuck from +10 µm).
Frictionless drops let pegs up to 20 µm oversize slip in for every method,
exact geometry included, so S1 cannot resolve errors below about 20 µm at a
2 ms contact time constant; the play probes can.

Every setting, both shapes: [`report/summary.md`](report/summary.md); figures:
`report/pareto_*.png`, `accuracy_*.png`, `drop_*.png`, `speed.png`, `cpu_vs_warp.png`.

## Hardware and software

CPU timings, peg and thread: Intel Core i9-13900KS, one thread, otherwise idle.
GPU timings: NVIDIA RTX 5090; the 512-world runs on the same i9-13900KS host,
the 256-world runs on a second host with the same GPU and an Intel Core Ultra
9 285K. Software: Python 3.12, MuJoCo 3.14.0, MuJoCo Warp 3.14.0, Warp 1.17.0,
CoACD 1.0.14, vhacdx 0.1.0; [`requirements.txt`](requirements.txt) pins all of
them.

## Install

```bash
pip install -e ".[benchmark]"               # mjfit plus the benchmark's dependencies
pip install -r benchmark/requirements.txt   # the exact versions above, incl. MuJoCo Warp
python -m benchmark.fetch cache             # the decompositions (99 MB), saves hours of CoACD
python -m benchmark.fetch videos            # optional: the behavior videos (20 MB)
```

The GPU stage needs a CUDA GPU (MuJoCo Warp is only in `requirements.txt`).
The thread benchmark's analytic SDF compiles a small MuJoCo plugin on first
use, which needs Linux and g++ (C++17) and links the `libmujoco.so` inside the
installed `mujoco` wheel; see [`thread/plugin/__init__.py`](thread/plugin/__init__.py).
Everything else is pure Python. Run from the repository root.

## Reproducing the paper

```bash
python -m benchmark.reproduce --stage report                  # rebuild report/ from results/ (minutes)
python -m benchmark.reproduce --stage probes speed gpu thread report --results /tmp/rerun
python -m benchmark.reproduce --stage gpu --dry-run           # print a stage's commands
```

| stage | what it runs | time (estimated, 32 threads, with the cache) | memory |
|---|---|---|---|
| `probes` | `run_probes` on `probes.yaml`, then on `circle_cpu.yaml` (geometry, memory, S2, S3) | about 25 min; 25 min more without the cache | about 35 GB peak (SDF depth 10: 13 GB per process) |
| `speed` | `run_speed --parts cpu_single`, then `circle_cpu.yaml` in the `wall` scene | about 15 min; idle machine | small |
| `gpu` | `run_warp_probes` (S2, S3 at c = 100, 20 µm) on both configs; `run_speed --parts warp` at 256 worlds (circle, both configs) and 512 worlds (`probes.yaml`) | about 1 h (one process per configuration) | under 2 GB of GPU memory per batch |
| `thread` | the bolt/nut sweep, see [`thread/README.md`](thread/README.md) | about 1.5 h; 45 min more without the cache | small |
| `report` | `benchmark.report` and `benchmark.thread.report --traces` | a few minutes | small |

`--help` describes every stage, including the optional `videos` and CoACD
0.01 on the thread, and `--dry-run` prints its exact commands. Run stages write
into `--results` (default `benchmark/results`), where a rerun measurement
replaces the stored one; use a fresh directory to rerun from scratch, then
`--stage report --results DIR --out DIR/report`. `--shapes`, `--methods` and
`--clearance` narrow every command for a quick check.

Where each number in the paper comes from:

- **Speed–accuracy figure**, (a) CPU and (b) GPU at 256 worlds with the error
  measured on Warp: `report/pareto_cpu_circle.png` and
  `report/pareto_gpu_circle.png`. (c) is the S1 drop at c = 20 µm on each
  method's default plate, simulated by `benchmark.render.simulate`; its
  outcomes are the `S1 wrong (fric)` column of `report/summary_circle.csv`.
- **Speed table** (real-time factors at 512 worlds, error at c = 20 µm):
  `report/speed_table.csv`, taking the better of MULTICCD on and off per
  column, as in the paper. `python -m benchmark.report --gpu-nworld 512
  --out DIR` also redraws the summaries and figures with 512-world GPU timings.
- **Probe and setup tables**: the two tables above. **Thread claims**:
  `report/thread_summary.md`.

## Raw results

`results/` holds every measurement the report and the paper use, one gzipped
CSV per table (`pandas.read_csv` reads them directly); 1.6 MB in all.

```
results/peg/probes_cpu/   geometry  wall offset per setting (µm, full wall and mid-wall)
                          memory    compiled model size (MB), octree nodes, compile time; circle
                          static    S0: reported contact per direction and true gap
                          drop      S1: outcome (passes / blocked / stuck) per clearance and friction
                          lateral   S2: settled error, spike, jitter, contact signed distances
                          tilt      S3, as lateral;   yaw  S4 (square only)
results/peg/probes_gpu/   lateral, tilt   S2 and S3 on MuJoCo Warp, with the overflow flags
results/peg/speed_cpu     CPU µs per step, collision share, contacts; per scenario
results/peg/speed_gpu     Warp throughput per batch size (nworld), buffers, GPU memory
results/thread/...        the bolt/nut tables, see thread/README.md
```

Every row names its setting (`method`, `label` = method and parameters,
`multiccd`) and, where relevant, `shape`, `clearance_um`, direction `theta`
(rad) or `phi_deg`; units are in the column names. Probe rows also carry the
decomposition time `setup_s` and the piece count `n_hulls`; timing rows carry
the CPU (and GPU) model. The circle-only extended settings have no S0, S1 or
512-world rows. Videos (`results/render/`, gitignored) and the decomposition
cache (`cache/`, gitignored) come from the release via `benchmark.fetch`.

## Code

`geometry.py` (task, exact SDFs, oracles), `representations.py` (plate
builders and their on-disk cache), `scene.py` (the common MjSpec scene),
`probes.py` and `wall.py` (CPU probes), `warp_backend.py` (batched scenes on
MuJoCo Warp), `run_probes.py`, `run_warp_probes.py`, `run_speed.py` (sweeps),
`tables.py` (the results layout), `report.py`, `render.py` (videos),
`reproduce.py`, `fetch.py`, and `thread/` for the bolt and nut. Tests:
`tests/test_benchmark_*.py`; `test_benchmark_report.py` checks that the
report regenerates the tracked summaries byte for byte.

## Platform notes (MuJoCo 3.14, MuJoCo Warp 3.14)

- The SDF octree depth exists only on the spec (`mjsMesh.octree_maxdepth`),
  and MuJoCo caches meshes by file name, so `scene.py` passes meshes as
  arrays; a file-loaded mesh would silently reuse its first octree.
- MULTICCD is on by default in 3.14. MuJoCo Warp rejects a positive margin on
  mesh pairs while it is on, and CPU SDF collision ignores margin, so every
  probe uses zero margin.
- In MuJoCo Warp, `nconmax` and `njmax` are per world and the broadphase keeps
  its candidate pairs in the contact buffer, so a plate with many hulls near
  the peg needs a buffer slot per hull: CoACD 0.01 needs about 490 per world.
  `put_data` copies the initial contacts too, so a plate that fills the
  clearance needs room for them.
- `vhacdx` expects VTK-style faces (`[3, a, b, c, …]`, flattened); plain
  `(n, 3)` faces give flat, meaningless hulls.
- CoACD at fine thresholds takes minutes per plate and runs single-threaded
  when BLAS threads are capped; `run_probes` builds all plates in parallel
  first.
