# Threads

```python
mjfit.thread(major_diameter=0.010, pitch=0.0015, clearance=1e-4, ...)
```

`mjfit.thread` returns an MJCF model of a bolt engaged in its nut. A thread is a tooth
profile swept along a helix, and each short angular slice of one turn is a twisted
prism, which is convex. The bolt is a core cylinder plus one wedge per slice. The nut is
a convex-decomposed shell plus inward-facing wedges. `help(mjfit.thread)` and
`mjfit thread --help` list every parameter; the defaults make a steel M10 x 1.5.

## Thread forms

`form` picks the tooth shape, and `form_params` overrides its defaults.

| `form` | Profile | Defaults (P = pitch) |
| --- | --- | --- |
| `iso` | ISO 68-1 metric (and UTS): a 60° V with flats | depth 5√3/16 P, crest flat P/8 |
| `trapezoidal` | ISO 2904 (Tr): 30° | depth P/2, crest flat 0.366 P |
| `acme` | ACME: 29° | depth P/2, crest flat 0.3707 P |
| `square` | Square: 0° flanks | depth P/2, crest flat P/2 |
| `buttress` | ANSI 7°/45° | depth 0.6 P, crest flat 0.1631 P |
| `wire` | A round wire wound on each part | wire diameter P/√3, tight |
| `custom` | Your own teeth, convex or not | — |

- **The trapezoid family** (`iso` to `buttress`) takes `flank_angles` (the bolt tooth's
  lower and upper flank, in degrees from the radial plane), `depth`, and `crest_flat`
  (both in pitches). Negative flank angles undercut the teeth into a dovetail.
- **`wire`** takes `wire_diameter`, `nut_wire_offset` (how far out the nut's wire center
  sits from the bolt's), `bore_diameter`, and `wire_segments` (sides of the polygon
  inscribed in the wire's section, 16 by default).
- **`custom`** takes `bolt_tooth` and `nut_tooth`, lists of (r, u) points in meters in
  one shared phase (u is the axial position along the helix, mod P), plus
  `core_diameter` and `root_diameter`. A tooth that is not convex (a hook, a T, a barb)
  is cut into convex pieces, and each piece is swept as its own wedges.

`n_starts` interleaves several helices (the lead is `n_starts * pitch`), and
`handedness` is `right` or `left`. `mjfit.thread_profile(form, major_diameter, pitch,
params)` returns a form's zero-clearance teeth.

## Clearance

Every exposed edge of both teeth moves back `clearance / 2` along its normal, and the
nut's root moves out by the same amount. Facing surfaces are then exactly `clearance`
apart, which for `iso` is the standard's rule: an ISO thread has an axial play of
±2c/√3 and a radial play of c/2.

## Discretization only adds play

A wedge's faces are chords of the helical surfaces. On the bolt they fall inside the
true thread. On the nut they would bulge toward the bolt, so each nut wedge is pushed
out by its chord sag and its flanks are moved back by the helicoid twist. The generator
then measures one wedge's hull against the exact tooth and moves back any edge that
still sticks out. Fewer segments per turn therefore only add play and never
interference. The tests check every form's wedges against the exact helical parts.

## Seams, chamfers, and the scene

- **`seam_relief`.** MuJoCo collides every pair of convex pieces separately, so a tooth
  sliding across the seam into the next wedge can catch on its end face. Seam relief
  dips the teeth by this depth at every seam, ramping back over about 20 times the
  depth, so the seams become gentle ramps. It only removes material. About 5 µm suits
  MuJoCo's default contact softness.
- **`chamfer`.** Chamfers the bolt's tip (and both ends of a bare rod) and countersinks
  both nut faces, over this axial length: 45° when it equals the thread depth. It gives
  a lead-in, and it removes the thin blade an undercut tooth leaves at a flat end.
- **`driven`.** `bolt` (the default) puts the bolt above the nut with its tip about one
  pitch in; `nut` puts the nut on top of an upright bolt. `fixed=True` welds the lower
  part to the world.
- **Frames.** The bolt's frame is at the center of its tip with the thread along +z.
  The nut's is at the center of its bottom face. The free joints are `bolt_joint` and
  `nut_joint`.

## Friction and segments

A thread self-locks once its friction exceeds the tangent of its lead angle, about 0.05
for an M10, so a nut runner spinning a part at constant speed fights friction all the
way. The table gives the advance of the bundled M10's bolt, spun at 15 rad/s with
24 segments per turn, as a fraction of the ideal lead (MuJoCo 3.8, from 0.2 s to 1.2 s):

| Friction, condim | c = 100 µm | with 5 µm relief | c = 20 µm | with 5 µm relief |
| --- | --- | --- | --- | --- |
| 1.0, 4 | 83% | 81% | 14% | 15% |
| 0.3, 3 | 92% | 96% | 92% | 95% |
| 0.1, 3 | 97% | 100% | 97% | 98% |

For parts that should spin in freely, use friction 0.1 to 0.3 with `condim=3`.

`segments_per_turn` trades speed for play: the discretization loosens the nut by up to
r (1/cos(π/n) − 1) radially. 24 is the default and what the paper's nut threading task
trained with. More segments are not automatically better: their wedges are thinner and
their seams more numerous, and in our spin tests 48 to 96 segments per turn without seam
relief made the simulation unstable at 20 µm. Check a finer thread with your own drive
before relying on it.

## Limitations

- Teeth must be simple polygons in the meridian plane. A non-convex tooth costs two or
  three times the geoms.
- A round wire's section is an inscribed polygon, so it only rounds the wire down.
- The geom count grows with thread length over pitch times segments per turn: an M10
  with 20 mm of thread and 24 segments per turn is about 500 meshes, which MuJoCo
  compiles in a few tens of milliseconds.
- Tapered threads, variable pitch, and non-round cores are not supported.
