"""Wacky screws: how far mjfit's thread forms, starts, handedness and custom
teeth stretch. Each design is a bolt/nut pair with its own nut-runner speed
(rev/s), chosen so the nut stays on the rod over the video's 4.3 s.

DESIGNS are convex teeth; UNDERCUT are teeth that hook under each other (a
dovetail, and non-convex L-hooks, T-slots and barbed arrowheads), which
mjfit cuts into convex pieces.

    MUJOCO_GL=egl python -m benchmark.thread.render --behaviors wacky --workers 1
    MUJOCO_GL=egl python -m benchmark.thread.render --behaviors undercut --workers 1
"""
from __future__ import annotations

from dataclasses import dataclass

from .geometry import ThreadTask

MM = 1e-3


@dataclass(frozen=True)
class Design:
    title: str
    blurb: str
    task: ThreadTask
    clearance_um: float = 100.0
    speed: float = 1.0          # nut runner, rev/s


def _task(name, **kw) -> ThreadTask:
    return ThreadTask(shape=f'wacky_{name}', **kw)


DESIGNS = {
    'sawtooth': Design(
        'Sawtooth 0°/60°', 'a flat ledge carries the load; the back is a 60° ramp',
        _task('sawtooth', form='buttress',
              form_params=(('crest_flat', 0.05), ('depth', 0.5), ('flank_angles', (60.0, 0.0))))),
    'knife': Design(
        'Knife-edge 90° V', 'sharp crests, sharp roots, no flats',
        _task('knife', form='trapezoidal',
              form_params=(('crest_flat', 0.0), ('depth', 0.5), ('flank_angles', (45.0, 45.0))))),
    'six_start': Design(
        'Six starts, 18 mm lead', 'lead angle 22° beats friction: back-drivable',
        _task('six_start', form='trapezoidal', major_diameter=16 * MM, pitch=3 * MM, starts=6,
              bolt_length=46 * MM, nut_height=10 * MM, nut_outer_radius=12 * MM),
        speed=0.3),
    'lefty_square': Design(
        'Left-hand double-start square', 'turns the other way; two threads interleaved',
        _task('lefty_square', form='square', major_diameter=12 * MM, pitch=2 * MM, starts=2,
              handedness='left', bolt_length=36 * MM, nut_outer_radius=9 * MM)),
    'auger': Design(
        'Auger', 'deep 5° blades on a thin 6.4 mm core',
        _task('auger', form='trapezoidal', major_diameter=16 * MM, pitch=4 * MM,
              form_params=(('crest_flat', 0.3), ('depth', 1.2), ('flank_angles', (5.0, 5.0))),
              bolt_length=36 * MM, nut_outer_radius=11 * MM)),
    'coil': Design(
        'Coil spring', 'a thin wire wound loose on both parts (1.6 mm wire, 6 mm pitch)',
        _task('coil', form='wire', major_diameter=14 * MM, pitch=6 * MM,
              form_params=(('nut_wire_offset', 1.0 * MM), ('wire_diameter', 1.6 * MM)),
              bolt_length=30 * MM, nut_outer_radius=10 * MM),
        clearance_um=0.0, speed=0.6),
    'house_dart': Design(
        'House and dart', 'custom teeth: pentagon ridges on the bolt, darts in the nut',
        _task('house_dart', form='custom', major_diameter=16 * MM, pitch=3 * MM,
              form_params=(
                  ('bolt_tooth', ((6.0 * MM, 0.6 * MM), (6.0 * MM, 2.4 * MM), (7.2 * MM, 2.4 * MM),
                                  (8.0 * MM, 1.5 * MM), (7.2 * MM, 0.6 * MM))),
                  ('core_diameter', 12 * MM),
                  ('nut_tooth', ((8.2 * MM, 2.6 * MM), (8.2 * MM, 3.4 * MM), (6.5 * MM, 3.0 * MM))),
                  ('root_diameter', 16.4 * MM)),
              bolt_length=30 * MM, nut_outer_radius=11 * MM),
        clearance_um=50.0),
    'fine': Design(
        'Fine pitch M10×0.5', 'sixteen engaged turns in an 8 mm nut',
        _task('fine', pitch=0.5 * MM), clearance_um=50.0),
    'triple_wire': Design(
        'Left-hand triple-start wire', 'three round wires, 6 mm lead, turning the other way',
        _task('triple_wire', form='wire', major_diameter=12 * MM, pitch=2 * MM, starts=3,
              handedness='left', bolt_length=34 * MM, nut_outer_radius=9 * MM),
        speed=0.6),
}


def _custom(name, bolt, nut, core, root, **kw) -> ThreadTask:
    """Custom teeth given in mm as (r, u) points, on a 16 mm bolt with a 4 mm
    pitch, 36 mm long, under a 12 mm nut, with 45° chamfers."""
    def mm(pts):
        return tuple((r * MM, u * MM) for r, u in pts)
    return _task(name, form='custom', major_diameter=16 * MM, pitch=4 * MM,
                 form_params=(('bolt_tooth', mm(bolt)), ('core_diameter', core * MM),
                              ('nut_tooth', mm(nut)), ('root_diameter', root * MM)),
                 bolt_length=36 * MM, nut_height=12 * MM, nut_outer_radius=11 * MM,
                 chamfer=3 * MM, **kw)


# Each pair fills the 3 mm band between the core (r 5 mm) and the nut root
# (r 8 mm) exactly, one pitch at a time; clearance then opens every face.
# All are chamfered 45° (thread_chamfer = thread depth): cut off flat, an
# undercut thread would leave a thin blade at each end.
UNDERCUT = {
    'dovetail': Design(
        'Dovetail', '15° undercut flanks: both teeth widen away from their base',
        _task('dovetail', form='trapezoidal', major_diameter=16 * MM, pitch=4 * MM,
              form_params=(('crest_flat', 0.6), ('depth', 0.5), ('flank_angles', (-15.0, -15.0))),
              bolt_length=36 * MM, nut_height=12 * MM, nut_outer_radius=11 * MM,
              chamfer=2 * MM)),
    'hook': Design(
        'L-hooks', "interlocking L's: the nut's toe tucks under the bolt's",
        _custom('hook',
                bolt=[(5, 0), (5, 1), (7, 1), (7, 3), (8, 3), (8, 0)],
                nut=[(5, 1), (5, 4), (8, 4), (8, 3), (7, 3), (7, 1)], core=10, root=16)),
    't_slot': Design(
        'T-slot', "T's on the bolt ride in inverted-T slots in the nut",
        _custom('t_slot',
                bolt=[(5, 1.5), (5, 2.5), (7, 2.5), (7, 3.5), (8, 3.5), (8, 0.5), (7, 0.5),
                      (7, 1.5)],
                nut=[(5, 2.5), (5, 5.5), (7, 5.5), (7, 4.5), (8, 4.5), (8, 3.5), (7, 3.5),
                     (7, 2.5)], core=10, root=16)),
    'barb': Design(
        'Barbed arrowheads', 'arrowheads on stems; the nut fills the pockets under the barbs',
        _custom('barb',
                bolt=[(5, 1.6), (5, 2.4), (6.5, 2.4), (6.5, 3.2), (8, 2.0), (6.5, 0.8),
                      (6.5, 1.6)],
                nut=[(5, 2.4), (5, 5.6), (6.5, 5.6), (6.5, 4.8), (8, 6.0), (8, 2.0), (6.5, 3.2),
                     (6.5, 2.4)], core=10, root=16)),
}
