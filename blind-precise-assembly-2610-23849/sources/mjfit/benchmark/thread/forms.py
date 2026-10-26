"""Thread forms for the physics videos: M10-sized parts (D 10 mm, P 1.5 mm)
in several of mjfit's thread forms, and the FurnitureBench square-table leg
in its tabletop hole (mjfit's bundled config threads/furniture_bench_leg.yaml).

The FurnitureBench thread is a loose round wire (pitch 9.375 mm, about ±1.6 mm
of axial play as modelled), so it runs at zero extra clearance. Its rod is
longer than the real 19 mm stud so the nut has room for the same four turns as
the others, and the nut is the tabletop's threaded boss (PLA).
"""
from __future__ import annotations

from .geometry import ThreadTask

FB_WIRE = (('bore_diameter', 0.025), ('nut_wire_offset', 0.0031), ('wire_diameter', 0.00436))

FORMS = {
    'iso': ThreadTask(),
    'trapezoidal': ThreadTask(form='trapezoidal', shape='thread_trapezoidal'),
    'square': ThreadTask(form='square', shape='thread_square'),
    'buttress': ThreadTask(form='buttress', shape='thread_buttress'),
    'wire': ThreadTask(form='wire', shape='thread_wire'),
    'furniture_bench': ThreadTask(major_diameter=0.02436, pitch=0.009375, bolt_length=0.056,
                                  nut_height=0.017, nut_outer_radius=0.02, density=1250.0,
                                  form='wire', form_params=FB_WIRE,
                                  shape='thread_furniture_bench'),
}
CLEARANCE_UM = {name: 100.0 for name in FORMS} | {'furniture_bench': 0.0}
TITLES = {
    'iso': 'ISO 60° metric',
    'trapezoidal': 'Trapezoidal 30°',
    'square': 'Square',
    'buttress': 'Buttress 7°/45°',
    'wire': 'Round wire',
    'furniture_bench': 'FurnitureBench leg (round wire)',
}
