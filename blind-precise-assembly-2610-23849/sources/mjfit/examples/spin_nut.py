"""Spin a bolt into its nut, as a nut runner would, and measure how far it advances.

A perfect thread advances one lead per turn; friction and the slack between
the wedges make it a little less.

    python examples/spin_nut.py              # the bundled M10
    python examples/spin_nut.py --config m6 --rate 20
"""

import argparse
import math

import mujoco

import mjfit

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument('--config', default='m10', help='a bundled thread config')
parser.add_argument('--rate', type=float, default=15.0, help='spin rate, rad/s')
parser.add_argument('--seconds', type=float, default=1.0)
args = parser.parse_args()

cfg = mjfit.load(args.config)
model = mujoco.MjModel.from_xml_string(mjfit.build(cfg))
data = mujoco.MjData(model)
joint = model.joint('bolt_joint')
dof, z = joint.dofadr[0], joint.qposadr[0] + 2
start = data.qpos[z]

while data.time < args.seconds:
    data.qvel[dof : dof + 6] = [0, 0, data.qvel[dof + 2], 0, 0, -args.rate]  # turn clockwise
    mujoco.mj_step(model, data)

ideal = args.rate * args.seconds / (2 * math.pi) * cfg['pitch'] * cfg.get('n_starts', 1)
advance = start - data.qpos[z]
print(
    f'{args.config}: advanced {advance * 1e3:.2f} mm of an ideal {ideal * 1e3:.2f} mm '
    f'({advance / ideal:.0%})'
)
