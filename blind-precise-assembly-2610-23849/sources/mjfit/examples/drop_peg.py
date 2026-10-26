"""Drop a peg into its hole and report whether it touched the plate on the way in.

    python examples/drop_peg.py usb_c
    python examples/drop_peg.py rj45 --clearance 2e-5
"""

import argparse

import mujoco

import mjfit

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument('config', nargs='?', default='usb_c', help='a bundled peg config')
parser.add_argument('--clearance', type=float, help='override the clearance, m')
args = parser.parse_args()

overrides = {} if args.clearance is None else {'clearance': args.clearance}
model = mujoco.MjModel.from_xml_string(mjfit.build(args.config, **overrides))
data = mujoco.MjData(model)
plate, peg = model.body('plate').id, model.body('peg').id
floor = model.geom('floor').id
landing = {floor} | {i for i in range(model.ngeom) if model.geom(i).name == 'bottom'}

touches = 0
while data.time < 1.0:
    mujoco.mj_step(model, data)
    geoms = {g for contact in data.contact[: data.ncon] for g in contact.geom}
    if geoms & landing:
        break  # reached the floor, or the bottom of a blind hole
    touches += sum(
        {model.geom_bodyid[g] for g in contact.geom} == {plate, peg}
        for contact in data.contact[: data.ncon]
    )

fell = model.body('peg').pos[2] - data.xpos[peg][2]
print(
    f'{args.config}: fell {fell * 1e3:.1f} mm in {data.time * 1e3:.0f} ms and '
    + ('never touched the plate.' if touches == 0 else f'touched the plate in {touches} contacts.')
)
