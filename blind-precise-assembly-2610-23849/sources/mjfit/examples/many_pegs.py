"""Sample random peg assemblies and lay them out in one scene.

    python examples/many_pegs.py -n 25 --family convex -o pegs.xml
    python -m mujoco.viewer --mjcf=pegs.xml
"""

import argparse
from pathlib import Path

import mjfit

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument('-n', type=int, default=25)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--family', choices=('primitive', 'convex'), default='primitive')
parser.add_argument('-o', '--output', type=Path, default=Path('pegs.xml'))
args = parser.parse_args()

configs = mjfit.sample_pegs(args.n, seed=args.seed, family=args.family)
parts = [mjfit.build(cfg, standalone=False) for cfg in configs]
spec = mjfit.grid(parts, spacing=0.3 if args.family == 'primitive' else 0.2)
model = spec.compile()
args.output.write_text(spec.to_xml())
print(f'wrote {args.output}: {args.n} assemblies, {model.ngeom} convex pieces')
