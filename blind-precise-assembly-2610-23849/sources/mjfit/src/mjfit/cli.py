"""mjfit: exact convex collision geometry for MuJoCo assemblies.

    mjfit peg --shape star -o star.xml
    mjfit peg --config usb_c --clearance 1e-4 -o usb_c.xml
    mjfit thread --config m10 -o m10.xml
    mjfit gear --teeth 18 24 -o gears.xml
    mjfit sample -n 16 --family convex -o samples/
    mjfit configs

Every generator option is a flag (--peg-length 0.03), and flags override a
--config's values. Lengths are in meters. Add --view to open the result in
MuJoCo's viewer.
"""
from __future__ import annotations

import argparse
import ast
import inspect
import re
import sys
from pathlib import Path

import yaml

from . import (
    THREAD_FORMS,
    __version__,
    build,
    configs,
    gear,
    grid,
    peg,
    sample_pegs,
    shapes,
    thread,
)
from ._config import load

OUTLINES = ('hole', 'peg', 'handle', 'plate')
CHOICES = {'form': THREAD_FORMS, 'handedness': ('right', 'left'), 'driven': ('bolt', 'nut'),
           'bolt_head': ('hex', 'cylinder'), 'nut_shape': ('hex', 'cylinder')}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog='mjfit', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--version', action='version', version=f'mjfit {__version__}')
    sub = parser.add_subparsers(dest='command', required=True)
    for name, fn in (('peg', peg), ('thread', thread), ('gear', gear)):
        p = sub.add_parser(name, help=_summary(fn), description=_summary(fn),
                           argument_default=argparse.SUPPRESS)
        p.add_argument('--config', help='bundled config name or YAML file to start from')
        if name == 'peg':
            p.add_argument('--shape', choices=shapes.PRESETS,
                           help='a named hole shape (with --size)')
            p.add_argument('--size', type=float, default=0.012,
                           help='circumradius of --shape, m (default 0.012)')
            for key in OUTLINES:
                p.add_argument(f'--{key}', type=_outline, metavar='OUTLINE',
                               help=f'{key} outline: a CSV file of x,y rows, or a YAML shape '
                                    'such as "{type: circle, radius: 0.008}"')
            p.add_argument('--island', type=_outline, action='append', dest='islands',
                           metavar='OUTLINE', help='island outline (repeatable)')
        _add_parameters(p, fn, skip=set(OUTLINES) | {'islands', 'mesh_dir'})
        _add_output(p)
        p.set_defaults(fn=fn, kind=name)

    p = sub.add_parser('sample', help='random peg configs and their models',
                       description='Write N random peg assemblies as YAML configs and MJCF.')
    p.add_argument('-n', type=int, default=16, help='number of samples (default 16)')
    p.add_argument('--seed', type=int, default=0, help='seed of the set (default 0)')
    p.add_argument('--family', choices=('primitive', 'convex'), default='primitive')
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                   help='a sampler option, e.g. --set clearance=1e-4 or --set "islands=(1, 2)"')
    p.add_argument('-o', '--output', type=Path, required=True, help='folder to write into')
    p.add_argument('--grid', action='store_true', help='also write all samples in one grid.xml')

    sub.add_parser('configs', help='list the bundled configs')
    args = parser.parse_args(argv)
    try:
        if args.command == 'configs':
            _list_configs()
        elif args.command == 'sample':
            _sample(args)
        else:
            _generate(args)
    except (ValueError, FileNotFoundError) as e:
        parser.exit(2, f'mjfit {args.command}: error: {e}\n')


def _generate(args) -> None:
    params = {k: v for k, v in vars(args).items()
              if k not in ('command', 'fn', 'kind', 'config', 'shape', 'size', 'output',
                           'export_meshes', 'view')}
    if getattr(args, 'shape', None):
        if 'hole' in params or 'peg' in params:
            raise ValueError('--shape gives the hole; do not combine it with --hole or --peg.')
        params['hole'] = shapes.preset(args.shape, args.size)
    cfg = {'kind': args.kind}
    if getattr(args, 'config', None):
        cfg = load(args.config)
        if cfg.get('kind') != args.kind:
            raise ValueError(f"{args.config} is a {cfg.get('kind')} config, not a {args.kind} one.")
        if 'hole' in params or 'peg' in params:
            cfg.pop('hole', None)
            cfg.pop('peg', None)
    elif args.kind == 'peg' and 'hole' not in params and 'peg' not in params:
        raise ValueError('Give the hole: --shape, --hole, --peg or --config.')
    cfg.update(params)
    output = getattr(args, 'output', None)
    if getattr(args, 'export_meshes', False):
        if output is None:
            raise ValueError('--export-meshes needs --output.')
        cfg['mesh_dir'] = output.parent / f'{output.stem}_meshes'
    xml = build(cfg)
    if output is None:
        sys.stdout.write(xml)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(xml)
        print(f'wrote {output}', file=sys.stderr)
    if getattr(args, 'view', False):
        _view(xml, output)


def _sample(args) -> None:
    options = {}
    for item in args.set:
        key, sep, value = item.partition('=')
        if not sep:
            raise ValueError(f'--set expects KEY=VALUE, got {item!r}.')
        options[key.strip()] = _literal(value)
    cfgs = sample_pegs(args.n, seed=args.seed, family=args.family, **options)
    args.output.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, cfg in enumerate(cfgs):
        stem = args.output / f'peg_{i:04d}'
        stem.with_suffix('.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        stem.with_suffix('.xml').write_text(build(cfg))
        if args.grid:
            parts.append(build(cfg, standalone=False))
    if args.grid:
        spacing = max(_footprint(cfg) for cfg in cfgs) * 1.3
        spec = grid(parts, spacing)
        spec.compile()                 # MuJoCo before 3.4 writes only compiled specs
        (args.output / 'grid.xml').write_text(spec.to_xml())
    print(f'wrote {len(cfgs)} samples to {args.output}', file=sys.stderr)


def _footprint(cfg: dict) -> float:
    """Width of a sampled assembly's plate."""
    if isinstance(cfg.get('plate'), list):
        return 2 * max(abs(c) for point in cfg['plate'] for c in point)
    return float(cfg.get('plate_size', 0.2))


def _list_configs() -> None:
    by_kind: dict[str, list[str]] = {}
    for name, kind in configs().items():
        by_kind.setdefault(kind, []).append(name)
    for kind in ('peg', 'thread', 'gear'):
        print(f'{kind}: {", ".join(by_kind.get(kind, []))}')


def _view(xml: str, output: Path | None) -> None:
    import mujoco
    import mujoco.viewer
    model = (mujoco.MjModel.from_xml_path(str(output)) if output is not None
             else mujoco.MjModel.from_xml_string(xml))
    mujoco.viewer.launch(model)


# ----- flags from signatures ---------------------------------------------------

def _summary(fn) -> str:
    return inspect.getdoc(fn).split('\n\n')[0].replace('\n', ' ')


def _arg_docs(fn) -> dict[str, str]:
    doc = inspect.getdoc(fn) or ''
    section = doc.split('Args:', 1)[1].split('Returns:', 1)[0] if 'Args:' in doc else ''
    out: dict[str, str] = {}
    name = None
    for line in section.splitlines():
        m = re.match(r'^    (\w+): (.*)', line)
        if m:
            name = m.group(1)
            out[name] = m.group(2)
        elif name and line.strip():
            out[name] += ' ' + line.strip()
    return {k: v.replace('``', '').replace('%', '%%') for k, v in out.items()}


def _add_parameters(parser, fn, skip) -> None:
    docs = _arg_docs(fn)
    for name, param in inspect.signature(fn).parameters.items():
        if name in skip:
            continue
        default = param.default
        flag = '--' + name.replace('_', '-')
        help_ = docs.get(name, '')
        shown = f' (default: {default})' if default is not None else ''
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction,
                                help=help_ + shown)
        elif isinstance(default, int):
            parser.add_argument(flag, type=int, metavar='N', help=help_ + shown)
        elif isinstance(default, float) or (default is None and 'float' in str(param.annotation)):
            parser.add_argument(flag, type=float, metavar='X', help=help_ + shown)
        elif isinstance(default, str):
            parser.add_argument(flag, choices=CHOICES.get(name), help=help_ + shown)
        elif isinstance(default, tuple):
            parser.add_argument(flag, type=_number, nargs='+', metavar='X', help=help_ + shown)
        else:
            parser.add_argument(flag, type=_literal, metavar='VALUE', help=help_ + shown)


def _add_output(parser) -> None:
    parser.add_argument('-o', '--output', type=Path, help='MJCF file to write (default: stdout)')
    parser.add_argument('--export-meshes', action='store_true',
                        help='write meshes as OBJ files into <output>_meshes/ instead of inline')
    parser.add_argument('--view', action='store_true', help="open the result in MuJoCo's viewer")


def _number(text: str):
    value = _literal(text)
    if not isinstance(value, (int, float)):
        raise argparse.ArgumentTypeError(f'expected a number, got {text!r}')
    return value


def _literal(text: str):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        value = yaml.safe_load(text)
        return value


def _outline(text: str):
    path = Path(text)
    if path.suffix.lower() == '.csv':
        if not path.exists():
            raise argparse.ArgumentTypeError(f'no such file: {text}')
        return {'type': 'csv', 'path': str(path.resolve())}
    value = yaml.safe_load(text)
    if not isinstance(value, (dict, list)):
        raise argparse.ArgumentTypeError(f'not a CSV file or a YAML shape: {text!r}')
    return value


if __name__ == '__main__':
    main()
