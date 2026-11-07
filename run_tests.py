"""run_tests.py -- run every test module in order and report pass / fail.

    python -m synth        (once; writes the synthetic inputs the tests read)
    python run_tests.py

There is no test framework dependency: each module is the project's own style, a set of asserts behind a
__main__ entry point. The first run compiles the numba kernels, so it is slower than later runs.
"""
from __future__ import annotations

import importlib
import sys
import time
import traceback
from pathlib import Path

MODULES = [
    ('test_ssm', 'the generic linear-Gaussian filter and smoother, against statsmodels'),
    ('test_bonds', 'the zero-curve bootstrap and the Z-spread engine'),
    ('test_afns', 'the shadow-rate rate model: quadrature, Monte Carlo, Jacobian, likelihood'),
    ('test_lattice_synth', 'the call price, the lattice, the OAS inversion and the recovery kernel'),
    ('test_filters_synth', 'the four-state panel filters on a simulated panel'),
]


def main():
    if not (Path(__file__).parent / 'data' / 'fred').exists():
        print('data/ is missing: run  python -m synth  first', file=sys.stderr)
        return 2
    results = []
    for name, what in MODULES:
        print(f'\n{"=" * 100}\n{name}  --  {what}\n{"=" * 100}', flush=True)
        t = time.time()
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, 'main'):
                mod.main()
            else:                                        # module-level asserts, run on import
                tests = [v for k, v in sorted(vars(mod).items()) if k.startswith('test_') and callable(v)]
                for fn in tests:
                    fn()
                    print(f'ok  {fn.__name__}')
            results.append((name, 'PASS', time.time() - t))
        except Exception:
            traceback.print_exc()
            results.append((name, 'FAIL', time.time() - t))
    print(f'\n{"=" * 100}\nsummary\n{"=" * 100}')
    for name, status, secs in results:
        print(f'  {status}  {name:<22} {secs:7.1f} s')
    n_fail = sum(1 for _, s, _ in results if s == 'FAIL')
    print(f'\n{len(results) - n_fail}/{len(results)} modules passed')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
