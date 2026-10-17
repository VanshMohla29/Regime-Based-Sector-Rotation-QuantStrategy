"""Plain (non-fixture) helpers shared across test modules."""

import numpy as np
import pandas as pd

SECTORS = ['BANKBEES', 'ITBEES', 'PHARMABEES', 'AUTOBEES',
           'METALIETF', 'MOREALTY', 'CPSEETF', 'INFRABEES']
REGIMES = ['Bull', 'Bear', 'HighVol', 'Sideways']


def make_sector_frame(index, daily_returns=None, seed=7):
    """
    Sector log-return frame covering all 8 ETFs.

    Keep >= 100 rows: compute_sector_rotation_returns() falls back to a *live*
    Yahoo Finance download when fewer rows are supplied.
    """
    n = len(index)
    if daily_returns is None:
        rng = np.random.default_rng(seed)
        data = rng.normal(0.0003, 0.008, (n, len(SECTORS)))
    else:
        data = np.tile(np.asarray(daily_returns, dtype=float).reshape(-1, 1), (1, len(SECTORS)))
    return pd.DataFrame(data, index=index, columns=SECTORS)


def runs(seq):
    """Run-length encode a sequence -> list of (value, length)."""
    out = []
    for v in seq:
        if out and out[-1][0] == v:
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return [(v, n) for v, n in out]
