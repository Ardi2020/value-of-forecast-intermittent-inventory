"""Automated checks that no forecast uses information dated at or after its origin.

Run directly (`python test_leakage.py`) or under pytest. 03_ml_models.py also asserts the
vintage property internally so that a leaking configuration cannot silently produce
forecasts; this file re-checks the property from the outside, on the forecast frame that
the rest of the pipeline consumes.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VINTAGE_CUTOFF = {2021: pd.Timestamp('2021-12-01'),
                  2022: pd.Timestamp('2022-12-01'),
                  2023: pd.Timestamp('2023-12-01')}


def vintage_for(origin):
    """The rule used by 03_ml_models.py: the last completed annual vintage at the origin."""
    return min(max(origin.year - 1, 2021), 2023)


def test_vintage_precedes_origin():
    fc = pd.read_parquet(os.path.join(HERE, 'forecasts.parquet'))
    origins = sorted(pd.to_datetime(fc.Origin).unique())
    bad = [(o, vintage_for(pd.Timestamp(o))) for o in origins
           if VINTAGE_CUTOFF[vintage_for(pd.Timestamp(o))] >= pd.Timestamp(o)]
    assert not bad, f'vintage trained on data dated at or after the origin: {bad[:5]}'


def test_horizons_share_one_information_set():
    """All five horizons issued at one origin must come from the same vintage."""
    fc = pd.read_parquet(os.path.join(HERE, 'forecasts.parquet'))
    ml = fc[fc.Model.isin(['rf', 'xgb', 'hybrid'])]
    for o, g in ml.groupby('Origin'):
        assert len({vintage_for(pd.Timestamp(o))}) == 1
        assert set(g.h.unique()) <= {0, 1, 2, 3, 4}


def test_forecast_targets_start_at_the_origin():
    fc = pd.read_parquet(os.path.join(HERE, 'forecasts.parquet'))
    off = fc[pd.to_datetime(fc.Month) < pd.to_datetime(fc.Origin)]
    assert off.empty, f'{len(off)} rows forecast a month earlier than their origin'


if __name__ == '__main__':
    test_vintage_precedes_origin()
    test_horizons_share_one_information_set()
    test_forecast_targets_start_at_the_origin()
    print('all leakage checks passed')
