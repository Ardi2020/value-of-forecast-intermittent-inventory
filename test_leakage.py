"""Row-level information-set audit of the released forecast frame.

Round 3 (R3-M-19) noted that a hard-coded rule check is not the same as tracing what each
forecast was actually built from. `03_ml_models.py` therefore stamps every machine-learning
forecast row with the vintage it came from, the last date in that vintage's training data,
and the most recent feature date used for that target month. This file audits those stamps.

Run directly (`python test_leakage.py`) or under pytest.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ML = ['rf', 'xgb', 'hybrid']


def _frame():
    fc = pd.read_parquet(os.path.join(HERE, 'forecasts.parquet'))
    for c in ('Origin', 'Month', 'training_end', 'feature_max_date'):
        if c in fc.columns:
            fc[c] = pd.to_datetime(fc[c])
    return fc


def test_provenance_columns_present():
    fc = _frame()
    for c in ('vintage_year', 'training_end', 'feature_max_date'):
        assert c in fc.columns, f'forecast frame has no {c} column'
    ml = fc[fc.Model.isin(ML)]
    assert ml.training_end.notna().all(), 'machine-learning rows without a training_end'
    assert ml.feature_max_date.notna().all(), 'machine-learning rows without a feature_max_date'


def test_training_data_precedes_every_origin():
    ml = _frame()
    ml = ml[ml.Model.isin(ML)]
    bad = ml[ml.training_end >= ml.Origin]
    assert bad.empty, (f'{len(bad)} rows trained on data dated at or after their origin, '
                       f'first: {bad.iloc[0].to_dict() if len(bad) else None}')


def test_features_precede_every_origin():
    ml = _frame()
    ml = ml[ml.Model.isin(ML)]
    bad = ml[ml.feature_max_date >= ml.Origin]
    assert bad.empty, f'{len(bad)} rows use a feature dated at or after their origin'


def test_one_information_set_per_origin():
    """All horizons issued at one origin must come from a single vintage."""
    ml = _frame()
    ml = ml[ml.Model.isin(ML)]
    mixed = ml.groupby('Origin').vintage_year.nunique()
    assert (mixed == 1).all(), f'origins drawing on more than one vintage: {mixed[mixed > 1].index.tolist()}'


def test_horizons_are_the_protection_interval():
    ml = _frame()
    ml = ml[ml.Model.isin(ML)]
    assert set(ml.h.unique()) <= {0, 1, 2, 3, 4}, 'unexpected horizon outside 0..4'
    off = ml[ml.Month < ml.Origin]
    assert off.empty, f'{len(off)} rows forecast a month earlier than their origin'


def test_vintage_matches_the_origin_rule():
    """The stamped vintage must be the last completed annual vintage at the origin."""
    ml = _frame()
    ml = ml[ml.Model.isin(ML)]
    expected = ml.Origin.dt.year.sub(1).clip(2021, 2023)
    mismatch = ml[ml.vintage_year.astype(int) != expected]
    assert mismatch.empty, f'{len(mismatch)} rows carry a vintage other than the origin rule'


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok:', name)
    print('\nall row-level leakage checks passed')
