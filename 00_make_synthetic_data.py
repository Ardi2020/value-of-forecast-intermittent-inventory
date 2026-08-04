"""Generate a synthetic demand panel with the same structure as the confidential
case data, so that the full pipeline (02 -> 07) can be executed by anyone.

The real export records of the case company cannot be redistributed. This script
reproduces the *structure* used in the paper -- 18 grade-level demand units over
55 monthly periods (Jun 2020 - Dec 2024), spanning the same intermittency range,
with lot-sized broken-cassia quantities, ex-ante annual "partner" forecasts, and
three exogenous drivers. Numbers are drawn from parametric distributions and do
NOT reproduce the company's actual demand.

Outputs (consumed by the later scripts):
  unit_demand_monthly.csv   monthly demand per unit
  mitra_annual.json         partner annual forecast per unit and year
  exog_monthly.csv          exchange rate and three search-interest indices
"""
import numpy as np, pandas as pd, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20260803)

MONTHS = pd.date_range('2020-06-01', '2024-12-01', freq='MS')

# (name, P(demand in a month), mean size kg, size CV, lot size kg)
SPEC = [
    ('BR_2.1-2.6', 0.84, 130_000, 0.90, 25_025),   # large, erratic
    ('BR_1.7-2.2', 0.81,  90_000, 0.68, 25_025),   # large, smooth
    ('BR_1.0-1.5', 0.51,  60_000, 0.64, 25_025),
    ('BR_2.6-3.1', 0.33,  42_000, 0.84, 25_025),
    ('BR_3.1-3.6', 0.23,  40_000, 0.44, 25_025),
    ('BR_3.5',     0.26,  27_000, 0.28, 25_025),
    ('ST_VA7cm',   0.30,  13_000, 0.44, 20),
    ('ST_Reg4.00in', 0.30, 14_000, 0.60, 20),
    ('ST_Reg3.50in', 0.26, 14_000, 0.41, 20),
    ('ST_Reg6.00in', 0.21,  1_500, 0.42, 20),
    ('ST_A3.50in',   0.14,  4_500, 0.74, 20),
    ('ST_A4-6cm',    0.07, 10_000, 0.05, 20),
    ('ST_Reg3-4cm',  0.07,  8_300, 0.35, 20),
    ('ST_VA10cm',    0.02,  1_500, 0.10, 20),
    ('ST_VA7cmbox',  0.02, 16_000, 0.10, 20),
    ('ST_VA7cmrej',  0.02, 10_000, 0.10, 20),
    ('ST_Reg6.00cm', 0.00,      0, 0.00, 20),
    ('ST_VA8cm',     0.00,      0, 0.00, 20),
]

panel = {}
for name, p, mu, cv, lot in SPEC:
    occ = rng.random(len(MONTHS)) < p
    if mu == 0:
        panel[name] = np.zeros(len(MONTHS))
        continue
    sigma = np.sqrt(np.log(1 + cv ** 2))
    size = rng.lognormal(np.log(mu) - sigma ** 2 / 2, sigma, len(MONTHS))
    q = np.where(occ, size, 0.0)
    q = np.round(q / lot) * lot            # respect shipping-lot granularity
    panel[name] = q

df = pd.DataFrame(panel, index=MONTHS)
df.index.name = ''
df.to_csv(os.path.join(HERE, 'unit_demand_monthly.csv'))

# ex-ante annual "partner" forecast: realised annual total distorted by a large,
# unbiased error, matching the ~45% annual MAPE reported for the case company
mitra = {}
for name in df.columns:
    mitra[name] = {}
    for year in (2021, 2022, 2023, 2024):
        actual = df.loc[df.index.year == year, name].sum()
        factor = np.clip(rng.normal(1.0, 0.45), 0.05, 2.5)
        mitra[name][str(year)] = float(np.round(actual * factor, -3))
with open(os.path.join(HERE, 'mitra_annual.json'), 'w') as f:
    json.dump(mitra, f, indent=1)

# exogenous drivers: a random-walk exchange rate and three bounded interest indices
fx = 14_000 + np.cumsum(rng.normal(35, 90, len(MONTHS)))
exog = pd.DataFrame({
    'Kurs_USD': fx,
    'GT_Interest_FR': np.clip(30 + np.cumsum(rng.normal(0, 3, len(MONTHS))), 5, 100),
    'GT_Interest_US': np.clip(28 + np.cumsum(rng.normal(0, 3, len(MONTHS))), 5, 100),
    'GT_Price_US': np.clip(50 + np.cumsum(rng.normal(0, 4, len(MONTHS))), 5, 100),
}, index=MONTHS)
exog.index.name = 'Date'
exog.to_csv(os.path.join(HERE, 'exog_monthly.csv'))

nz = (df > 0).mean()
print('synthetic panel:', df.shape)
print('share of months with demand: min %.2f  median %.2f  max %.2f'
      % (nz.min(), nz.median(), nz.max()))
print('written: unit_demand_monthly.csv, mitra_annual.json, exog_monthly.csv')
