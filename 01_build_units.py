"""Map the company's raw export records onto the 18 grade-level analysis units and
verify the transcribed partner forecasts against realised annual totals.

This script documents the data-preparation step used in the paper. It cannot be run
from this repository: both inputs are commercially confidential and are not
redistributed here.

  export_records.csv       raw transaction file (Type, Destination, Grade,
                           Specification, Date, Qty_Ekspor)
  partner_annual.json      the trading partner's ex-ante annual forecast per unit and
                           year, transcribed from the company's planning documents,
                           together with the annual totals stated in those documents:
                           {"<unit>": {"<year>": {"forecast": x, "stated_actual": y}}}

Outputs `unit_demand_monthly.csv` and `mitra_annual.json`, which the remaining scripts
consume. Use `00_make_synthetic_data.py` to produce structurally equivalent files for
a runnable demonstration.
"""
import pandas as pd, numpy as np, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'export_records.csv')
PARTNER = os.path.join(HERE, 'partner_annual.json')

# --- map raw series to the grade groups at which the partner states its forecasts ---
BROKEN = {'A+': 'BR_3.5', 'A': 'BR_3.1-3.6', 'B': 'BR_2.6-3.1', 'C': 'BR_2.1-2.6',
          'D': 'BR_1.7-2.2', 'E': 'BR_1.0-1.5'}          # D and E merge USA + France
STICK = {'AA 6 Inch': 'ST_Reg6.00in', 'AA 4 Inch': 'ST_Reg4.00in',
         'AA 3.5 Inch': 'ST_Reg3.50in', 'AA 6 cm': 'ST_Reg6.00cm',
         'AA 3-4 cm': 'ST_Reg3-4cm', 'A Mojave': 'ST_A3.50in',
         'A Estonia': 'ST_A4-6cm', 'VA A 10 cm': 'ST_VA10cm',
         'VA A 7 cm (box)': 'ST_VA7cmbox', 'VA A 8 cm': 'ST_VA8cm',
         'VA A 7 cm': 'ST_VA7cm', 'VA B': 'ST_VA7cmrej'}


def unit_of(row):
    return BROKEN[row['Grade']] if row['Type'] == 'Broken' else STICK[row['Grade']]


def main():
    for path in (RAW, PARTNER):
        if not os.path.exists(path):
            raise SystemExit(
                f'missing confidential input: {os.path.basename(path)}\n'
                'Run 00_make_synthetic_data.py instead for a runnable demonstration.')

    df = pd.read_csv(RAW, parse_dates=['Date'])
    df['Unit'] = df.apply(unit_of, axis=1)
    panel = (df.groupby(['Unit', 'Date'], as_index=False)['Qty_Ekspor'].sum()
               .pivot(index='Date', columns='Unit', values='Qty_Ekspor').fillna(0.0))
    panel.to_csv(os.path.join(HERE, 'unit_demand_monthly.csv'))

    partner = json.load(open(PARTNER))

    # verification: the annual totals stated in the planning documents must agree with
    # the transaction records; disagreements are resolved in favour of the transactions
    checks = []
    for unit, years in partner.items():
        for year, rec in years.items():
            realised = panel.loc[panel.index.year == int(year), unit].sum()
            checks.append({'Unit': unit, 'Year': int(year),
                           'match': bool(np.isclose(realised, rec['stated_actual'], atol=1))})
    checks = pd.DataFrame(checks)
    print(f'annual totals verified: {checks.match.sum()} of {len(checks)} matched')
    if (~checks.match).any():
        print(checks.loc[~checks.match, ['Unit', 'Year']].to_string(index=False))

    mitra = {u: {y: rec['forecast'] for y, rec in years.items()}
             for u, years in partner.items()}
    with open(os.path.join(HERE, 'mitra_annual.json'), 'w') as f:
        json.dump(mitra, f)
    print('panel:', panel.shape, panel.index.min().date(), panel.index.max().date())


if __name__ == '__main__':
    main()
