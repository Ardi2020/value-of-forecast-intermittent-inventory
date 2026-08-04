# From forecast accuracy to inventory performance

Reproducible code for the study *"From forecast accuracy to inventory performance: a
simulation-based value assessment of machine-learning demand forecasts in a
periodic-review policy for intermittent export demand"* (submitted to *Transactions on
Computational Modelling and Intelligent Systems*).

The study asks whether better demand forecasts actually produce cheaper inventory
decisions. Ten forecasting approaches — an exporter's incumbent ex-ante annual plan,
naive and exponential smoothing, Croston, SBA, TSB, ARIMA, and three machine-learning
configurations — are each fed through a sequential simulation of the company's
periodic-review `(R,S)` replenishment policy, with a three-month lead time, shipping-lot
rounding, and a safety stock calibrated to that method's own realised forecast errors.
Methods are then compared on inventory cost at equivalent service, not on accuracy alone.

## Data availability

The company's export records are commercially confidential and are **not** redistributed
here. To keep the pipeline fully executable, `00_make_synthetic_data.py` generates a
synthetic panel with the same structure as the case data: 18 grade-level demand units over
55 monthly periods (June 2020 – December 2024), spanning the same intermittency range,
with lot-sized quantities, ex-ante annual "partner" forecasts, and three exogenous
drivers. The synthetic numbers are drawn from parametric distributions and do not
reproduce the company's actual demand, so results obtained from them will resemble the
paper qualitatively but will not match it numerically.

The result tables reported in the paper, computed from the real data, are included as
`results_*.csv`. They contain normalised cost indices, accuracy metrics, and rank
correlations only — no absolute demand quantities.

Researchers who need the underlying data may contact the corresponding author; access
depends on the case company's consent.

## Reproducing the analysis

```bash
pip install -r requirements.txt

python 00_make_synthetic_data.py    # synthetic demand panel, partner forecasts, exogenous data
python 02_forecasts.py              # rolling-origin forecasts, 10 methods, horizons 1-5
python 03_ml_models.py              # regenerate the ML forecasts (RF, XGBoost-Tweedie, hybrid)
python 04_simulate.py               # sequential (R,S) simulation over the cost and service grid
python 05_analyze.py                # accuracy metrics, intermittency profile, alignment
python 06_analyze_segments.py       # SBC segment analysis and sensitivity tables
python 07_figures.py                # publication figures
```

`01_build_units.py` maps the company's raw transaction file to the 18 analysis units and
verifies the transcribed partner forecasts against realised totals. It requires the
confidential source file and is included for transparency rather than execution.

Running times on a laptop: the forecasting step is the slowest (ARIMA order selection at
every origin), typically 5–15 minutes; the remaining steps take under a minute each.

## What each script does

| Script | Purpose |
|---|---|
| `00_make_synthetic_data.py` | Generates the synthetic demand panel, partner annual forecasts, and exogenous series |
| `01_build_units.py` | Aggregates raw company records into 18 demand units; cross-checks partner forecasts |
| `02_forecasts.py` | Rolling-origin forecasts for all ten methods, horizons 1–5, origins 2022–2024 |
| `03_ml_models.py` | Retrains the three ML configurations (Tweedie objective, probability-weighted hybrid) |
| `04_simulate.py` | `(R,S)` simulation: order-up-to levels, lot rounding, backorders, cost and service grid |
| `05_analyze.py` | MASE, RMSSE, adjusted MAPE, SBC intermittency profile, accuracy-to-cost alignment |
| `06_analyze_segments.py` | Cost index by SBC segment, sensitivity to cost ratio and service level |
| `07_figures.py` | Figures 1–5 of the paper at 300 dpi |

## Result files

| File | Content |
|---|---|
| `results_table1_accuracy.csv` | Median accuracy per method across the 18 units (Table 1) |
| `results_table2_cost_sweep.csv` | Cost index by backorder-to-holding ratio (Table 2) |
| `results_segment_index.csv` | Cost index by SBC demand segment (Figure 4) |
| `results_alignment.csv` | Rank correlation between accuracy and simulated cost (Figure 5) |

## Citation

If you use this code, please cite the paper. See `CITATION.cff` for machine-readable
metadata; the reference will be updated once the article is published.

## Licence

Code is released under the MIT Licence (`LICENSE`). The licence covers the code and the
synthetic data generator only; it does not extend to the company's operational data,
which are not part of this repository.
