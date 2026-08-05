# From forecast accuracy to inventory performance

Reproducible code for the study *"From forecast accuracy to inventory performance: a
simulation-based value assessment of machine-learning demand forecasts in a
periodic-review policy for intermittent export demand"* (prepared for submission to *Transactions on
Computational Modelling and Intelligent Systems*).

The study asks whether better demand forecasts actually produce cheaper inventory
decisions. Ten forecasting approaches — an exporter's incumbent, company-described
ex-ante annual plan, naive and exponential smoothing, Croston, SBA, TSB, ARIMA, and three
machine-learning configurations — are each fed through a sequential simulation of the
company's periodic-review `(R,S)` replenishment policy, with a three-month lead time,
shipping-lot rounding, and a safety stock calibrated to that method's own realised
forecast errors. Methods are then compared on inventory cost **under a common achieved-service
requirement**, not on accuracy alone and not at a common service target.

## Running it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python smoke_test.py       # ~30 s: imports, dependencies, manifest, vintage rule, simulator
python verify_manifest.py  # recompute every released checksum against MANIFEST.md
python run_all.py          # add --quick to skip the three slow sensitivity studies
python test_leakage.py     # information-set checks, also run inside 03_ml_models.py
```

`smoke_test.py` is what CI runs on every push (`.github/workflows/smoke.yml`). It checks
the things that actually broke in earlier revisions: a script referencing a file that is
not in the repository, a policy-logic import error, and the origin-safety of the vintage
rule.

`run_all.py` is the only entry point you need. It deletes every generated file first,
runs the steps in order, stops at the first step that exits non-zero **or** fails to
produce the outputs it declares, and finishes by printing a SHA-256 prefix for each
generated file so two runs can be compared directly. Exact package versions are pinned in
`requirements.txt`; see `environment.md` for the rest of the environment.

Expected running time: roughly 50 minutes for `--quick` and about two hours for the full
pipeline on a modest container. `02_forecasts.py` dominates, because ARIMA orders are
selected by AIC at every origin and the synthetic series converge slowly; it accounts for
about 45 minutes on its own.

## What is real and what is synthetic

The company's export records are commercially confidential and are **not** redistributed.
Two different things are provided, and they should not be confused:

* **`realcase_*.csv` — real.** The aggregate result tables computed from the confidential
  data. Every table and figure reported in the paper is derived from these files. They
  contain normalised cost indices, accuracy metrics, achieved service, and rank
  correlations only; no absolute demand quantity appears in them. Their SHA-256 hashes
  and provenance are recorded in `MANIFEST.md`, which `verify_manifest.py` re-checks in
  CI on every push, and whose provenance is attested in `SIGNOFF.md`. **No pipeline step writes, reads or
  deletes anything in this namespace**, so a synthetic run cannot overwrite them or be
  mistaken for them; `smoke_test.py` asserts that separation on every push.
* **The pipeline — synthetic.** `00_make_synthetic_data.py` generates a panel with the
  same structure as the case data: 18 grade-level demand units over 55 monthly periods
  (June 2020 – December 2024), spanning the same intermittency range, with lot-sized
  quantities, ex-ante annual "partner" forecasts, and three exogenous drivers. It exists
  so that the pipeline can be executed end to end and inspected. **It does not reproduce
  the paper's numbers**, and running `run_all.py` will not regenerate the contents of
  those tables; the fixture is drawn from parametric distributions, not from the case data.

`01_build_units.py` maps the company's raw transaction file to the 18 analysis units and
cross-checks the transcribed partner forecasts. It requires the confidential source file
and is included for transparency rather than execution.

Researchers who need the underlying data may contact the corresponding author; access
depends on the case company's consent.

## Design decisions the code enforces

**Origin-safe information set, checked row by row.** The machine-learning models are
annual vintages, and the vintage is chosen from the *forecast origin*, never from the
target month. Every forecast row carries `vintage_year`, `training_end` and
`feature_max_date`, and `test_leakage.py` asserts on each row that both dates precede the
issue date and that all horizons from one origin share one vintage. A model fitted
on data through 31 December of year Y serves every origin in year Y+1. Choosing by target
year would let a September 2023 origin forecasting January 2024 use a model trained to the
end of December 2023 — information that did not exist at the origin. `03_ml_models.py`
asserts the property and `test_leakage.py` re-checks it from the outside.

**Separate decision and outcome windows.** Reviews fall on odd months and the protection
interval is five months, so with demand recorded to December 2024 the last review whose
whole interval is observed is July 2024. The primary configuration therefore places no
order after the July 2024 review (`simcore.LAST_DECISION`) and counts cost and service
from January to November 2024 (`simcore.EVAL_START` to `EVAL_END`). Every judged decision
has a fully observed consequence and no counted consequence comes from an unjudged
decision. The calendar year to December, and terminal holding charges of 2.5 and 5 months
on the stock standing at the end, are sensitivities rather than the headline.

**A common service requirement, not exact matching.** A common buffer multiplier does not
deliver a common achieved service, because the calibrating σ is a root-mean-square that
absorbs bias and intermittent errors are not normal. `06_service_frontier.py` sweeps the
multiplier from 0 to 4 for every method and charges each one the cost of its cheapest
configuration reaching *at least* a given achieved aggregate fill. That requirement is a
floor, not an exact match: the selected configurations land above it by differing
margins, so the achieved fill and cycle service of every selected configuration are
reported beside the cost, and a variant that interpolates cost at exactly the required
fill is reported as a robustness check.

**Uncertainty that says what it means.** `07_bootstrap.py` resamples demand units with
replacement, keeping each unit's whole time path. The sixteen units are the operating
portfolio rather than a sample from a population, so the interval measures sensitivity to
portfolio composition. An interval spanning 100 means the data do not resolve the
direction of a difference; it is not evidence of equivalence, and no equivalence margin
was specified.

## What each script does

| Script | Purpose |
|---|---|
| `simcore.py` | The single implementation of the `(R,S)` simulator, imported by every downstream script |
| `00_make_synthetic_data.py` | Generates the synthetic demand panel, partner annual forecasts, and exogenous series |
| `01_build_units.py` | Aggregates raw company records into 18 demand units; cross-checks partner forecasts (needs confidential input) |
| `02_forecasts.py` | Rolling-origin forecasts for the seven non-ML methods, horizons 1–5, origins 2022–2024 |
| `03_ml_models.py` | The three ML configurations as origin-safe annual vintages, with the leakage assertion |
| `04_simulate.py` | The full simulation grid: service targets, cost ratios, lot rounding, terminal charges, current fixed-mark-up policy |
| `05_alignment.py` | Accuracy metrics, SBC intermittency profile, and the accuracy-to-cost rank correlations |
| `06_service_frontier.py` | Buffer-multiplier sweep and cost at matched achieved fill |
| `07_bootstrap.py` | 1,000-resample intervals for both the common-target and matched-service indices |
| `08_figures.py` | Figures 1–6 of the paper at 600 dpi |
| `09_report_numbers.py` | Prints every number quoted in the manuscript from the stored tables |
| `10_seed_study.py` | Repeats the ML configurations under alternative random seeds |
| `11_alpha_sensitivity.py` | Varies the smoothing constant of Croston, SBA, and TSB from 0.05 to 0.30 |
| `12_refit_frequency.py` | Refits the ML models at every origin on all data available at that origin |
| `13_refit_eval.py` | Compares annual vintages with per-origin refitting on the same policy |
| `test_leakage.py` | Row-level information-set audit of the forecast provenance stamps |
| `verify_manifest.py` | Recomputes every released checksum and fails on drift |
| `smoke_test.py` | Fast CI check: script inventory, imports, cross-references, real/synthetic namespace separation, vintage rule, one simulated unit |

## Result files (real data)

These are the aggregate outputs computed from the confidential data and reported in the
paper. `MANIFEST.md` records their SHA-256 hashes, the scripts and command that produced
them, the environment, and the decision and outcome windows.

| File | Content |
|---|---|
| `realcase_table1_accuracy.csv` | Median accuracy per method across the sixteen valid units (Table 1) |
| `realcase_table2_cost_sweep.csv` | Cost index at a common target by backorder-to-holding ratio (Table 2) |
| `realcase_table3_service_requirement.csv` | Cost, achieved fill, achieved cycle service and z at four service requirements, plus the exact-match interpolation (Table 3) |
| `realcase_table4_current_policy.csv` | Fixed mark-up versus error-calibrated buffer (Table 4) |
| `realcase_segment_index.csv` | Cost index by SBC demand segment (Figure 5) |
| `realcase_alignment.csv` | Rank correlation between accuracy and simulated cost (Figure 6) |
| `realcase_intermittency_profile.csv` | Per-unit non-zero rate, ADI, CV², mean demand and SBC class on the pre-2024 history (Section 3, Figure 1) |
| `realcase_service_frontier.csv` | The whole buffer-multiplier sweep behind Figure 4 |
| `realcase_bootstrap_intervals.csv` | Bootstrap intervals on both bases |
| `realcase_error_decomposition.csv` | Bias, standard deviation and RMS of protection-interval errors |
| `realcase_seed_variability.csv`, `realcase_alpha_sensitivity.csv`, `realcase_refit_frequency.csv` | The three sensitivity studies |
| `realcase_simulation_grid.csv` | The full simulation grid underlying everything above |

Two of the eighteen demand units record no demand at all and are excluded from every
aggregation, so the reported results cover sixteen units. Four further units record no
demand inside the outcome window; they contribute cost but not service, and the aggregate
fill rate is a served-over-demanded ratio that handles them correctly.

**What an independent party can and cannot verify.** The code path, the internal
consistency of these tables, and every design rule above can be checked from this
repository, and `verify_manifest.py` confirms that the released files are the ones the
manifest describes. Regenerating the values requires the confidential transaction file and
therefore the data owner; `SIGNOFF.md` is where that attestation is recorded. The boundary
is stated deliberately, here and in the paper.

## Citation

If you use this code, please cite the paper. See `CITATION.cff` for machine-readable
metadata; the reference will be updated once the article is published.

## Licence

Code is released under the MIT Licence (`LICENSE`). The licence covers the code and the
synthetic data generator only; it does not extend to the company's operational data,
which are not part of this repository.
# From forecast accuracy to inventory performance

Reproducible code for the study *"From forecast accuracy to inventory performance: a
simulation-based value assessment of machine-learning demand forecasts in a
periodic-review policy for intermittent export demand"* (prepared for submission to *Transactions on
Computational Modelling and Intelligent Systems*).

The study asks whether better demand forecasts actually produce cheaper inventory
decisions. Ten forecasting approaches — an exporter's incumbent, company-described
ex-ante annual plan, naive and exponential smoothing, Croston, SBA, TSB, ARIMA, and three
machine-learning configurations — are each fed through a sequential simulation of the
company's periodic-review `(R,S)` replenishment policy, with a three-month lead time,
shipping-lot rounding, and a safety stock calibrated to that method's own realised
forecast errors. Methods are then compared on inventory cost **under a common achieved-service
requirement**, not on accuracy alone and not at a common service target.

## Running it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python smoke_test.py       # ~30 s: imports, dependencies, manifest, vintage rule, simulator
python verify_manifest.py  # recompute every released checksum against MANIFEST.md
python run_all.py          # add --quick to skip the three slow sensitivity studies
python test_leakage.py     # information-set checks, also run inside 03_ml_models.py
```

`smoke_test.py` is what CI runs on every push (`.github/workflows/smoke.yml`). It checks
the things that actually broke in earlier revisions: a script referencing a file that is
not in the repository, a policy-logic import error, and the origin-safety of the vintage
rule.

`run_all.py` is the only entry point you need. It deletes every generated file first,
runs the steps in order, stops at the first step that exits non-zero **or** fails to
produce the outputs it declares, and finishes by printing a SHA-256 prefix for each
generated file so two runs can be compared directly. Exact package versions are pinned in
`requirements.txt`; see `environment.md` for the rest of the environment.

Expected running time: roughly 50 minutes for `--quick` and about two hours for the full
pipeline on a modest container. `02_forecasts.py` dominates, because ARIMA orders are
selected by AIC at every origin and the synthetic series converge slowly; it accounts for
about 45 minutes on its own.

## What is real and what is synthetic

The company's export records are commercially confidential and are **not** redistributed.
Two different things are provided, and they should not be confused:

* **`realcase_*.csv` — real.** The aggregate result tables computed from the confidential
  data. Every table and figure reported in the paper is derived from these files. They
  contain normalised cost indices, accuracy metrics, achieved service, and rank
  correlations only; no absolute demand quantity appears in them. Their SHA-256 hashes
  and provenance are recorded in `MANIFEST.md`, which `verify_manifest.py` re-checks in
  CI on every push, and whose provenance is attested in `SIGNOFF.md`. **No pipeline step writes, reads or
  deletes anything in this namespace**, so a synthetic run cannot overwrite them or be
  mistaken for them; `smoke_test.py` asserts that separation on every push.
* **The pipeline — synthetic.** `00_make_synthetic_data.py` generates a panel with the
  same structure as the case data: 18 grade-level demand units over 55 monthly periods
  (June 2020 – December 2024), spanning the same intermittency range, with lot-sized
  quantities, ex-ante annual "partner" forecasts, and three exogenous drivers. It exists
  so that the pipeline can be executed end to end and inspected. **It does not reproduce
  the paper's numbers**, and running `run_all.py` will not regenerate the contents of
  those tables; the fixture is drawn from parametric distributions, not from the case data.

`01_build_units.py` maps the company's raw transaction file to the 18 analysis units and
cross-checks the transcribed partner forecasts. It requires the confidential source file
and is included for transparency rather than execution.

Researchers who need the underlying data may contact the corresponding author; access
depends on the case company's consent.

## Design decisions the code enforces

**Origin-safe information set, checked row by row.** The machine-learning models are
annual vintages, and the vintage is chosen from the *forecast origin*, never from the
target month. Every forecast row carries `vintage_year`, `training_end` and
`feature_max_date`, and `test_leakage.py` asserts on each row that both dates precede the
issue date and that all horizons from one origin share one vintage. A model fitted
on data through 31 December of year Y serves every origin in year Y+1. Choosing by target
year would let a September 2023 origin forecasting January 2024 use a model trained to the
end of December 2023 — information that did not exist at the origin. `03_ml_models.py`
asserts the property and `test_leakage.py` re-checks it from the outside.

**Separate decision and outcome windows.** Reviews fall on odd months and the protection
interval is five months, so with demand recorded to December 2024 the last review whose
whole interval is observed is July 2024. The primary configuration therefore places no
order after the July 2024 review (`simcore.LAST_DECISION`) and counts cost and service
from January to November 2024 (`simcore.EVAL_START` to `EVAL_END`). Every judged decision
has a fully observed consequence and no counted consequence comes from an unjudged
decision. The calendar year to December, and terminal holding charges of 2.5 and 5 months
on the stock standing at the end, are sensitivities rather than the headline.

**A common service requirement, not exact matching.** A common buffer multiplier does not
deliver a common achieved service, because the calibrating σ is a root-mean-square that
absorbs bias and intermittent errors are not normal. `06_service_frontier.py` sweeps the
multiplier from 0 to 4 for every method and charges each one the cost of its cheapest
configuration reaching *at least* a given achieved aggregate fill. That requirement is a
floor, not an exact match: the selected configurations land above it by differing
margins, so the achieved fill and cycle service of every selected configuration are
reported beside the cost, and a variant that interpolates cost at exactly the required
fill is reported as a robustness check.

**Uncertainty that says what it means.** `07_bootstrap.py` resamples demand units with
replacement, keeping each unit's whole time path. The sixteen units are the operating
portfolio rather than a sample from a population, so the interval measures sensitivity to
portfolio composition. An interval spanning 100 means the data do not resolve the
direction of a difference; it is not evidence of equivalence, and no equivalence margin
was specified.

## What each script does

| Script | Purpose |
|---|---|
| `simcore.py` | The single implementation of the `(R,S)` simulator, imported by every downstream script |
| `00_make_synthetic_data.py` | Generates the synthetic demand panel, partner annual forecasts, and exogenous series |
| `01_build_units.py` | Aggregates raw company records into 18 demand units; cross-checks partner forecasts (needs confidential input) |
| `02_forecasts.py` | Rolling-origin forecasts for the seven non-ML methods, horizons 1–5, origins 2022–2024 |
| `03_ml_models.py` | The three ML configurations as origin-safe annual vintages, with the leakage assertion |
| `04_simulate.py` | The full simulation grid: service targets, cost ratios, lot rounding, terminal charges, current fixed-mark-up policy |
| `05_alignment.py` | Accuracy metrics, SBC intermittency profile, and the accuracy-to-cost rank correlations |
| `06_service_frontier.py` | Buffer-multiplier sweep and cost at matched achieved fill |
| `07_bootstrap.py` | 1,000-resample intervals for both the common-target and matched-service indices |
| `08_figures.py` | Figures 1–6 of the paper at 600 dpi |
| `09_report_numbers.py` | Prints every number quoted in the manuscript from the stored tables |
| `10_seed_study.py` | Repeats the ML configurations under alternative random seeds |
| `11_alpha_sensitivity.py` | Varies the smoothing constant of Croston, SBA, and TSB from 0.05 to 0.30 |
| `12_refit_frequency.py` | Refits the ML models at every origin on all data available at that origin |
| `13_refit_eval.py` | Compares annual vintages with per-origin refitting on the same policy |
| `test_leakage.py` | Row-level information-set audit of the forecast provenance stamps |
| `verify_manifest.py` | Recomputes every released checksum and fails on drift |
| `smoke_test.py` | Fast CI check: script inventory, imports, cross-references, real/synthetic namespace separation, vintage rule, one simulated unit |

## Result files (real data)

These are the aggregate outputs computed from the confidential data and reported in the
paper. `MANIFEST.md` records their SHA-256 hashes, the scripts and command that produced
them, the environment, and the decision and outcome windows.

| File | Content |
|---|---|
| `realcase_table1_accuracy.csv` | Median accuracy per method across the sixteen valid units (Table 1) |
| `realcase_table2_cost_sweep.csv` | Cost index at a common target by backorder-to-holding ratio (Table 2) |
| `realcase_table3_service_requirement.csv` | Cost, achieved fill, achieved cycle service and z at four service requirements, plus the exact-match interpolation (Table 3) |
| `realcase_table4_current_policy.csv` | Fixed mark-up versus error-calibrated buffer (Table 4) |
| `realcase_segment_index.csv` | Cost index by SBC demand segment (Figure 5) |
| `realcase_alignment.csv` | Rank correlation between accuracy and simulated cost (Figure 6) |
| `realcase_service_frontier.csv` | The whole buffer-multiplier sweep behind Figure 4 |
| `realcase_bootstrap_intervals.csv` | Bootstrap intervals on both bases |
| `realcase_error_decomposition.csv` | Bias, standard deviation and RMS of protection-interval errors |
| `realcase_seed_variability.csv`, `realcase_alpha_sensitivity.csv`, `realcase_refit_frequency.csv` | The three sensitivity studies |
| `realcase_simulation_grid.csv` | The full simulation grid underlying everything above |

Two of the eighteen demand units record no demand at all and are excluded from every
aggregation, so the reported results cover sixteen units. Four further units record no
demand inside the outcome window; they contribute cost but not service, and the aggregate
fill rate is a served-over-demanded ratio that handles them correctly.

**What an independent party can and cannot verify.** The code path, the internal
consistency of these tables, and every design rule above can be checked from this
repository, and `verify_manifest.py` confirms that the released files are the ones the
manifest describes. Regenerating the values requires the confidential transaction file and
therefore the data owner; `SIGNOFF.md` is where that attestation is recorded. The boundary
is stated deliberately, here and in the paper.

## Citation

If you use this code, please cite the paper. See `CITATION.cff` for machine-readable
metadata; the reference will be updated once the article is published.

## Licence

Code is released under the MIT Licence (`LICENSE`). The licence covers the code and the
synthetic data generator only; it does not extend to the company's operational data,
which are not part of this repository.
