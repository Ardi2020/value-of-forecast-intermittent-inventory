# Release manifest — real-case outputs

These files are the aggregate results computed from the confidential company data. They
are the evidence behind every table and figure in the paper. They are not produced by
`run_all.py`, which operates only on the synthetic fixture and cannot write to this
namespace.

| File | SHA-256 | Bytes |
|---|---|---|
| `realcase_alignment.csv` | `cc25362b0ad715d09e7c6ce313248602…` | 4,353 |
| `realcase_alpha_sensitivity.csv` | `dbfb44019ddaf0f8a184cdc9b8a3da81…` | 210 |
| `realcase_bootstrap_intervals.csv` | `ed8b69d80174abcbe83d54944869f981…` | 1,231 |
| `realcase_error_decomposition.csv` | `83847d091b09961c82e8de3efa137111…` | 11,164 |
| `realcase_refit_frequency.csv` | `95dc00318c6fd577e5abec86fac50c0d…` | 871 |
| `realcase_seed_variability.csv` | `0b6eb27d012046429f60b305eb65c8b4…` | 147 |
| `realcase_segment_index.csv` | `da0cac73975c1de7f081bd8e167d7c34…` | 375 |
| `realcase_service_frontier.csv` | `249eac7d0cf505822b17a731fd301bef…` | 802,588 |
| `realcase_simulation_grid.csv` | `801fa428c83027d12508d2cf408074d4…` | 538,423 |
| `realcase_table1_accuracy.csv` | `3894aa956560050fecba0ed96036c158…` | 372 |
| `realcase_table2_cost_sweep.csv` | `4a0cafed5245759bfbd59c0401b9ea02…` | 410 |
| `realcase_table3_service_requirement.csv` | `31b8663f6c398912b37b5913b3063456…` | 1,998 |
| `realcase_table4_current_policy.csv` | `cd1222420a9f043edbde3aedb59a4fde…` | 212 |

## Provenance

| Item | Value |
|---|---|
| Produced by | `04_simulate.py`, `05_alignment.py`, `06_service_frontier.py`, `07_bootstrap.py`, `10_seed_study.py`, `11_alpha_sensitivity.py`, `13_refit_eval.py` |
| Input | the company transaction file, not distributed |
| Command | `python 04_simulate.py && python 05_alignment.py && python 06_service_frontier.py && python 07_bootstrap.py` |
| Python | 3.11, packages pinned in `requirements.txt` |
| Seeds | 42 for every model fit, 20260804 for the bootstrap |
| Decision window | reviews to 2024-07-01 |
| Outcome window | 2024-01-01 to 2024-11-01 |

The run that produced these files was executed by the authors on the confidential data.
An independent party can verify the code path and the internal consistency of these
tables, but cannot regenerate them without the data; regeneration requires the data
owner. That boundary is stated in the manuscript and in this manifest deliberately.
