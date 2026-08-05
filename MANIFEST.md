# Release manifest — real-case outputs

These files are the aggregate results computed from the confidential company data. They
are the evidence behind every table and figure in the paper, and behind the descriptive
statistics of Section 3 — `realcase_intermittency_profile.csv` carries the per-unit
non-zero rate, ADI, CV² and SBC class on which the portfolio description and the segment
analysis rest. They are not produced by `run_all.py`, which operates only on the synthetic
fixture and cannot write into this namespace.

`verify_manifest.py` recomputes every hash below and exits non-zero on any mismatch; it
runs in CI on every push, so a stale entry fails the build rather than sitting unnoticed.

## Files

| File | Produced by | SHA-256 | Bytes |
|---|---|---|---|
| `realcase_alignment.csv` | `05_alignment.py` | `cc25362b0ad715d09e7c6ce313248602fa809c3767e51f4e2f152cfc8a971cf6` | 4,353 |
| `realcase_alpha_sensitivity.csv` | `11_alpha_sensitivity.py` | `dbfb44019ddaf0f8a184cdc9b8a3da8131546a15129fdef36ee0c1915852827b` | 210 |
| `realcase_bootstrap_intervals.csv` | `07_bootstrap.py` | `ed8b69d80174abcbe83d54944869f9817eaca9e6fa4302c2d96020c454beefee` | 1,231 |
| `realcase_error_decomposition.csv` | `04_simulate.py` | `83847d091b09961c82e8de3efa13711108a5a5bc71dfd6ba23b751334b07fd30` | 11,164 |
| `realcase_intermittency_profile.csv` | `05_alignment.py` | `9b78cc9079f1c2b806fba7af5cb35279a740583499a6baf8320926d201d3dfe5` | 1,576 |
| `realcase_refit_frequency.csv` | `12_refit_frequency.py + 13_refit_eval.py` | `dbbda481004fe379e70df33711133609721831af8f59df45c2bba63584d0dea7` | 742 |
| `realcase_seed_variability.csv` | `10_seed_study.py` | `0b6eb27d012046429f60b305eb65c8b41ebef9f0a6141bf2e53a1ad6e0331d90` | 147 |
| `realcase_segment_index.csv` | `05_alignment.py + 04_simulate.py` | `da0cac73975c1de7f081bd8e167d7c34ecc78522499ba747cb7aa9d00150b02d` | 375 |
| `realcase_service_frontier.csv` | `06_service_frontier.py` | `249eac7d0cf505822b17a731fd301befd1e9aefd7b6489d99afe65aeb5346a3f` | 802,588 |
| `realcase_simulation_grid.csv` | `04_simulate.py` | `801fa428c83027d12508d2cf408074d44c9bb53e8c656ec41a64d8a5bafbda75` | 538,423 |
| `realcase_table1_accuracy.csv` | `05_alignment.py` | `3894aa956560050fecba0ed96036c158604b18ecfb7c03092c73a1ef73386060` | 372 |
| `realcase_table2_cost_sweep.csv` | `04_simulate.py` | `4a0cafed5245759bfbd59c0401b9ea02415d9a651042916e5b65bb6b5b8d00ba` | 410 |
| `realcase_table3_service_requirement.csv` | `06_service_frontier.py` | `31b8663f6c398912b37b5913b3063456717e757cf70560aa969b49f66fa74bbe` | 1,998 |
| `realcase_table4_current_policy.csv` | `04_simulate.py` | `cd1222420a9f043edbde3aedb59a4fde47e55411753072b2baaa4563193221af` | 212 |

## Provenance

| Item | Value |
|---|---|
| Input | the company transaction file, not distributed |
| Python | 3.11, packages pinned exactly in `requirements.txt` |
| Seeds | 42 for every model fit; 20260804 for the bootstrap |
| Decision window | reviews up to and including 2024-07-01 (`simcore.LAST_DECISION`) |
| Outcome window | 2024-01-01 to 2024-11-01 (`simcore.EVAL_START` to `EVAL_END`) |
| Calendar-year sensitivity | outcome window extended to 2024-12-01 |
| Terminal sensitivity | k months of holding on ending on-hand plus pipeline, k in {0, 2.5, 5} |

### Command

The full sequence that produced every file above, in order. The sensitivity studies
(10, 11, 13) are part of it: the earlier version of this manifest listed only 04-07,
which did not account for three of the fourteen files.

```bash
python 02_forecasts.py          # rolling-origin forecasts, seven non-ML methods
python 03_ml_models.py          # origin-safe annual ML vintages, with the leakage assertion
python 04_simulate.py           # the full simulation grid
python 05_alignment.py          # accuracy, intermittency profile, rank alignment
python 06_service_frontier.py   # buffer sweep and the service-requirement readings
python 07_bootstrap.py          # 1,000-resample intervals on both bases
python 10_seed_study.py         # random-seed variability of the ML configurations
python 11_alpha_sensitivity.py  # smoothing constant of Croston, SBA, TSB
python 12_refit_frequency.py    # per-origin ML refits
python 13_refit_eval.py         # yearly vintages versus per-origin refitting
```

`01_build_units.py` precedes all of these and requires the confidential transaction file.

## Verification boundary

The code path, the internal consistency of these tables, and every design rule in the
README can be checked from this repository. Regenerating the values requires the
confidential transaction file and therefore the data owner. `SIGNOFF.md` records that
attestation.
# Release manifest — real-case outputs

These files are the aggregate results computed from the confidential company data. They
are the evidence behind every table and figure in the paper. They are not produced by
`run_all.py`, which operates only on the synthetic fixture and cannot write into this
namespace.

`verify_manifest.py` recomputes every hash below and exits non-zero on any mismatch; it
runs in CI on every push, so a stale entry fails the build rather than sitting unnoticed.

## Files

| File | Produced by | SHA-256 | Bytes |
|---|---|---|---|
| `realcase_alignment.csv` | `05_alignment.py` | `cc25362b0ad715d09e7c6ce313248602fa809c3767e51f4e2f152cfc8a971cf6` | 4,353 |
| `realcase_alpha_sensitivity.csv` | `11_alpha_sensitivity.py` | `dbfb44019ddaf0f8a184cdc9b8a3da8131546a15129fdef36ee0c1915852827b` | 210 |
| `realcase_bootstrap_intervals.csv` | `07_bootstrap.py` | `ed8b69d80174abcbe83d54944869f9817eaca9e6fa4302c2d96020c454beefee` | 1,231 |
| `realcase_error_decomposition.csv` | `04_simulate.py` | `83847d091b09961c82e8de3efa13711108a5a5bc71dfd6ba23b751334b07fd30` | 11,164 |
| `realcase_refit_frequency.csv` | `12_refit_frequency.py + 13_refit_eval.py` | `dbbda481004fe379e70df33711133609721831af8f59df45c2bba63584d0dea7` | 742 |
| `realcase_seed_variability.csv` | `10_seed_study.py` | `0b6eb27d012046429f60b305eb65c8b41ebef9f0a6141bf2e53a1ad6e0331d90` | 147 |
| `realcase_segment_index.csv` | `05_alignment.py + 04_simulate.py` | `da0cac73975c1de7f081bd8e167d7c34ecc78522499ba747cb7aa9d00150b02d` | 375 |
| `realcase_service_frontier.csv` | `06_service_frontier.py` | `249eac7d0cf505822b17a731fd301befd1e9aefd7b6489d99afe65aeb5346a3f` | 802,588 |
| `realcase_simulation_grid.csv` | `04_simulate.py` | `801fa428c83027d12508d2cf408074d44c9bb53e8c656ec41a64d8a5bafbda75` | 538,423 |
| `realcase_table1_accuracy.csv` | `05_alignment.py` | `3894aa956560050fecba0ed96036c158604b18ecfb7c03092c73a1ef73386060` | 372 |
| `realcase_table2_cost_sweep.csv` | `04_simulate.py` | `4a0cafed5245759bfbd59c0401b9ea02415d9a651042916e5b65bb6b5b8d00ba` | 410 |
| `realcase_table3_service_requirement.csv` | `06_service_frontier.py` | `31b8663f6c398912b37b5913b3063456717e757cf70560aa969b49f66fa74bbe` | 1,998 |
| `realcase_table4_current_policy.csv` | `04_simulate.py` | `cd1222420a9f043edbde3aedb59a4fde47e55411753072b2baaa4563193221af` | 212 |

## Provenance

| Item | Value |
|---|---|
| Input | the company transaction file, not distributed |
| Python | 3.11, packages pinned exactly in `requirements.txt` |
| Seeds | 42 for every model fit; 20260804 for the bootstrap |
| Decision window | reviews up to and including 2024-07-01 (`simcore.LAST_DECISION`) |
| Outcome window | 2024-01-01 to 2024-11-01 (`simcore.EVAL_START` to `EVAL_END`) |
| Calendar-year sensitivity | outcome window extended to 2024-12-01 |
| Terminal sensitivity | k months of holding on ending on-hand plus pipeline, k in {0, 2.5, 5} |

### Command

The full sequence that produced every file above, in order. The sensitivity studies
(10, 11, 13) are part of it: the earlier version of this manifest listed only 04-07,
which did not account for three of the thirteen files.

```bash
python 02_forecasts.py          # rolling-origin forecasts, seven non-ML methods
python 03_ml_models.py          # origin-safe annual ML vintages, with the leakage assertion
python 04_simulate.py           # the full simulation grid
python 05_alignment.py          # accuracy, intermittency profile, rank alignment
python 06_service_frontier.py   # buffer sweep and the service-requirement readings
python 07_bootstrap.py          # 1,000-resample intervals on both bases
python 10_seed_study.py         # random-seed variability of the ML configurations
python 11_alpha_sensitivity.py  # smoothing constant of Croston, SBA, TSB
python 12_refit_frequency.py    # per-origin ML refits
python 13_refit_eval.py         # yearly vintages versus per-origin refitting
```

`01_build_units.py` precedes all of these and requires the confidential transaction file.

## Verification boundary

The code path, the internal consistency of these tables, and every design rule in the
README can be checked from this repository. Regenerating the values requires the
confidential transaction file and therefore the data owner. `SIGNOFF.md` records that
attestation.
