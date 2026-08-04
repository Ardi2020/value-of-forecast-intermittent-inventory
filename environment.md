# Environment used for the reported results

| Item | Value |
|---|---|
| Python | 3.11 |
| Operating system | Debian GNU/Linux (container) |
| BLAS | OpenBLAS bundled with the numpy wheel |
| Package versions | pinned exactly in `requirements.txt` |
| Random seeds | 42 for every model fit; 20260804 for the bootstrap |

Reproduce with:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

`run_all.py` deletes every generated file before it starts, stops at the first step that
exits non-zero or fails to produce its declared outputs, and prints a SHA-256 prefix for
each generated file so that two runs can be compared directly.

The tree ensembles are seeded but not bit-reproducible across CPU architectures with
different thread counts; `10_seed_study.py` quantifies how much of the reported cost
index moves under alternative seeds.
