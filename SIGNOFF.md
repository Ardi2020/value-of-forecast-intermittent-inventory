# Data-owner sign-off — real-case outputs

The fourteen `realcase_*` files in this repository were computed from the case company's
confidential transaction records, which are not distributed. An independent party can
verify the code path and the internal consistency of those tables, but cannot regenerate
their values. This file records the attestation that closes that gap.

## Attestation

> I confirm that the fourteen `realcase_*` files listed in `MANIFEST.md`, with the
> SHA-256 hashes recorded there, were produced by the code at the tagged release stated
> below, run against the case company's transaction records, and that no value in them
> was edited by hand after the run.

| Field | Value |
|---|---|
| Release / tag | `v3.0.2` |
| Concept DOI | 10.5281/zenodo.21783340 |
| Files attested | the 14 `realcase_*` files listed in `MANIFEST.md` |
| Decision window | reviews up to and including 2024-07-01 |
| Outcome window | 2024-01-01 to 2024-11-01 |
| Command | the sequence in `MANIFEST.md` § Command |
| Environment | Python 3.11, versions pinned in `requirements.txt` |

| | |
|---|---|
| Name | ................................................ |
| Role | authorised analyst / data owner |
| Affiliation | Department of Industrial Engineering, Universitas Andalas |
| Date | ................................................ |
| Signature | ................................................ |

## Why this is required rather than optional

The evaluation in the paper rests on numbers that no reader can recompute. Releasing the
code, pinning the environment, publishing the hashes and running the pipeline on a
synthetic fixture establishes that the code does what the paper says it does. It does not
establish that the released tables came from running that code on the real data. Only
someone with access to the data can attest to that, and this file is where they do it.

The attestation is deliberately narrow: it covers provenance of the released files, not
the correctness of the research design, which is the manuscript's own responsibility and
is open to review in the usual way.
