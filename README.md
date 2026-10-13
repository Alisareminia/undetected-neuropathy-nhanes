# Undetected neuropathy in US adults with a normal monofilament examination

Analysis code for:

> Saremi A. *Evidence of undetected neuropathy in US adults with neuropathic foot symptoms and a normal monofilament
> examination: machine-learning transfer and positive-unlabelled learning.* Manuscript in preparation.

The study uses NHANES 1999–2004 (7,894 adults aged ≥40 years with a monofilament examination and symptom answers)
and the NCHS public-use linked mortality files to 2019. "Hidden neuropathy" is foot pain or tingling (DIQ120/130)
with a normal 10-g monofilament examination. A gradient-boosting model trained only on exam-confirmed painless
neuropathy versus asymptomatic adults scores the hidden group and two negative-control groups (back pain only,
hand symptoms only). The notebooks also compare adjusted biomarker profiles, rank normal-exam adults with
positive-unlabelled learning and estimate survey-weighted mortality.

Author: Ali Saremi.

## Data

All data are public. Each analysis notebook downloads what it needs at run time:

- NHANES 1999–2000, 2001–2002 and 2003–2004 files from `wwwn.cdc.gov/Nchs/Data/Nhanes/Public/`
- NHANES public-use linked mortality files (2019) from `ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/`

No data file has to be downloaded by hand.

## Notebooks

Each notebook is kept in two forms: the percent-format script (`.py`, the source) and the notebook that was run on
Kaggle (`.ipynb`, built with `tools/make_nb.py`). `kernel-metadata.json` holds the Kaggle settings.

| Notebook | What it does | Used in the paper |
|---|---|---|
| `hn-01-analysis` | Data build, transfer test (LightGBM, cross-fitted), group contrasts, mortality (Cox) | Transfer test, mortality, Table 2 |
| `hn-03-strengthen` | Held-out 2003–2004 cycle, prevalence, exclusions, cause-specific mortality, SHAP, E-value | Results, Fig. 2–3, Fig. 6 |
| `hn-04-fixes` | Corroborated-symptom definition, further exclusions, circulation adjustment, logistic and elastic-net learners | Fig. 2–3, Fig. 6 |
| `hn-05-pu` | Positive-unlabelled ranking and its validation against symptoms and death | Fig. 5 |
| `hn-06-pu2` | Known-answer test of the PU hidden-fraction estimate (failed; no fraction reported) | Table 2 |
| `hn-07-figdata` | Exports figure aggregates: Kaplan–Meier curves, biomarker fingerprints, PU deciles, UMAP | Fig. 2, 4, 5, 6 |
| `hn-09-figures` | Plot-only figure notebook; reads the outputs above | All figures |
| `hn-10-table1` | Survey-weighted participant characteristics | Table 1 |
| `hn-02-movement` | Exploratory accelerometry analysis from the protocol (H5) | Not reported; outside the paper's scope |

`hn-01-analysis` also contains two exploratory protocol analyses that are outside the paper's scope: painful versus
painless neuropathy (H3) and GrimAge2 epigenetic ageing (H4).

Run order: `hn-01`, `hn-03`, `hn-04`, `hn-05`, `hn-06`, `hn-07`, `hn-10` (each runs on its own), then
`hn-09-figures`.

## Running

The notebooks were run on Kaggle (CPU, internet on). To rerun locally:

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute notebooks/hn-01-analysis/hn-01-analysis.ipynb
```

`hn-09-figures` looks for its inputs under `/kaggle/input/`. Locally, either copy `results/` there or change the
path in its `find()` helper to point at `results/`.

## Outputs

`results/` holds the saved output of each notebook:

- `hn-01` … `hn-06`: model estimates as JSON (contrasts, confidence intervals, hazard ratios, odds ratios, AUROCs)
- `hn-01/hn_scores.parquet`: out-of-fold transfer scores per participant (NHANES `SEQN`)
- `hn-07`: figure aggregates (CSV)
- `hn-10/table1.json`: Table 1

`figures/` holds the final figures (PDF) and their legends. Notebook figure numbers differ from the paper's:

| Paper | File |
|---|---|
| Fig. 1 | `Fig1_study_concept.pdf` |
| Fig. 2 | `Fig2_transfer_test.pdf` |
| Fig. 3 | `Fig6_robustness.pdf` |
| Fig. 4 | `Fig3_fingerprint.pdf` |
| Fig. 5 | `Fig4_pu_ranking.pdf` |
| Fig. 6 | `Fig5_mortality.pdf` |

## Protocol

`protocol/HN_PROTOCOL.md` is the written study protocol: groups, models, falsification criteria, later analysis
phases with their pre-specified criteria, a change log and the results recorded against each criterion.

## License

Code is released under the MIT License (see `LICENSE`). NHANES data are public-use data from the National Center for
Health Statistics.
