# Neuropathic foot symptoms despite a normal monofilament examination (NHANES 1999–2004)

Analysis code for:

> Saremi A, Murin PJ. *Neuropathic Foot Symptoms Despite a Normal Monofilament Examination in US Adults: Metabolic
> Profile and Mortality (NHANES 1999–2004).* Manuscript in preparation.

The study uses NHANES 1999–2004 (7,894 adults aged ≥40 years with a monofilament examination and symptom answers)
and the NCHS public-use linked mortality files to 2019. The symptomatic, monofilament-normal (SMN) group reported foot
pain or tingling (DIQ120/130) with a normal 10-g monofilament examination. A gradient-boosting model trained only on
insensate adults without symptoms versus asymptomatic adults scores the SMN group and two negative-control groups
(back pain only, hand symptoms only). The notebooks also compare survey-weighted biomarker profiles and estimate
survey-weighted mortality with design-based variance.

Authors: Ali Saremi, Peyton J. Murin.

## Data

All data are public. Each data notebook downloads what it needs at run time:

- NHANES 1999–2000, 2001–2002 and 2003–2004 files from `wwwn.cdc.gov/Nchs/Data/Nhanes/Public/`
- NHANES public-use linked mortality files (2019) from `ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/`

No data file has to be downloaded by hand.

## Notebooks for the current manuscript

| Notebook | Language | What it does | Used in the paper |
|---|---|---|---|
| `hn-11-revdata` | Python | Cohort and STROBE flow counts, groups, covariates, exclusion flags (urate and gout medication, neurotoxic medication, hepatitis C, thyroid disease and earlier exclusions), eight unrelated conditions, transfer-model scores for every sensitivity analysis | Fig. 1, Methods, Supplementary Tables S1–S3 |
| `hn-12-revsurvey` | R (`survey`) | Design-based analyses: Taylor linearization with masked strata and PSUs, subpopulations, design degrees of freedom; prevalence, score contrasts, Cox models, circulation interaction, Schoenfeld residuals, E-value, Kaplan–Meier curves | Results, Fig. 1, 2b, 4, Supplementary Fig. S1, Table S3 |
| `hn-13-revboot` | Python | Rao–Wu bootstrap (1,000 replicates) of the whole pipeline: model refitting, scoring, contrasts, survey-weighted biomarker SMDs and cosine similarities; bias-corrected percentile intervals | Results, Fig. 2b, 3, Supplementary Fig. S2, Table S4 |
| `hn-14-revfigures` | Python | Plot-only figure notebook; reads the outputs above | All figures |
| `hn-10-table1` | Python | Survey-weighted participant characteristics | Table 1 |

Run order: `hn-11`, then `hn-12` and `hn-13` (both read `hn-11`'s output), then `hn-14`. `hn-10` runs on its own.

## Earlier notebooks

`hn-01` to `hn-07` and `hn-09` are the analyses of an earlier version of the manuscript (cluster-robust variance,
SHAP, UMAP and a positive-unlabelled ranking). They are kept for transparency. The positive-unlabelled estimate of the
hidden fraction (`hn-06-pu2`) failed its known-answer test and is mentioned in one sentence of the supplement.
`hn-02-movement` is an exploratory accelerometry analysis outside the paper's scope. Their figures are in
`figures/earlier/`.

## Running

The notebooks were run on Kaggle (CPU, internet on). Each notebook is kept as source (`.py` percent format or `.R`)
and, for Python, as the notebook that was run (`.ipynb`, built with `tools/make_nb.py`); `kernel-metadata.json` holds
the Kaggle settings. To rerun locally:

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute notebooks/hn-11-revdata/hn-11-revdata.ipynb
```

`hn-12`, `hn-13` and `hn-14` look for their inputs under `/kaggle/input/`; locally, copy `results/` there or change the
input path. `hn-12` needs R ≥ 4.4 with the `survey` and `jsonlite` packages.

## Outputs

`results/` holds the saved output of each notebook:

- `hn-11/hn11_analysis.csv.gz`: analysis file (every examined participant, with the cohort flag, groups, covariates,
  exclusion flags and model scores; derived from public NHANES files); `hn11_results.json`: flow counts, AUROCs, flag counts
- `hn-12/hn12_results.json`: design-based estimates; `km_curves.csv`, `km_at_risk.csv`: Kaplan–Meier curves
- `hn-13/hn13_results.json`: bootstrap estimates and intervals; `hn13_replicates.csv.gz`: all 1,000 replicates;
  `fingerprint_weighted.csv`: biomarker SMDs with intervals
- `hn-10/table1.json`: Table 1
- `hn-01` … `hn-07`: outputs of the earlier notebooks

`figures/` holds the final figures (PDF) and their legends:

| Paper | File |
|---|---|
| Fig. 1 | `Fig1_flow.pdf` |
| Fig. 2 | `Fig2_transfer_test.pdf` |
| Fig. 3 | `Fig3_biomarker_profile.pdf` |
| Fig. 4 | `Fig4_mortality.pdf` |
| Supplementary Fig. S1 | `FigS1_sensitivity.pdf` |
| Supplementary Fig. S2 | `FigS2_similarity.pdf` |

## Protocol

`protocol/HN_PROTOCOL.md` is the study protocol written for the earlier version, with its change log.

## License

Code is released under the MIT License (see `LICENSE`). NHANES data are public-use data from the National Center for
Health Statistics.
