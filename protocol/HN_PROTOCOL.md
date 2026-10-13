# Protocol: Hidden neuropathy — neuropathic symptoms missed by the standard foot exam (NHANES 1999–2004)

*Pre-specified on 2026-10-10. Group counts were seen in the PN-01 pilot; no associations had been examined. Changes are logged at the bottom.*

## Question
About 8% of US adults aged 40+ report neuropathic foot symptoms (painful sensation or tingling, DIQ120/130) but have a normal 10-g monofilament exam. The monofilament tests large fibers only. **Are these people a hidden neuropathy population, likely small-fiber, that shares the biology and prognosis of exam-confirmed neuropathy?**

## Groups (adults ≥40 with a valid exam and symptom answers)
| Group | Definition |
|---|---|
| A: reference | normal exam, no foot pain/tingling |
| B: painless PN | ≥1 insensate site, no foot pain/tingling |
| C: painful PN | ≥1 insensate site, with foot pain/tingling |
| **D: hidden** | normal exam, foot pain/tingling (feet or both) |
| E: negative control (non-neuropathic pain) | normal exam, no foot pain/tingling, low back pain in the past 3 months (MPQ070 = 1) |
| F: comparison (hands only) | normal exam, pain/tingling in hands only (DIQ130 = 1). Likely compression or other causes |

Groups E and F are subsets defined among people who would otherwise be in A, and A excludes E and F in the comparisons.

## Analyses and predictions
**H1: ML neuropathy-similarity score (the core ML test).**
- **Model:** gradient boosting (LightGBM) trained to separate **B (painless PN) from A**.
- **Features (routine biology only, no symptoms):** demographics, height, weight, BMI, waist, blood pressure, self-reported diabetes, and about 35 routine labs (chemistry panel, CBC, HbA1c, CRP).
- **Validation:** 5-fold cross-fitting, giving out-of-fold scores for A and B. The model averaged over folds then scores C, D, E and F, which it never saw.
- **Comparison:** the logit-score by group in a linear model adjusted for age, sex and race.
- **Prediction (hidden-neuropathy hypothesis):** D > A (p < 0.05), **and** D > E. E is expected ≈ A.
- **Falsified if:** D ≤ A, or D is not greater than E.
- **Sensitivity:** the model refit without diabetes, HbA1c and glucose, and the analysis restricted to people without diabetes.

**H2: Mortality** (linked deaths to 2019, survey-weighted Cox).
- **Model:** groups vs A, adjusted for age, sex, race, education, income, smoking, BMI, diabetes and eGFR.
- **Prediction:** HR(D) > 1. HR(E) is reported as the non-neuropathic-pain comparator.
- **Sensitivities:** excluding deaths in the first 2 years, and people without diabetes only.

**H3: Painful vs painless neuropathy** (C vs B, ML-centered).
- **Model:** LightGBM C vs B on the same features, cross-fitted. Report AUROC (with a permutation null) and SHAP top features.
- **Logistic model:** C vs B with the same adjustment as H2.
- **Prediction:** exploratory, with no directional hypothesis. AUROC is reported with its 95% CI and permutation p-value.

**H4: Epigenetic aging** (1999–2002 DNAm subsample).
- **Contrasts:** GrimAge2 acceleration, (a) D vs A and (b) C vs B. The latter tests *Geroscience* 2025 ("painful DPN = accelerated aging") in a population.
- **Prediction:** exploratory, given the small n (C = 84).

**H5 (separate notebook): Movement signature** (2003–2004 hip accelerometry).
- **Model:** trained B vs A on day-level and minute-level activity features, then applied to D and E.
- **Prediction:** D resembles PN more than E does.

## Reporting
All hypotheses are reported whatever the result. ML performance is reported with 95% CIs. Multiple comparisons are noted. The results are cross-sectional (except mortality), and symptoms are self-reported.

## Phase 2: strengthening analyses (pre-specified 2026-10-10, after H1–H5 results; reported whatever the result)
| ID | Analysis | Supported if |
|---|---|---|
| **S1** | **Cross-cycle replication.** Fit the H1 model (B vs A) on 1999–2002 only; score the 2003–2004 groups; group differences in 2003–2004 only. H2 Cox fit separately in 1999–2002 and 2003–2004 | In 2003–04: D > A (p < 0.05) and D − E > 0. Mortality HR(D) > 1 in both periods (point estimates) |
| **S2** | **Survey-weighted prevalence** of A–F among US adults ≥40 (6-year MEC weights), with PSU-bootstrap 95% CI and an estimated population count | Descriptive |
| **S3** | **Excluding look-alikes**, applied to all groups: (a) sciatica-like pain (low back pain radiating below the knee, MPQ080 = 1); (b) peripheral artery disease (ABI < 0.9) or calf pain on walking (DIQ140 & DIQ150); (c) low vitamin B12 (< 200 pg/mL) where measured. H1 and H2 re-run after each exclusion and all combined | D > A and D − E > 0 (H1); HR(D) > 1 with CI excluding 1 (H2) after combined exclusion |
| **S4** | **Cause-specific mortality** (NCHS leading causes): cardiovascular (heart disease + cerebrovascular), cancer, all other. Cause-specific Cox, groups vs A | Descriptive |
| **S5** | **Drivers of the hidden group's score:** mean SHAP contribution in D minus A, by feature (full model) | Descriptive |
| **S6** | **E-value** for HR(D) and its lower confidence limit | Descriptive |

## Phase 3: caveat fixes (pre-specified 2026-10-10, after Phase 2; sensitivity analyses only — primary results unchanged; all reported)
| ID | Analysis | Reported as |
|---|---|---|
| **F1** | **Stricter hidden-group definition:** D plus corroboration (foot numbness DIQ100/110 in feet/both, **or** foot marked as a pain region MPQ120AE/AF). H1 and H2 re-run with D_strict replacing D | D_strict − A, D_strict − E, HR(D_strict) |
| **F2** | **Additional exclusions** (all groups): self-reported arthritis or gout, cancer history, heavy drinking (≥4 drinks/day), eGFR < 30. Then **all exclusions combined** (F2 + S3) | H1 group differences; HR(D) |
| **F3** | **Mortality with circulation modeled, not excluded:** Cox adding minimum ABI (continuous), claudication and prevalent CVD as covariates. Also restricted to ABI 1.0–1.4 with no claudication. All-cause and cardiovascular deaths | HR(D) |
| **F4** | **Model-agnostic similarity:** H1 repeated with L2 logistic regression and elastic net (standardized, median-imputed) instead of LightGBM | D − A, D − E per model |

## H6: positive-unlabeled (PU) learning — additive ML section (pre-specified 2026-10-10, after H1–H5, S1–S6, F1–F4)
**Framing.** The monofilament exam is an imperfect label. Exam-positive adults (B ∪ C) are **labeled positives**. Everyone with a normal exam is **unlabeled**: a mixture of truly unaffected people and missed neuropathy. **Symptoms are never used as model input.**

**Method.**
- **Model:** LightGBM g(x) = P(labeled | biology), 5-fold cross-fitted and Platt-calibrated, on the H1 feature set.
- **Elkan–Noto (SCAR):** c = mean g(x) over held-out labeled positives. Per-person posterior P(neuropathy | x, unlabeled) = ((1 − c)/c) · g/(1 − g), capped at 1.
- **Hidden fraction α** = weighted mean posterior among unlabeled adults, with a PSU cluster-bootstrap CI (200 refits) and a population count.
- **Assumption:** SCAR is likely violated, because the exam detects more severe disease. That plausibly makes α an **underestimate**. Reported as a limitation.

**Validations (pre-specified criteria).**
| ID | Test | Supported if |
|---|---|---|
| V1 | Among unlabeled adults, does the PU posterior (per SD) predict **foot symptoms** (pain/tingling in the feet)? Logistic, adjusted for age, sex, race. Specificity check: the same model for back pain only | Foot-symptom OR > 1 with p < 0.05, **and** larger than the back-pain OR |
| V2 | Does the posterior predict **mortality** among unlabeled adults? Cox, adjusted for age, sex, race; then also for ABI, claudication and CVD history | HR > 1, p < 0.05 in the first model. The second model is reported |
| V3 | **Replication:** train on 1999–2002, then apply V1 to 2003–2004 unlabeled adults | V1 criterion met in 2003–04 |
| V4 | **Silent fraction:** the share of estimated missed cases (sum of posteriors) among people **without** foot symptoms | Descriptive |

## H6b: diagnosing and improving the PU hidden-fraction estimate (pre-specified 2026-10-10, after H6)
1. **Known-answer (semi-synthetic) test.** Restricted to A ∪ B. Treat B as positives and A as negatives. Randomly move 30% of B into the unlabeled set, so the true hidden fraction among the unlabeled is known. Apply the same Elkan–Noto pipeline and record the estimate vs the truth over 20 random repeats. Done with both the original and the improved model.
2. **Stronger model.** Add vitamin B12, folate, homocysteine, methylmalonic acid, blood lead, blood cadmium, serum cotinine, urine albumin/creatinine ratio, calf circumference, upper-leg and upper-arm length (where measured). LightGBM hyperparameters are tuned by 5-fold CV AUROC on a small grid.
3. **Corrected uncertainty.** Resample 80% of PSUs **without replacement** (200 repeats), refit, and estimate α on the subsample.

**α is credible only if all hold:**
- the known-answer relative error is < 25% with the model used;
- the subsampling 95% interval contains the point estimate;
- α with the improved model is within 0.05 of its subsampling median.

Otherwise α is reported as not estimable from these data.

## Change log
| Date | Change | Reason |
|---|---|---|
| 2026-10-10 | Initial version | — |
| 2026-10-10 | Added Phase 2 strengthening analyses S1–S6 (after H1–H5 results) | Reviewer-anticipated robustness checks; criteria fixed before running |
| 2026-10-10 | Added Phase 3 caveat-fix sensitivity analyses F1–F4 | Address self-report, look-alike conditions, vascular confounding of mortality, model dependence |
| 2026-10-10 | Added H6 positive-unlabeled learning section (additive; primary results unchanged) | Makes ML the primary estimator of missed neuropathy, with symptoms and mortality as independent validation |
| 2026-10-10 | Added H6b: known-answer test, stronger model, corrected uncertainty for α | H6 α estimate unstable; credibility rule fixed before running |

## Results (HN-01, 2026-10-10; results/hn-01/hn_results.json)
| Hypothesis | Verdict | Key numbers |
|---|---|---|
| H1 ML similarity (D > A and D > E) | **Supported** | Model B vs A out-of-fold AUROC 0.715. Logit-score difference vs A: B +0.34, **D +0.26**, E +0.07, F +0.07, C +0.60. **D − E = +0.19 (0.13–0.26), p < 1e-4.** Holds without glycemic features (D +0.25) and in people without diabetes (D +0.18; D − E +0.13, p = 0.0004) |
| H2 Mortality | **Supported** | HR vs A: B 1.38 (1.20–1.59), C 1.88 (1.38–2.56), **D 1.29 (1.06–1.56)**, E 1.01 (0.88–1.17), F 1.17 (0.96–1.41). D excluding deaths < 2 y: 1.25 (1.03–1.52); D without diabetes: 1.31 (1.03–1.65) |
| H3 Painful vs painless | Exploratory | AUROC 0.62 (0.59–0.66), permutation p < 0.001. Top SHAP features: HbA1c, triglycerides, diabetes. Logistic: HbA1c OR 1.38 per %; never vs current smoking OR 0.41; eGFR OR 0.984 per unit |
| H4 Epigenetic aging | Not supported | D vs A +0.09 SD (−0.13 to 0.30); C vs B −0.14 SD (−0.36 to 0.08). Small n |
| H5 Movement | **Not supported** | 2003–04, n = 2,019 with ≥4 valid days. Movement-only model B vs A AUROC 0.63, but B ≈ A after age/sex/race adjustment (+0.05, NS). D +0.10 (NS); D − E +0.03 (NS). Only **painful PN** differed: +0.25 (0.07–0.43), p = 0.007. Hip counts capture painful but not painless/hidden neuropathy |

## Phase 2 results (HN-03, 2026-10-10; results/hn-03/)
| ID | Verdict | Key numbers |
|---|---|---|
| S1 Cross-cycle replication | **H1 replicated**; H2 criterion met on point estimates only | Model trained on 1999–2002, tested on 2003–04: AUROC 0.726; D +0.17 (0.07–0.28), p = 0.001; E +0.03 (NS); **D − E +0.15 (0.02–0.27), p = 0.027**. HR(D): 1999–2002 1.39 (1.11–1.74); 2003–04 1.13 (0.79–1.61) |
| S2 Prevalence (weighted) | — | **D 8.2% (7.4–9.1%) ≈ 8.9 million US adults ≥40** (1999–2004); painless PN 11.0% (11.9 M); painful PN 2.7% (2.9 M) |
| S3 Excluding look-alikes | **H1 robust; H2 not met** (combined exclusion CI includes 1) | D − E stays positive under every exclusion (+0.17 to +0.22, all p < 0.001). HR(D): no sciatica 1.41 (1.08–1.83); no low B12 1.32 (1.08–1.61); **no vascular 1.16 (0.86–1.57)**; all excluded 1.44 (0.98–2.12), p = 0.064 |
| S4 Cause-specific | — | Cardiovascular: D 1.40 (1.01–1.92), E 0.99, F 0.96; painful PN 2.60. Cancer: D 1.41, but E 1.31 and F 1.49 too (non-specific). Other causes: D 1.17 (NS) |
| S5 Drivers | — | D − A SHAP: waist, self-reported diabetes, glucose, RDW, ALP, weight, LDH, BMI, HbA1c. A **metabolic** profile; E − A small |
| S6 E-value | — | HR 1.29 → E-value 1.67 (lower CI 1.26) |

## Phase 3 results (HN-04, 2026-10-10; results/hn-04/)
| ID | Biology (H1) | Mortality (H2) |
|---|---|---|
| F1 strict D (n = 431 corroborated of 644) | **Stronger**: D − A +0.28, D − E +0.21 (both p < 1e-4) | HR 1.31 (1.03–1.66) |
| F2 new exclusions (arthritis/gout, cancer, heavy drinking, eGFR < 30) | **Stronger**: D − A +0.31, D − E +0.22 | HR 1.82 (1.31–2.53) |
| F2 all exclusions combined (n_D = 99) | **Stronger**: D − A +0.31, D − E +0.24 (p = 0.015) | HR 2.34 (1.44–3.78) |
| F3 circulation as covariates (ABI, claudication, CVD history) | — | All-cause 1.20 (0.98–1.49), p = 0.08; CVD 1.16 (NS) |
| F3 restricted to ABI 1.0–1.4, no claudication (n_D = 238) | — | **All-cause 0.98 (0.69–1.37); CVD 0.81 (NS)** — no excess |
| F4 model-agnostic (L2 logistic, elastic net) | **Holds**: AUROC 0.72–0.73; D − A +0.23 / +0.23; D − E +0.17 / +0.17 (p < 1e-8) | — |

**Interpretation:** the biological-similarity finding is robust to symptom definition, exclusions and model choice. The mortality association is not robust: it strengthens after excluding non-neuropathic conditions but disappears among people with clearly normal leg circulation. It cannot be separated from peripheral vascular status in these data.

## H6 results (HN-05, 2026-10-10; results/hn-05/)
| ID | Verdict | Key numbers |
|---|---|---|
| α (hidden fraction) | **Not credible as estimated** | AUROC(labeled vs unlabeled) 0.736; c = 0.275; α = 0.38 (≈ 36 M), but the cluster-bootstrap 95% CI 0.22–0.30 **excludes the point estimate**, so the estimate is unstable. The weak classifier plus the SCAR violation makes the class-prior estimate unreliable. Mean posterior differs little by group (A 0.47, D 0.52, E 0.46, F 0.44) |
| V1 symptoms | **Supported** | Per SD of posterior, among normal-exam adults (age/sex/race adjusted): foot pain/tingling OR **1.64 (1.43–1.89)**; foot numbness 1.61 (1.37–1.90); back pain 1.19 (1.08–1.30); hands only 1.02 (NS) |
| V2 mortality | **Supported** | HR per SD **1.45 (1.35–1.55)**; plus ABI, claudication and CVD history: **1.36 (1.26–1.47)**. Caveat: the posterior is partly a general metabolic-risk score |
| V3 replication (2003–04) | **Supported** | Foot pain/tingling OR **1.30 (1.08–1.55)**; foot numbness 1.54 (1.18–1.99); back pain 1.09 (NS) |
| V4 silent fraction | Not interpretable | 89% of the posterior mass falls in people without symptoms, but it inherits the instability of α |

## H6b results (HN-06, 2026-10-10; results/hn-06/)
| Step | Result |
|---|---|
| Stronger model | 12 extra biomarkers/measures plus tuning: AUROC 0.736 → **0.743**. Negligible gain; biology alone cannot separate exam-positive from exam-negative much better |
| Known-answer test | True hidden fraction 0.09. Estimated **0.58 (big model) / 0.60 (original)**, a relative error of about **+540%**. Elkan–Noto massively overestimates at this level of separability |
| Corrected uncertainty | Subsampling 95% interval 0.355–0.395 now contains α = 0.367. The earlier CI problem came from duplicate resampling |
| **Credibility rule** | **Failed** (known-answer error ≫ 25%). **α is not estimable from these data**; the 36-million figure is withdrawn |
| V1 with stronger model | Foot pain/tingling OR 1.66 per SD (p < 1e-13); back pain 1.17; hands only 1.05 (NS). **The ranking validation stands** |
