# %% [markdown]
# # HN-13 — Design bootstrap of the whole transfer pipeline (co-author revision)
# Rao–Wu rescaling bootstrap: in each stratum, n_h − 1 PSUs are drawn with replacement and weights are rescaled by
# n_h / (n_h − 1) × (times drawn). Each replicate refits the model (5-fold cross-fitting), rescores every group,
# re-estimates the survey-weighted group contrasts and recomputes the survey-weighted biomarker fingerprints and
# cosine similarities. 1,000 replicates; percentile 95% intervals. Point estimates come from the full sample.

# %%
import glob
import json
import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
B = 1000
f = glob.glob("/kaggle/input/**/hn11_analysis.csv.gz", recursive=True)[0]
J = json.load(open(glob.glob("/kaggle/input/**/hn11_results.json", recursive=True)[0]))
d = pd.read_csv(f)
FEATS, FP = J["feature_list"], J["fingerprint_biomarkers"]
co = d[(d.cohort == 1)].reset_index(drop=True)
for lv, code in {"nhb": "NHBlack", "mex": "MexAm", "oth_hisp": "OthHisp", "other": "Other"}.items():
    co[f"race_{lv}"] = (co.race == code).astype(int)
SCORED = ["A_reference", "B_painless_PN", "C_painful_PN", "D_hidden", "E_backpain_ctrl", "F_hands_only"]
GK = dict(zip("ABCDEF", SCORED))
co = co[co.group.isin(SCORED)].reset_index(drop=True)
CONDS = [c for c in co if c.startswith("cond_")]
DOMAINS = {"Glucose & lipids": ["hba1c", "glucose", "trig", "chol", "uric"],
           "Body size": ["waist", "bmi", "weight", "height"], "Blood pressure": ["sbp", "dbp"],
           "Liver & protein": ["alp", "ast", "alt", "ggt", "bili", "albumin", "protein", "ldh"],
           "Kidney & electrolytes": ["egfr", "bun", "creat", "potassium", "sodium", "chloride", "bicarb", "calcium", "phos"],
           "Blood count & inflammation": ["wbc", "lymph_pct", "mono_pct", "neut_pct", "eos_pct", "hgb", "mcv", "rdw", "platelets",
                                          "mpv", "crp", "iron"]}
assert sorted(sum(DOMAINS.values(), [])) == sorted(FP)
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=8, min_child_samples=50, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=5, verbose=-1, random_state=2026, n_jobs=1)
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
# PSU structure from the full examined sample (not only the cohort)
allpsu = d[["SDMVSTRA", "SDMVPSU"]].drop_duplicates().astype(int)
STRATA = {h: g.SDMVPSU.tolist() for h, g in allpsu.groupby("SDMVSTRA")}
print("strata:", len(STRATA), " PSUs:", len(allpsu), " PSUs per stratum:", sorted({len(v) for v in STRATA.values()}))
print("cohort:", len(co), co.group.value_counts().to_dict())

# %%
X_ADJ = np.c_[np.ones(len(co)), co.age, co.female, co.race_nhb, co.race_mex, co.race_oth_hisp, co.race_other].astype(float)
GRP = co.group.values
DUM = np.c_[[(GRP == GK[k]).astype(float) for k in "BCDEF"]].T


def wls(y, X, w):
    sw = np.sqrt(w)
    return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]


def contrasts(score, w):
    m = (w > 0) & np.isfinite(score)
    beta = wls(score[m], np.c_[DUM[m], X_ADJ[m]], w[m])
    b = dict(zip("BCDEF", beta[:5]))
    return {"B_vs_A": b["B"], "C_vs_A": b["C"], "D_vs_A": b["D"], "E_vs_A": b["E"], "F_vs_A": b["F"],
            "D_minus_E": b["D"] - b["E"], "D_minus_F": b["D"] - b["F"], "DE_over_BA": (b["D"] - b["E"]) / b["B"]}


def wmean(x, w):
    return np.sum(x * w) / np.sum(w)


def fingerprints(w):
    """Survey-weighted SMD of each biomarker vs reference (after weighted adjustment for age, sex, race)."""
    prof = {}
    for f_ in FP:
        y = co[f_].values
        m = (w > 0) & np.isfinite(y)
        beta = wls(y[m], X_ADJ[m], w[m])
        res = np.full(len(co), np.nan)
        res[m] = y[m] - X_ADJ[m] @ beta
        ref = m & (GRP == "A_reference")
        mu, sd = wmean(res[ref], w[ref]), np.sqrt(wmean((res[ref] - wmean(res[ref], w[ref])) ** 2, w[ref]))
        for k in "BCDEF":
            g = m & (GRP == GK[k])
            prof.setdefault(k, {})[f_] = (wmean(res[g], w[g]) - mu) / sd
        for c in CONDS:
            cg = m & np.isin(GRP, ["A_reference", "E_backpain_ctrl", "F_hands_only"]) & (co[c].values == 1)
            cr = m & (GRP == "A_reference") & (co[c].values == 0)
            mu_c = wmean(res[cr], w[cr])
            sd_c = np.sqrt(wmean((res[cr] - mu_c) ** 2, w[cr]))
            prof.setdefault(c, {})[f_] = (wmean(res[cg], w[cg]) - mu_c) / sd_c
    return pd.DataFrame(prof)  # rows: biomarkers, columns: profiles


def cosine(a, b):
    return float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))


def similarity_stats(P):
    out = {}
    keys = list("BCDEF")
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            out[f"cos_{a}{b}"] = cosine(P[a], P[b])
    for c in CONDS:
        out[f"cos_B_{c}"] = cosine(P["B"], P[c])
        out[f"cos_D_{c}"] = cosine(P["D"], P[c])
    out["cosdiff_DB_minus_DE"] = out["cos_BD"] - out["cos_DE"]
    out["cosdiff_DB_minus_DF"] = out["cos_BD"] - out["cos_DF"]
    ref = [out[f"cos_B_{c}"] for c in CONDS]
    out["cos_B_conditions_max"] = max(ref)
    out["cosdiff_DB_minus_max_condition"] = out["cos_BD"] - max(ref)
    for dom, fs in DOMAINS.items():
        keep = [x for x in FP if x not in fs]
        out[f"loo_DB_{dom}"] = cosine(P.loc[keep, "D"], P.loc[keep, "B"])
        out[f"loo_DE_{dom}"] = cosine(P.loc[keep, "D"], P.loc[keep, "E"])
    return out


def pipeline(mult, seed):
    """mult: per-person bootstrap multiplicity (0, 1, 2, …); weights are w6 × rescaled multiplicity."""
    w = co.w6.values * mult
    keep = mult > 0
    tr_idx = np.where(keep & np.isin(GRP, ["A_reference", "B_painless_PN"]))[0]
    ot_idx = np.where(keep & ~np.isin(GRP, ["A_reference", "B_painless_PN"]))[0]
    rep = np.round(mult / mult[keep].min()).astype(int)  # copies of each drawn person (1 when n_h = 2)
    tr_rows = np.repeat(tr_idx, rep[tr_idx])
    y = (GRP[tr_rows] == "B_painless_PN").astype(int)
    Xtr, Xot = co.loc[tr_rows, FEATS].values, co.loc[ot_idx, FEATS].values
    oof = np.zeros(len(tr_rows))
    fold = np.zeros((5, len(ot_idx)))
    for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, y)):
        m = lgb.LGBMClassifier(**PARAMS).fit(Xtr[a], y[a])
        oof[b] = m.predict_proba(Xtr[b])[:, 1]
        fold[k] = m.predict_proba(Xot)[:, 1]
    s_avg, s_uni = np.full(len(co), np.nan), np.full(len(co), np.nan)
    # a person drawn twice gets the mean of the copies' out-of-fold scores
    oof_s = pd.Series(logit(oof)).groupby(tr_rows).mean()
    s_avg[oof_s.index], s_uni[oof_s.index] = oof_s.values, oof_s.values
    s_avg[ot_idx] = logit(fold.mean(0))
    pick = np.random.default_rng(seed).integers(0, 5, len(ot_idx))
    s_uni[ot_idx] = logit(fold[pick, np.arange(len(ot_idx))])
    out = {"auroc": roc_auc_score(y, oof)}
    out.update(contrasts(s_avg, w))
    out.update({f"uniform_{k}": v for k, v in contrasts(s_uni, w).items()})
    P = fingerprints(w)
    out.update(similarity_stats(P))
    out.update({f"smd_{c}_{f_}": P.loc[f_, c] for c in P.columns for f_ in FP})
    return out


def draw(b):
    rng = np.random.default_rng(10_000 + b)
    mult_psu = {}
    for h, ps in STRATA.items():
        n = len(ps)
        if n == 1:
            mult_psu[(h, ps[0])] = 1.0
            continue
        picks = rng.choice(ps, n - 1, replace=True)
        for p_ in ps:
            mult_psu[(h, p_)] = (picks == p_).sum() * n / (n - 1)
    return np.array([mult_psu[(int(h), int(p_))] for h, p_ in zip(co.SDMVSTRA, co.SDMVPSU)])


# %%
t0 = time.time()
point = pipeline(np.ones(len(co)), 2026)
print("point estimates (%.1fs):" % (time.time() - t0),
      {k: round(v, 4) for k, v in point.items() if not k.startswith(("smd_", "loo_", "cos_D_cond"))})

# %%
t0 = time.time()
def run(b):
    return pipeline(draw(b), 2027 + b)


reps = Parallel(n_jobs=os.cpu_count(), verbose=5)(delayed(run)(b) for b in range(B))
R = pd.DataFrame(reps)
R.to_csv("hn13_replicates.csv.gz", index=False)
print("replicates: %d in %.1f min" % (len(R), (time.time() - t0) / 60))

# %%
# Bias-corrected (BC) percentile intervals (Efron): z0 = Φ⁻¹(share of replicates below the estimate); the interval takes
# the Φ(2·z0 ± z) quantiles. Bounded statistics such as cosine similarity are pulled down by sampling noise in every
# replicate, so plain percentile intervals would sit below the estimate; BC corrects this median bias and equals the
# percentile interval when the replicates are centred on the estimate.
from scipy.stats import norm

summ = {}
for k, v in point.items():
    r = R[k].dropna().values
    z0 = norm.ppf(np.clip((r < v).mean() + 0.5 * (r == v).mean(), 1 / (2 * len(r)), 1 - 1 / (2 * len(r))))
    bc = lambda z: float(np.quantile(r, norm.cdf(2 * z0 + z)))
    summ[k] = dict(est=float(v), lo=bc(-1.96), hi=bc(1.96), lo66=bc(-0.954), hi66=bc(0.954),
                   pct_lo=float(np.quantile(r, 0.025)), pct_hi=float(np.quantile(r, 0.975)), median=float(np.median(r)),
                   z0=float(z0), se=float(r.std()), p_le_0=float((r <= 0).mean()), n_rep=int(len(r)))
json.dump(dict(B=B, n=len(co), ci="bias-corrected percentile", stats=summ), open("hn13_results.json", "w"), indent=1)
for k in ["auroc", "D_minus_E", "D_minus_F", "D_vs_A", "E_vs_A", "F_vs_A", "B_vs_A", "C_vs_A", "DE_over_BA",
          "uniform_D_minus_E", "uniform_D_vs_A", "uniform_DE_over_BA", "cos_BD", "cos_DE", "cos_DF", "cos_CD", "cos_BE", "cos_BF",
          "cos_BC", "cos_CE", "cos_CF", "cos_EF",
          "cosdiff_DB_minus_DE", "cosdiff_DB_minus_DF", "cos_B_conditions_max", "cosdiff_DB_minus_max_condition"] + \
         [f"cos_B_{c}" for c in CONDS]:
    s = summ[k]
    print(f"{k:34s} {s['est']:7.3f}  BC ({s['lo']:.3f}, {s['hi']:.3f})  pct ({s['pct_lo']:.3f}, {s['pct_hi']:.3f})  median {s['median']:.3f}")

# fingerprint table for the heatmap (bootstrap CIs)
rows = []
for c in list("BCDEF") + CONDS:
    for f_ in FP:
        s = summ[f"smd_{c}_{f_}"]
        rows.append(dict(profile=GK.get(c, c), feature=f_, smd=s["est"], lo=s["lo"], hi=s["hi"]))
pd.DataFrame(rows).to_csv("fingerprint_weighted.csv", index=False)
