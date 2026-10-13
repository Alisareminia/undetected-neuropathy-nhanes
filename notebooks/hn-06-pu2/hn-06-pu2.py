# %% [markdown]
# # HN-06 — H6b: diagnosing and improving the PU hidden-fraction estimate (HN_PROTOCOL.md).

# %%
import subprocess, sys
try:
    import lifelines  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lifelines"], check=True)

import io
import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
S = requests.Session()
S.headers["User-Agent"] = "neuropath-research/0.1 (academic project)"
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"
RNG = np.random.default_rng(2026)
pd.set_option("display.width", 220)
OUT = {}

# %% [markdown]
# ## Data

# %%
FILES = {
    "DEMO": {1999: "DEMO", 2001: "DEMO_B", 2003: "DEMO_C"}, "LEXPN": {1999: "LEXPN", 2001: "LEXPN_B", 2003: "LEXPN_C"},
    "DIQ": {1999: "DIQ", 2001: "DIQ_B", 2003: "DIQ_C"}, "MPQ": {1999: "MPQ", 2001: "MPQ_B", 2003: "MPQ_C"},
    "BMX": {1999: "BMX", 2001: "BMX_B", 2003: "BMX_C"}, "BPX": {1999: "BPX", 2001: "BPX_B", 2003: "BPX_C"},
    "SMQ": {1999: "SMQ", 2001: "SMQ_B", 2003: "SMQ_C"}, "GHB": {1999: "LAB10", 2001: "L10_B", 2003: "L10_C"},
    "BIO": {1999: "LAB18", 2001: "L40_B", 2003: "L40_C"}, "CBC": {1999: "LAB25", 2001: "L25_B", 2003: "L25_C"},
    "CRP": {1999: "LAB11", 2001: "L11_B", 2003: "L11_C"},
    "LEXAB": {1999: "LEXABPI", 2001: "LEXAB_B", 2003: "LEXAB_C"},
    "MCQ": {1999: "MCQ", 2001: "MCQ_B", 2003: "MCQ_C"}, "ALQ": {1999: "ALQ", 2001: "ALQ_B", 2003: "ALQ_C"},
}


def xpt(year, name):
    r = S.get(BASE.format(year=year, name=name), timeout=240)
    df = pd.read_sas(io.BytesIO(r.content), format="xport")
    df["SEQN"] = df["SEQN"].round().astype(int)
    return df.drop_duplicates("SEQN")


T = {k: pd.concat([xpt(y, n) for y, n in v.items()], ignore_index=True) for k, v in FILES.items()}
d = T["DEMO"][["SEQN", "SDDSRVYR", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "DMDEDUC2", "INDFMPIR", "WTMEC2YR", "WTMEC4YR",
               "SDMVPSU", "SDMVSTRA"]]
for k in ["LEXPN", "DIQ", "MPQ", "BMX", "BPX", "SMQ", "GHB", "BIO", "CBC", "CRP", "LEXAB", "MCQ", "ALQ"]:
    d = d.merge(T[k], on="SEQN", how="left", suffixes=("", f"__{k}"))
d = d.round(10)

MORT = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_{a}_{b}_MORT_2019_PUBLIC.dat"
mort = pd.concat([pd.read_fwf(io.StringIO(S.get(MORT.format(a=y, b=y + 1), timeout=120).text),
                              colspecs=[(0, 6), (14, 15), (15, 16), (16, 19), (45, 48)],
                              names=["SEQN", "ELIGSTAT", "MORTSTAT", "UCOD", "PERMTH_EXAM"], na_values=["."]) for y in (1999, 2001, 2003)])
d = d.merge(mort, on="SEQN", how="left")
dn = pd.read_sas(io.BytesIO(S.get("https://wwwn.cdc.gov/nchs/data/nhanes/dnam/dnmepi.sas7bdat", timeout=300).content), format="sas7bdat")
dn.columns = [c.upper() for c in dn.columns]
dn["SEQN"] = dn.SEQN.round().astype(int)
d = d.merge(dn[["SEQN", "GRIMAGE2MORT", "WTDN4YR"]], on="SEQN", how="left")

# %% [markdown]
# ## Groups and variables

# %%
def first(df, names):
    s = pd.Series(np.nan, index=df.index)
    for n in names:
        if n in df:
            s = s.fillna(df[n])
    return s


L_, R_ = d.LEALPN.where(d.LEALPN >= 0), d.LEARPN.where(d.LEARPN >= 0)
d["pn"] = np.where((L_ >= 1) | (R_ >= 1), 1, np.where((L_ == 0) & (R_ == 0), 0, np.nan))
d["foot_sx"] = np.where(d.DIQ120 == 1, d.DIQ130.isin([2, 3]).astype(float), np.where(d.DIQ120 == 2, 0.0, np.nan))
d["hand_only_sx"] = ((d.DIQ120 == 1) & (d.DIQ130 == 1)).astype(float)
d["back_pain"] = (d.MPQ070 == 1).astype(float)
d = d[(d.RIDAGEYR >= 40) & d.pn.notna() & d.DIQ120.isin([1, 2])].copy()
d["group"] = np.select(
    [(d.pn == 1) & (d.foot_sx == 1), (d.pn == 1) & (d.foot_sx == 0), (d.pn == 0) & (d.foot_sx == 1),
     (d.pn == 0) & (d.foot_sx == 0) & (d.hand_only_sx == 1),
     (d.pn == 0) & (d.foot_sx == 0) & (d.hand_only_sx == 0) & (d.back_pain == 1),
     (d.pn == 0) & (d.foot_sx == 0) & (d.hand_only_sx == 0) & (d.back_pain == 0)],
    ["C_painful_PN", "B_painless_PN", "D_hidden", "F_hands_only", "E_backpain_ctrl", "A_reference"], default="other")
print(d.group.value_counts())

d["age"], d["female"] = d.RIDAGEYR, (d.RIAGENDR == 2).astype(int)
d["race"] = d.RIDRETH1.map({1: "MexAm", 2: "OthHisp", 3: "NHWhite", 4: "NHBlack", 5: "Other"})
d["educ"] = d.DMDEDUC2.map({1: "<HS", 2: "<HS", 3: "HS", 4: "SomeCol", 5: "College"})
d["pir"] = d.INDFMPIR
d["smoke"] = np.select([d.SMQ020 == 2, d.SMQ040.isin([1, 2]), d.SMQ040 == 3], ["never", "current", "former"], default="na")
d.loc[d.smoke == "na", "smoke"] = np.nan
d["dm_self"] = (d.DIQ010 == 1).astype(int)
d["hba1c"] = d.LBXGH
d["diabetes"] = ((d.DIQ010 == 1) | (d.hba1c >= 6.5)).astype(int)
scr = first(d, ["LBXSCR", "LBDSCR"])
scr_c = scr.where(d.SDDSRVYR != 1, 1.013 * scr + 0.147)
k_, a_ = np.where(d.female == 1, 0.7, 0.9), np.where(d.female == 1, -0.241, -0.302)
d["egfr"] = 142 * np.minimum(scr_c / k_, 1) ** a_ * np.maximum(scr_c / k_, 1) ** -1.2 * 0.9938 ** d.age * np.where(d.female == 1, 1.012, 1)
d["sbp"] = d[["BPXSY1", "BPXSY2", "BPXSY3"]].replace(0, np.nan).mean(axis=1)
d["dbp"] = d[["BPXDI1", "BPXDI2", "BPXDI3"]].replace(0, np.nan).mean(axis=1)
LABMAP = {"albumin": ["LBXSAL"], "alp": ["LBXSAPSI", "LBDSAPSI"], "ast": ["LBXSASSI"], "alt": ["LBXSATSI"], "bun": ["LBXSBU"],
          "calcium": ["LBXSCA"], "chol": ["LBXSCH"], "bicarb": ["LBXSC3SI"], "creat": ["LBXSCR", "LBDSCR"], "ggt": ["LBXSGTSI"],
          "glucose": ["LBXSGL"], "iron": ["LBXSIR"], "ldh": ["LBXSLDSI", "LBDSLDSI"], "phos": ["LBXSPH", "LBDSPH"],
          "bili": ["LBXSTB", "LBDSTB"], "protein": ["LBXSTP"], "trig": ["LBXSTR"], "uric": ["LBXSUA"], "sodium": ["LBXSNASI"],
          "potassium": ["LBXSKSI"], "chloride": ["LBXSCLSI"], "wbc": ["LBXWBCSI"], "lymph_pct": ["LBXLYPCT"],
          "mono_pct": ["LBXMOPCT"], "neut_pct": ["LBXNEPCT"], "eos_pct": ["LBXEOPCT"], "hgb": ["LBXHGB"], "mcv": ["LBXMCVSI"],
          "rdw": ["LBXRDW"], "platelets": ["LBXPLTSI"], "mpv": ["LBXMPSI"], "crp": ["LBXCRP"]}
for k, v in LABMAP.items():
    d[k] = first(d, v)
d["height"], d["weight"], d["bmi"], d["waist"] = d.BMXHT, d.BMXWT, d.BMXBMI, d.BMXWAIST
for lv, code in {"nhb": 4, "mex": 1, "oth_hisp": 2, "other": 5}.items():
    d[f"race_{lv}"] = (d.RIDRETH1 == code).astype(int)
FEATS = ["age", "female", "race_nhb", "race_mex", "race_oth_hisp", "race_other", "height", "weight", "bmi", "waist", "sbp", "dbp",
         "dm_self", "hba1c", "egfr"] + list(LABMAP)
FEATS_NO_GLY = [f for f in FEATS if f not in ("dm_self", "hba1c", "glucose")]
d["cluster"] = (d.SDMVSTRA * 10 + d.SDMVPSU).astype(int)
d["w"] = np.where(d.SDDSRVYR.isin([1, 2]), d.WTMEC4YR * 2 / 3, d.WTMEC2YR / 3)
d["w"] = d.w / d.w.mean()

# %% [markdown]
# ## Extra variables: vitamin B12, look-alike flags, raw 6-year weights

# %%
b12 = []
for year, cands in {1999: ["LAB06"], 2001: ["L06_B"], 2003: ["L06BMT_C", "L06_C", "L06MH_C"]}.items():
    for nm in cands:
        try:
            x = xpt(year, nm)
            if any(c in x for c in ["LBXB12", "LBDB12"]):
                b12.append(x[["SEQN"] + [c for c in ["LBXB12", "LBDB12"] if c in x]])
                print("B12 from", nm)
                break
        except Exception:
            pass
if b12:
    bb = pd.concat(b12, ignore_index=True).drop_duplicates("SEQN")
    bb["b12"] = first(bb, ["LBXB12", "LBDB12"])
    d = d.merge(bb[["SEQN", "b12"]], on="SEQN", how="left")
else:
    d["b12"] = np.nan
print("B12 measured:", d.b12.notna().mean().round(3))

abi = d[[c for c in ["LEXLABPI", "LEXRABPI"] if c in d]].min(axis=1)
d["x_sciatica"] = (d.MPQ080 == 1).astype(int)
d["x_vascular"] = ((abi < 0.9) | ((d.DIQ140 == 1) & (d.DIQ150 == 1))).astype(int)
d["x_lowb12"] = (d.b12 < 200).astype(int)
d["w_raw"] = np.where(d.SDDSRVYR.isin([1, 2]), d.WTMEC4YR * 2 / 3, d.WTMEC2YR / 3)
GROUPS = ["B_painless_PN", "C_painful_PN", "D_hidden", "E_backpain_ctrl", "F_hands_only"]
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=8, min_child_samples=50, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=5, verbose=-1, random_state=2026)


def group_diffs(x, score_col, wcol="w"):
    x = x[x.group != "other"].dropna(subset=[score_col, "race"])
    r = smf.wls(f"{score_col} ~ C(group, Treatment('A_reference')) + age + female + C(race)", x, weights=x[wcol]).fit(
        cov_type="cluster", cov_kwds={"groups": x.cluster})
    tab = pd.DataFrame({"diff_vs_A": r.params, "lo": r.conf_int()[0], "hi": r.conf_int()[1], "p": r.pvalues})
    tab = tab[tab.index.str.startswith("C(group")]
    tab.index = tab.index.str.extract(r"T\.(\w+)\]")[0].values
    cv = pd.Series(0.0, index=r.params.index)
    cv[[i for i in cv.index if "D_hidden" in i]] = 1
    cv[[i for i in cv.index if "E_backpain_ctrl" in i]] = -1
    t = r.t_test(cv.values)
    return tab.round(5).to_dict("index"), dict(diff=float(t.effect[0]), lo=float(t.conf_int()[0][0]),
                                               hi=float(t.conf_int()[0][1]), p=float(t.pvalue))


def crossfit_scores(df, feats):
    """Cross-fitted B-vs-A model within df; returns logit scores for every non-'other' row."""
    train = df[df.group.isin(["A_reference", "B_painless_PN"])]
    y = (train.group == "B_painless_PN").astype(int).values
    oof = np.zeros(len(train))
    oth = df[~df.group.isin(["A_reference", "B_painless_PN", "other"])]
    op = np.zeros(len(oth))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=2026).split(train, y):
        m = lgb.LGBMClassifier(**PARAMS).fit(train[feats].iloc[tr], y[tr])
        oof[te] = m.predict_proba(train[feats].iloc[te])[:, 1]
        op += m.predict_proba(oth[feats])[:, 1] / 5
    sc = pd.concat([pd.Series(oof, index=train.index), pd.Series(op, index=oth.index)]).clip(1e-4, 1 - 1e-4)
    return np.log(sc / (1 - sc)), roc_auc_score(y, oof)


def cox_groups(df, event_mask=None):
    x = df[(df.ELIGSTAT == 1) & (df.PERMTH_EXAM > 0) & (df.group != "other")].copy()
    x["t"] = x.PERMTH_EXAM / 12
    x["e"] = ((x.MORTSTAT == 1) & (event_mask(x) if event_mask is not None else True)).astype(int)
    for g in GROUPS:
        x[g] = (x.group == g).astype(int)
    for lv in ["NHBlack", "MexAm", "OthHisp", "Other"]:
        x[f"r_{lv}"] = (x.race == lv).astype(int)
    x["sm_cur"], x["sm_for"] = (x.smoke == "current").astype(int), (x.smoke == "former").astype(int)
    x["ed_lt"], x["ed_hs"], x["ed_sc"] = [(x.educ == v).astype(int) for v in ["<HS", "HS", "SomeCol"]]
    cols = GROUPS + ["age", "female", "r_NHBlack", "r_MexAm", "r_OthHisp", "r_Other", "pir", "sm_cur", "sm_for",
                     "ed_lt", "ed_hs", "ed_sc", "bmi", "diabetes", "egfr"]
    x = x[cols + ["t", "e", "w", "cluster"]].dropna()
    c = CoxPHFitter(penalizer=1e-6).fit(x, "t", "e", weights_col="w", robust=True, cluster_col="cluster")
    s = c.summary.loc[GROUPS, ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    s.columns = ["HR", "lo", "hi", "p"]
    s["n"] = x[GROUPS].sum()
    s["events"] = x[GROUPS].mul(x.e, axis=0).sum()
    return s.round(4), int(x.e.sum())



# %% [markdown]
# ## Extra flags for Phase 3

# %%
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

d["foot_numb"] = ((d.DIQ100 == 1) & d.DIQ110.isin([2, 3])).astype(int)
d["foot_region"] = (d.get("MPQ120AE").notna() | d.get("MPQ120AF").notna()).astype(int)
d["corroborated"] = ((d.foot_numb == 1) | (d.foot_region == 1)).astype(int)
gout = d["MCQ160N"] if "MCQ160N" in d else pd.Series(np.nan, index=d.index)
alc = d["ALQ130"].where(d["ALQ130"] < 50) if "ALQ130" in d else pd.Series(np.nan, index=d.index)
d["x_f2"] = ((d.get("MCQ160A") == 1) | (gout == 1) | (d.get("MCQ220") == 1) | (alc >= 4) | (d.egfr < 30)).astype(int)
d["abi_min"] = abi
d["claud"] = ((d.DIQ140 == 1) & (d.DIQ150 == 1)).astype(int)
d["cvd_hist"] = d[[c for c in ["MCQ160B", "MCQ160C", "MCQ160E", "MCQ160F"] if c in d]].eq(1).any(axis=1).astype(int)
print("D corroborated:", int(d.loc[d.group == "D_hidden", "corroborated"].sum()), "of", int((d.group == "D_hidden").sum()))
print("F2 exclusion flag rate:", d.x_f2.mean().round(3))


def h1_h2(df, label):
    df = df.copy()
    df["ls"], auc = crossfit_scores(df, FEATS)
    tab, de = group_diffs(df, "ls")
    cx, ev = cox_groups(df)
    res = dict(n=len(df), n_D=int((df.group == "D_hidden").sum()), auroc=auc, D_vs_A=tab.get("D_hidden"),
               E_vs_A=tab.get("E_backpain_ctrl"), D_minus_E=de, HR_D=cx.loc["D_hidden"].to_dict(),
               HR_E=cx.loc["E_backpain_ctrl"].to_dict(), deaths=ev)
    print(f"\n[{label}] n_D={res['n_D']}  D−A={res['D_vs_A']}  D−E={ {k: round(v, 4) for k, v in de.items()} }  HR_D={res['HR_D']}")
    return res




# %% [markdown]
# ## Extra biology for the stronger model (whatever NHANES files exist)

# %%
from sklearn.linear_model import LogisticRegression

EXTRA_FILES = {1999: ["LAB06", "LAB06HM", "LAB16", "LAB13"], 2001: ["L06_B", "L06_2_B", "L16_B", "L13_B"],
               2003: ["L06BMT_C", "L06COT_C", "L06HCY_C", "L06NB_C", "L06TFR_C", "L16_C", "L13_C", "L06VID_C"]}
ex_parts = []
for year, names in EXTRA_FILES.items():
    cyc = None
    for nm in names:
        try:
            x = xpt(year, nm)
            cyc = x if cyc is None else cyc.merge(x, on="SEQN", how="outer", suffixes=("", f"__{nm}"))
            print("loaded", nm, x.shape)
        except Exception:
            pass
    if cyc is not None:
        ex_parts.append(cyc)
ex = pd.concat(ex_parts, ignore_index=True).drop_duplicates("SEQN")
EXV = {"x_b12": ["LBXB12", "LBDB12"], "x_folate": ["LBXFOL", "LBDFOL"], "x_hcy": ["LBXHCY"], "x_mma": ["LBXMMA", "LBXMMASI"],
       "x_lead": ["LBXBPB"], "x_cadmium": ["LBXBCD"], "x_cotinine": ["LBXCOT"], "x_ualb": ["URXUMA", "URXUMS"],
       "x_ucr": ["URXUCR", "URXCRS"]}
for k, v in EXV.items():
    ex[k] = first(ex, v)
ex["x_acr"] = ex.x_ualb / ex.x_ucr
d = d.merge(ex[["SEQN"] + [k for k in EXV if k not in ("x_ualb", "x_ucr")] + ["x_acr"]], on="SEQN", how="left")
d["x_calf"], d["x_leg"], d["x_arm"] = d.get("BMXCALF"), d.get("BMXLEG"), d.get("BMXARML")
NEWF = [c for c in d.columns if c.startswith("x_") and c not in ("x_sciatica", "x_vascular", "x_lowb12")]
NEWF = [c for c in NEWF if d[c].notna().mean() > 0.3]
print("added features (coverage):", {c: round(d[c].notna().mean(), 2) for c in NEWF})
FEATS_BIG = FEATS + NEWF
d["s"] = d.pn.astype(int)

# %% [markdown]
# ## Tune the stronger model (labeled vs unlabeled, 5-fold CV AUROC)

# %%
def cv_auc(df, feats, params):
    y = df.s.values
    oof = np.zeros(len(df))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=2026).split(df, y):
        m = lgb.LGBMClassifier(**params).fit(df[feats].iloc[tr], y[tr])
        oof[te] = m.predict_proba(df[feats].iloc[te])[:, 1]
    return roc_auc_score(y, oof)


base_params = dict(learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, random_state=2026)
grid = []
for nl in [4, 8, 16]:
    for mcs in [30, 80]:
        for ne in [300, 600]:
            for rl in [0, 5]:
                p = dict(base_params, num_leaves=nl, min_child_samples=mcs, n_estimators=ne, reg_lambda=rl)
                grid.append((cv_auc(d, FEATS_BIG, p), p))
grid.sort(key=lambda z: -z[0])
BEST = grid[0][1]
auc_orig = cv_auc(d, FEATS, dict(PARAMS, n_estimators=300))
print(f"AUROC original model {auc_orig:.3f} | best tuned big model {grid[0][0]:.3f} with {BEST}")
OUT["model"] = dict(auroc_original=auc_orig, auroc_big_tuned=grid[0][0], best_params=BEST, n_features=len(FEATS_BIG))

# %% [markdown]
# ## Elkan–Noto pipeline (shared)

# %%
def en_alpha(df, feats, params, seed=2026, wcol="w_raw"):
    y = df.s.values
    oof = np.zeros(len(df))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(df, y):
        m = lgb.LGBMClassifier(**dict(params, random_state=seed)).fit(df[feats].iloc[tr], y[tr])
        oof[te] = m.predict_proba(df[feats].iloc[te])[:, 1]
    lo = np.log(np.clip(oof, 1e-5, 1 - 1e-5) / (1 - np.clip(oof, 1e-5, 1 - 1e-5))).reshape(-1, 1)
    g = LogisticRegression(C=1e6).fit(lo, y).predict_proba(lo)[:, 1]
    c = g[y == 1].mean()
    post = np.clip(((1 - c) / c) * g / (1 - g), 0, 1)
    u = y == 0
    a = np.average(post[u], weights=df[wcol].values[u]) if wcol else post[u].mean()
    return a, c, pd.Series(np.where(u, post, np.nan), index=df.index)


# %% [markdown]
# ## 1. Known-answer test: hide 30% of B among A, can the method recover it?

# %%
ab = d[d.group.isin(["A_reference", "B_painless_PN"])].copy()
known = []
for rep in range(20):
    x = ab.copy()
    pos_idx = x.index[x.group == "B_painless_PN"]
    hide = RNG.choice(pos_idx, int(0.3 * len(pos_idx)), replace=False)
    x["s"] = (x.group == "B_painless_PN").astype(int)
    x.loc[hide, "s"] = 0
    true_alpha = len(hide) / (x.s == 0).sum()
    for label, feats, params in [("original", FEATS, dict(PARAMS, n_estimators=300)), ("big_tuned", FEATS_BIG, BEST)]:
        est, c, _ = en_alpha(x, feats, params, seed=rep, wcol=None)
        known.append(dict(rep=rep, model=label, true=true_alpha, est=est, rel_err=(est - true_alpha) / true_alpha))
known = pd.DataFrame(known)
kt = known.groupby("model").agg(true=("true", "mean"), est_mean=("est", "mean"), rel_err_mean=("rel_err", "mean"),
                                rel_err_abs=("rel_err", lambda s: s.abs().mean()))
print(kt)
OUT["known_answer"] = kt.round(4).to_dict("index")

# %% [markdown]
# ## 2–3. α with the stronger model + subsampling uncertainty (80% PSUs, no replacement)

# %%
d["s"] = d.pn.astype(int)
alpha_big, c_big, post_big = en_alpha(d, FEATS_BIG, BEST)
sub = []
clusters = d.cluster.unique()
for b in range(200):
    pick = RNG.choice(clusters, int(0.8 * len(clusters)), replace=False)
    bs = d[d.cluster.isin(pick)]
    try:
        sub.append(en_alpha(bs, FEATS_BIG, dict(BEST, n_estimators=min(BEST["n_estimators"], 300)), seed=b)[0])
    except Exception:
        pass
lo95, hi95, med = np.percentile(sub, 2.5), np.percentile(sub, 97.5), np.median(sub)
best_model_err = OUT["known_answer"]["big_tuned"]["rel_err_abs"]
credible = (best_model_err < 0.25) and (lo95 <= alpha_big <= hi95) and (abs(alpha_big - med) <= 0.05)
unl = d[d.s == 0]
OUT["alpha_big"] = dict(alpha=float(alpha_big), c=float(c_big), sub_median=float(med), sub_ci=[float(lo95), float(hi95)],
                        us_adults_40plus=float((post_big[d.s == 0] * unl.w_raw).sum()), known_answer_abs_rel_err=float(best_model_err),
                        CREDIBLE=bool(credible))
print(OUT["alpha_big"])

# symptom validation with the stronger model (same as V1)
unl = unl.assign(post=post_big[d.s == 0])
unl["z"] = (unl.post - unl.post.mean()) / unl.post.std()
unl["wn"] = unl.w_raw / unl.w_raw.mean()
v = {}
for outc in ["foot_sx", "back_pain", "hand_only_sx"]:
    xx = unl.dropna(subset=[outc, "race"])
    r = smf.glm(f"{outc} ~ z + age + female + C(race)", xx, family=sm.families.Binomial(), var_weights=xx.wn).fit(
        cov_type="cluster", cov_kwds={"groups": xx.cluster})
    v[outc] = dict(OR=float(np.exp(r.params["z"])), p=float(r.pvalues["z"]))
OUT["V1_big"] = v
print(v)
json.dump(OUT, open("hn_pu2_results.json", "w"), indent=1, default=float)
