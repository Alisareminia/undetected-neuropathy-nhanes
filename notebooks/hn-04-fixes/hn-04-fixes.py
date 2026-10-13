# %% [markdown]
# # HN-04 — Phase 3 caveat fixes F1–F4 (HN_PROTOCOL.md). Data build identical to HN-01/HN-03.

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
# ## F1 — Stricter hidden-group definition (corroborated symptoms)

# %%
f1 = d.copy()
f1.loc[(f1.group == "D_hidden") & (f1.corroborated == 0), "group"] = "other"
OUT["F1_strict_D"] = h1_h2(f1, "F1 strict D")

# %% [markdown]
# ## F2 — Additional exclusions, and all exclusions combined

# %%
OUT["F2_new_exclusions"] = h1_h2(d[d.x_f2 == 0], "F2 new exclusions")
allx = (d.x_f2 == 0) & (d.x_sciatica == 0) & (d.x_vascular == 0) & (d.x_lowb12 == 0)
OUT["F2_all_exclusions"] = h1_h2(d[allx], "F2 all exclusions")

# %% [markdown]
# ## F3 — Mortality with circulation modeled as covariates

# %%
def cox_extra(df, extra, event_mask=None):
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
                     "ed_lt", "ed_hs", "ed_sc", "bmi", "diabetes", "egfr"] + extra
    x = x[cols + ["t", "e", "w", "cluster"]].dropna()
    c = CoxPHFitter(penalizer=1e-6).fit(x, "t", "e", weights_col="w", robust=True, cluster_col="cluster")
    s = c.summary.loc["D_hidden", ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    return dict(HR=float(s.iloc[0]), lo=float(s.iloc[1]), hi=float(s.iloc[2]), p=float(s.iloc[3]),
                n_D=int(x.D_hidden.sum()), deaths=int(x.e.sum()))


cvd = lambda x: x.UCOD.isin([1, 5])
normal_circ = d[(d.abi_min >= 1.0) & (d.abi_min <= 1.4) & (d.claud == 0)]
OUT["F3"] = {
    "allcause_adj_ABI_claud_CVD": cox_extra(d, ["abi_min", "claud", "cvd_hist"]),
    "cvd_adj_ABI_claud_CVD": cox_extra(d, ["abi_min", "claud", "cvd_hist"], cvd),
    "allcause_normal_circulation_only": cox_extra(normal_circ, ["cvd_hist"]),
    "cvd_normal_circulation_only": cox_extra(normal_circ, ["cvd_hist"], cvd),
}
print(json.dumps(OUT["F3"], indent=1))

# %% [markdown]
# ## F4 — Model-agnostic similarity (logistic, elastic net)

# %%
def crossfit_linear(df, model, feats):
    train = df[df.group.isin(["A_reference", "B_painless_PN"])]
    y = (train.group == "B_painless_PN").astype(int).values
    oof = np.zeros(len(train))
    oth = df[~df.group.isin(["A_reference", "B_painless_PN", "other"])]
    op = np.zeros(len(oth))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=2026).split(train, y):
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model).fit(train[feats].iloc[tr], y[tr])
        oof[te] = m.predict_proba(train[feats].iloc[te])[:, 1]
        op += m.predict_proba(oth[feats])[:, 1] / 5
    sc = pd.concat([pd.Series(oof, index=train.index), pd.Series(op, index=oth.index)]).clip(1e-4, 1 - 1e-4)
    return np.log(sc / (1 - sc)), roc_auc_score(y, oof)


F4 = {}
for name, model in [("logistic_L2", LogisticRegression(C=0.1, max_iter=5000)),
                    ("elastic_net", LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.1, max_iter=5000))]:
    x = d.copy()
    x["ls"], auc = crossfit_linear(x, model, FEATS)
    tab, de = group_diffs(x, "ls")
    F4[name] = dict(auroc=auc, D_vs_A=tab["D_hidden"], E_vs_A=tab["E_backpain_ctrl"], D_minus_E=de)
    print(name, json.dumps(F4[name], default=float))
OUT["F4"] = F4

# %%
json.dump(OUT, open("hn_fixes_results.json", "w"), indent=1, default=float)
