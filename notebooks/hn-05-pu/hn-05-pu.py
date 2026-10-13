# %% [markdown]
# # HN-05 — H6 positive-unlabeled learning (HN_PROTOCOL.md). Data build identical to HN-01/HN-03/HN-04.

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
# ## H6 — Positive-unlabeled learning (Elkan–Noto), symptoms never used as input

# %%
from sklearn.linear_model import LogisticRegression

d["foot_numb"] = ((d.DIQ100 == 1) & d.DIQ110.isin([2, 3])).astype(int)
d["abi_min"] = abi
d["claud"] = ((d.DIQ140 == 1) & (d.DIQ150 == 1)).astype(int)
d["cvd_hist"] = d[[c for c in ["MCQ160B", "MCQ160C", "MCQ160E", "MCQ160F"] if c in d]].eq(1).any(axis=1).astype(int) \
    if any(c in d for c in ["MCQ160B", "MCQ160C"]) else 0
d["s"] = d.pn.astype(int)
PU_PARAMS = dict(PARAMS, n_estimators=300)


def pu_fit(df, params=PU_PARAMS, seed=2026):
    """Cross-fitted g(x)=P(s=1|x), Platt-calibrated; returns posterior for unlabeled rows and c."""
    y = df.s.values
    oof = np.zeros(len(df))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(df, y):
        m = lgb.LGBMClassifier(**dict(params, random_state=seed)).fit(df[FEATS].iloc[tr], y[tr])
        oof[te] = m.predict_proba(df[FEATS].iloc[te])[:, 1]
    lo = np.log(np.clip(oof, 1e-5, 1 - 1e-5) / (1 - np.clip(oof, 1e-5, 1 - 1e-5))).reshape(-1, 1)
    g = LogisticRegression(C=1e6).fit(lo, y).predict_proba(lo)[:, 1]
    c = g[y == 1].mean()
    post = np.clip(((1 - c) / c) * g / (1 - g), 0, 1)
    return pd.Series(np.where(y == 0, post, np.nan), index=df.index), c, roc_auc_score(y, oof)


d["post"], c_hat, auc_s = pu_fit(d)
unl = d[d.s == 0].copy()
alpha = np.average(unl.post, weights=unl.w_raw)
count = float((unl.post * unl.w_raw).sum())
print(f"AUROC(labeled vs unlabeled)={auc_s:.3f}  c={c_hat:.3f}  alpha={alpha:.3f}  ~{count/1e6:.1f} million US adults ≥40")

# cluster bootstrap of alpha (refit each time)
boots = []
clusters = d.cluster.unique()
for b in range(200):
    pick = RNG.choice(clusters, len(clusters), replace=True)
    bs = pd.concat([d[d.cluster == cl] for cl in pick], ignore_index=True)
    try:
        p_, c_, _ = pu_fit(bs, dict(PU_PARAMS, n_estimators=200), seed=b)
        u = bs[bs.s == 0]
        boots.append(np.average(p_[bs.s == 0], weights=u.w_raw))
    except Exception:
        pass
OUT["H6_alpha"] = dict(c=float(c_hat), auroc_labeled_vs_unlabeled=float(auc_s), alpha=float(alpha),
                       alpha_ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                       us_adults_40plus_missed=count, n_boot=len(boots),
                       alpha_by_diabetes={int(k): float(np.average(g.post, weights=g.w_raw)) for k, g in unl.groupby("diabetes")})
print(OUT["H6_alpha"])

# %% [markdown]
# ## V1 — does the posterior predict foot symptoms (and not back pain)?

# %%
def v1(u, label):
    u = u.dropna(subset=["post", "race"]).copy()
    u["z"] = (u.post - u.post.mean()) / u.post.std()
    u["wn"] = u.w_raw / u.w_raw.mean()
    res = {}
    for outc in ["foot_sx", "back_pain", "hand_only_sx", "foot_numb"]:
        x = u.dropna(subset=[outc])
        r = smf.glm(f"{outc} ~ z + age + female + C(race)", x, family=sm.families.Binomial(), var_weights=x.wn).fit(
            cov_type="cluster", cov_kwds={"groups": x.cluster})
        res[outc] = dict(OR=float(np.exp(r.params["z"])), lo=float(np.exp(r.conf_int().loc["z", 0])),
                         hi=float(np.exp(r.conf_int().loc["z", 1])), p=float(r.pvalues["z"]))
    sup = res["foot_sx"]["OR"] > 1 and res["foot_sx"]["p"] < 0.05 and res["foot_sx"]["OR"] > res["back_pain"]["OR"]
    print(f"[{label}]", json.dumps(res, indent=0), "SUPPORTED" if sup else "NOT SUPPORTED")
    return dict(res, supported=bool(sup))


OUT["V1"] = v1(unl, "V1 all cycles")

# %% [markdown]
# ## V2 — does the posterior predict mortality among unlabeled adults?

# %%
def v2(u, extra):
    x = u[(u.ELIGSTAT == 1) & (u.PERMTH_EXAM > 0)].copy()
    x["t"], x["e"] = x.PERMTH_EXAM / 12, (x.MORTSTAT == 1).astype(int)
    x["z"] = (x.post - x.post.mean()) / x.post.std()
    for lv in ["NHBlack", "MexAm", "OthHisp", "Other"]:
        x[f"r_{lv}"] = (x.race == lv).astype(int)
    cols = ["z", "age", "female", "r_NHBlack", "r_MexAm", "r_OthHisp", "r_Other"] + extra
    x = x[cols + ["t", "e", "w", "cluster"]].dropna()
    cph = CoxPHFitter(penalizer=1e-6).fit(x, "t", "e", weights_col="w", robust=True, cluster_col="cluster")
    s = cph.summary.loc["z"]
    return dict(HR=float(s["exp(coef)"]), lo=float(s["exp(coef) lower 95%"]), hi=float(s["exp(coef) upper 95%"]),
                p=float(s["p"]), n=len(x), deaths=int(x.e.sum()))


OUT["V2"] = {"age_sex_race": v2(unl, []), "plus_circulation_CVD": v2(unl, ["abi_min", "claud", "cvd_hist"])}
print(json.dumps(OUT["V2"], indent=1))

# %% [markdown]
# ## V3 — replication: train on 1999–2002, test V1 on 2003–2004

# %%
early, late = d[d.SDDSRVYR.isin([1, 2])].copy(), d[d.SDDSRVYR == 3].copy()
ye = early.s.values
oof = np.zeros(len(early))
models = []
for tr, te in StratifiedKFold(5, shuffle=True, random_state=2026).split(early, ye):
    m = lgb.LGBMClassifier(**PU_PARAMS).fit(early[FEATS].iloc[tr], ye[tr])
    oof[te] = m.predict_proba(early[FEATS].iloc[te])[:, 1]
    models.append(m)
lo_e = np.log(np.clip(oof, 1e-5, 1 - 1e-5) / (1 - np.clip(oof, 1e-5, 1 - 1e-5))).reshape(-1, 1)
cal = LogisticRegression(C=1e6).fit(lo_e, ye)
c_e = cal.predict_proba(lo_e)[:, 1][ye == 1].mean()
gl = np.mean([m.predict_proba(late[FEATS])[:, 1] for m in models], axis=0)
gl = cal.predict_proba(np.log(np.clip(gl, 1e-5, 1 - 1e-5) / (1 - np.clip(gl, 1e-5, 1 - 1e-5))).reshape(-1, 1))[:, 1]
late["post"] = np.where(late.s == 0, np.clip(((1 - c_e) / c_e) * gl / (1 - gl), 0, 1), np.nan)
OUT["V3"] = v1(late[late.s == 0], "V3 replication 2003–04")
OUT["V3"]["alpha_2003_04"] = float(np.average(late.loc[late.s == 0, "post"], weights=late.loc[late.s == 0, "w_raw"]))

# %% [markdown]
# ## V4 — silent fraction of estimated missed cases

# %%
tot = (unl.post * unl.w_raw).sum()
OUT["V4"] = dict(silent_fraction=float((unl.post * unl.w_raw)[unl.foot_sx == 0].sum() / tot),
                 symptomatic_fraction=float((unl.post * unl.w_raw)[unl.foot_sx == 1].sum() / tot),
                 mean_post_by_group=unl.groupby("group").post.mean().round(4).to_dict())
print(OUT["V4"])
json.dump(OUT, open("hn_pu_results.json", "w"), indent=1, default=float)
