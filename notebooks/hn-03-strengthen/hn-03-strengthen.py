# %% [markdown]
# # HN-03 — Strengthening analyses S1–S6 (HN_PROTOCOL.md, Phase 2). Data build identical to HN-01.

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
}


def xpt(year, name):
    r = S.get(BASE.format(year=year, name=name), timeout=240)
    df = pd.read_sas(io.BytesIO(r.content), format="xport")
    df["SEQN"] = df["SEQN"].round().astype(int)
    return df.drop_duplicates("SEQN")


T = {k: pd.concat([xpt(y, n) for y, n in v.items()], ignore_index=True) for k, v in FILES.items()}
d = T["DEMO"][["SEQN", "SDDSRVYR", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "DMDEDUC2", "INDFMPIR", "WTMEC2YR", "WTMEC4YR",
               "SDMVPSU", "SDMVSTRA"]]
for k in ["LEXPN", "DIQ", "MPQ", "BMX", "BPX", "SMQ", "GHB", "BIO", "CBC", "CRP", "LEXAB"]:
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
# ## S1 — Cross-cycle replication (train 1999–2002, test 2003–2004)

# %%
early = d[d.SDDSRVYR.isin([1, 2])]
late = d[d.SDDSRVYR == 3].copy()
trn = early[early.group.isin(["A_reference", "B_painless_PN"])]
m = lgb.LGBMClassifier(**PARAMS).fit(trn[FEATS], (trn.group == "B_painless_PN").astype(int))
p = np.clip(m.predict_proba(late[FEATS])[:, 1], 1e-4, 1 - 1e-4)
late["ls_rep"] = np.log(p / (1 - p))
late["w2"] = late.WTMEC2YR / late.WTMEC2YR.mean()
ab = late[late.group.isin(["A_reference", "B_painless_PN"])]
s1_auc = roc_auc_score((ab.group == "B_painless_PN").astype(int), ab.ls_rep)
s1_tab, s1_de = group_diffs(late, "ls_rep", "w2")
cox_early, ev_e = cox_groups(early)
cox_late, ev_l = cox_groups(late.assign(w=late.w2))
OUT["S1"] = dict(test_auroc_2003_04=s1_auc, group_diffs_2003_04=s1_tab, D_minus_E_2003_04=s1_de,
                 cox_1999_2002=cox_early.to_dict("index"), cox_2003_2004=cox_late.to_dict("index"),
                 deaths=dict(early=ev_e, late=ev_l))
print(f"S1 external AUROC (2003–04) = {s1_auc:.3f}")
print(pd.DataFrame(s1_tab).T, "\nD−E:", s1_de)
print("Cox 1999–2002:\n", cox_early, "\nCox 2003–2004:\n", cox_late)

# %% [markdown]
# ## S2 — Survey-weighted prevalence and population counts

# %%
pop = d[d.group != "other"].copy()
prev = pop.groupby("group").w_raw.sum() / pop.w_raw.sum()
counts = pop.groupby("group").w_raw.sum()
clusters = pop.cluster.unique()
boot = []
for _ in range(500):
    pick = RNG.choice(clusters, len(clusters), replace=True)
    bs = pd.concat([pop[pop.cluster == c] for c in pick])
    boot.append(bs.groupby("group").w_raw.sum() / bs.w_raw.sum())
boot = pd.DataFrame(boot)
s2 = pd.DataFrame({"prevalence": prev, "lo": boot.quantile(0.025), "hi": boot.quantile(0.975), "us_adults_40plus": counts.round(-3)})
OUT["S2"] = s2.round(4).to_dict("index")
print(s2)

# %% [markdown]
# ## S3 — Excluding look-alike conditions (H1 + H2 re-run)

# %%
S3 = {}
for label, mask in [("no_sciatica", d.x_sciatica == 0), ("no_vascular", d.x_vascular == 0),
                    ("no_lowB12", d.x_lowb12 == 0),
                    ("all_excluded", (d.x_sciatica == 0) & (d.x_vascular == 0) & (d.x_lowb12 == 0))]:
    sub = d[mask].copy()
    sub["ls"], auc = crossfit_scores(sub, FEATS)
    tab, de = group_diffs(sub, "ls")
    cx, ev = cox_groups(sub)
    S3[label] = dict(n=len(sub), n_D=int((sub.group == "D_hidden").sum()), auroc=auc, D_vs_A=tab["D_hidden"],
                     E_vs_A=tab["E_backpain_ctrl"], D_minus_E=de, HR_D=cx.loc["D_hidden"].to_dict(),
                     HR_E=cx.loc["E_backpain_ctrl"].to_dict(), deaths=ev)
    print(label, json.dumps(S3[label], default=float)[:600])
OUT["S3"] = S3

# %% [markdown]
# ## S4 — Cause-specific mortality

# %%
S4 = {}
for label, codes in [("cardiovascular", [1, 5]), ("cancer", [2]), ("other_causes", [3, 4, 6, 7, 8, 9, 10])]:
    cx, ev = cox_groups(d, event_mask=lambda x, codes=codes: x.UCOD.isin(codes))
    S4[label] = dict(deaths=ev, table=cx.to_dict("index"))
    print(label, ev, "\n", cx)
OUT["S4"] = S4

# %% [markdown]
# ## S5 — What drives the hidden group's score (SHAP, D minus A)

# %%
trn = d[d.group.isin(["A_reference", "B_painless_PN"])]
full = lgb.LGBMClassifier(**PARAMS).fit(trn[FEATS], (trn.group == "B_painless_PN").astype(int))
contrib = lambda g: pd.DataFrame(full.predict(d.loc[d.group == g, FEATS], pred_contrib=True)[:, :-1], columns=FEATS).mean()
s5 = (contrib("D_hidden") - contrib("A_reference")).sort_values(ascending=False)
s5_e = (contrib("E_backpain_ctrl") - contrib("A_reference")).reindex(s5.index)
OUT["S5"] = dict(D_minus_A=s5.head(12).round(4).to_dict(), E_minus_A_same_features=s5_e.head(12).round(4).to_dict())
print(pd.DataFrame({"D−A": s5, "E−A": s5_e}).head(15).round(4))

# %% [markdown]
# ## S6 — E-value for the hidden-group mortality HR (common outcome conversion)

# %%
cx_main, _ = cox_groups(d)


def evalue(hr):
    rr = (1 - 0.5 ** np.sqrt(hr)) / (1 - 0.5 ** np.sqrt(1 / hr))
    return rr + np.sqrt(rr * (rr - 1)) if rr > 1 else 1.0


hr, lo = cx_main.loc["D_hidden", "HR"], cx_main.loc["D_hidden", "lo"]
OUT["S6"] = dict(HR=float(hr), lo=float(lo), E_value_point=float(evalue(hr)), E_value_CI=float(evalue(lo)))
print(OUT["S6"])

# %%
json.dump(OUT, open("hn_strengthen_results.json", "w"), indent=1, default=float)
