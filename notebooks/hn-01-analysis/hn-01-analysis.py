# %% [markdown]
# # HN-01 — Hidden neuropathy: ML similarity score, mortality, painful vs painless, epigenetic aging (HN_PROTOCOL.md)

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
}


def xpt(year, name):
    r = S.get(BASE.format(year=year, name=name), timeout=240)
    df = pd.read_sas(io.BytesIO(r.content), format="xport")
    df["SEQN"] = df["SEQN"].round().astype(int)
    return df.drop_duplicates("SEQN")


T = {k: pd.concat([xpt(y, n) for y, n in v.items()], ignore_index=True) for k, v in FILES.items()}
d = T["DEMO"][["SEQN", "SDDSRVYR", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "DMDEDUC2", "INDFMPIR", "WTMEC2YR", "WTMEC4YR",
               "SDMVPSU", "SDMVSTRA"]]
for k in ["LEXPN", "DIQ", "MPQ", "BMX", "BPX", "SMQ", "GHB", "BIO", "CBC", "CRP"]:
    d = d.merge(T[k], on="SEQN", how="left", suffixes=("", f"__{k}"))
d = d.round(10)

MORT = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_{a}_{b}_MORT_2019_PUBLIC.dat"
mort = pd.concat([pd.read_fwf(io.StringIO(S.get(MORT.format(a=y, b=y + 1), timeout=120).text),
                              colspecs=[(0, 6), (14, 15), (15, 16), (45, 48)],
                              names=["SEQN", "ELIGSTAT", "MORTSTAT", "PERMTH_EXAM"], na_values=["."]) for y in (1999, 2001, 2003)])
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
# ## H1 — ML neuropathy-similarity score (train B vs A; score C, D, E, F)

# %%
def similarity(feats, label):
    train = d[d.group.isin(["A_reference", "B_painless_PN"])].copy()
    y = (train.group == "B_painless_PN").astype(int).values
    oof = np.zeros(len(train))
    others = d[~d.group.isin(["A_reference", "B_painless_PN", "other"])]
    oth_pred = np.zeros(len(others))
    params = dict(n_estimators=400, learning_rate=0.03, num_leaves=8, min_child_samples=50, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, verbose=-1, random_state=2026)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=2026).split(train, y):
        m = lgb.LGBMClassifier(**params).fit(train[feats].iloc[tr], y[tr])
        oof[te] = m.predict_proba(train[feats].iloc[te])[:, 1]
        oth_pred += m.predict_proba(others[feats])[:, 1] / 5
    sc = pd.concat([pd.Series(oof, index=train.index), pd.Series(oth_pred, index=others.index)])
    d[f"score_{label}"] = sc
    d[f"lscore_{label}"] = np.log(sc.clip(1e-4, 1 - 1e-4) / (1 - sc.clip(1e-4, 1 - 1e-4)))
    auc = roc_auc_score(y, oof)
    x = d[d.group != "other"].dropna(subset=[f"lscore_{label}", "race"])
    r = smf.wls(f"lscore_{label} ~ C(group, Treatment('A_reference')) + age + female + C(race)", x, weights=x.w).fit(
        cov_type="cluster", cov_kwds={"groups": x.cluster})
    tab = pd.DataFrame({"diff_vs_A": r.params, "lo": r.conf_int()[0], "hi": r.conf_int()[1], "p": r.pvalues})
    tab = tab[tab.index.str.startswith("C(group")]
    tab.index = tab.index.str.extract(r"T\.(\w+)\]")[0].values
    # D vs E contrast
    cvec = pd.Series(0.0, index=r.params.index)
    cvec[[i for i in cvec.index if "D_hidden" in i]] = 1
    cvec[[i for i in cvec.index if "E_backpain_ctrl" in i]] = -1
    t = r.t_test(cvec.values)
    de = dict(diff=float(t.effect[0]), lo=float(t.conf_int()[0][0]), hi=float(t.conf_int()[0][1]), p=float(t.pvalue))
    print(f"\n[{label}] OOF AUROC B vs A = {auc:.3f}\n", tab.round(4), "\nD − E:", {k: round(v, 4) for k, v in de.items()})
    return dict(auroc_B_vs_A=auc, group_diffs=tab.round(5).to_dict("index"), D_minus_E=de)


OUT["H1_full"] = similarity(FEATS, "full")
OUT["H1_no_glycemia"] = similarity(FEATS_NO_GLY, "nogly")
d_save = d.copy()
d = d[d.diabetes == 0].copy()
OUT["H1_no_diabetes"] = similarity(FEATS_NO_GLY, "nodm")
d = d_save

# %% [markdown]
# ## H2 — Mortality by group (weighted Cox, cluster-robust)

# %%
def cox_groups(df, label):
    x = df[(df.ELIGSTAT == 1) & (df.PERMTH_EXAM > 0) & (df.group != "other")].copy()
    x["t"], x["e"] = x.PERMTH_EXAM / 12, (x.MORTSTAT == 1).astype(int)
    for g in ["B_painless_PN", "C_painful_PN", "D_hidden", "E_backpain_ctrl", "F_hands_only"]:
        x[g] = (x.group == g).astype(int)
    for lv in ["NHBlack", "MexAm", "OthHisp", "Other"]:
        x[f"r_{lv}"] = (x.race == lv).astype(int)
    x["sm_cur"], x["sm_for"] = (x.smoke == "current").astype(int), (x.smoke == "former").astype(int)
    x["ed_lt"], x["ed_hs"], x["ed_sc"] = [(x.educ == v).astype(int) for v in ["<HS", "HS", "SomeCol"]]
    cols = ["B_painless_PN", "C_painful_PN", "D_hidden", "E_backpain_ctrl", "F_hands_only", "age", "female", "r_NHBlack",
            "r_MexAm", "r_OthHisp", "r_Other", "pir", "sm_cur", "sm_for", "ed_lt", "ed_hs", "ed_sc", "bmi", "diabetes", "egfr"]
    if df.diabetes.nunique() == 1:
        cols.remove("diabetes")
    x = x[cols + ["t", "e", "w", "cluster"]].dropna()
    c = CoxPHFitter().fit(x, "t", "e", weights_col="w", robust=True, cluster_col="cluster")
    s = c.summary.loc[["B_painless_PN", "C_painful_PN", "D_hidden", "E_backpain_ctrl", "F_hands_only"],
                      ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    s.columns = ["HR", "lo", "hi", "p"]
    n_by = x[["B_painless_PN", "C_painful_PN", "D_hidden", "E_backpain_ctrl", "F_hands_only"]].sum()
    deaths_by = x[["B_painless_PN", "C_painful_PN", "D_hidden", "E_backpain_ctrl", "F_hands_only"]].mul(x.e, axis=0).sum()
    s["n"], s["deaths"] = n_by, deaths_by
    print(f"\n[{label}] n={len(x)}, deaths={int(x.e.sum())}\n", s.round(4))
    return s.round(5).to_dict("index")


OUT["H2_main"] = cox_groups(d, "all")
OUT["H2_excl_2y"] = cox_groups(d[~((d.MORTSTAT == 1) & (d.PERMTH_EXAM < 24))], "exclude deaths <2y")
OUT["H2_no_diabetes"] = cox_groups(d[d.diabetes == 0], "no diabetes")

# %% [markdown]
# ## H3 — Painful vs painless neuropathy (ML + adjusted logistic)

# %%
pn = d[d.group.isin(["B_painless_PN", "C_painful_PN"])].copy()
y = (pn.group == "C_painful_PN").astype(int).values
oof = np.zeros(len(pn))
imp = np.zeros(len(FEATS))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=2026).split(pn, y):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=4, min_child_samples=40, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, verbose=-1, random_state=2026).fit(pn[FEATS].iloc[tr], y[tr])
    oof[te] = m.predict_proba(pn[FEATS].iloc[te])[:, 1]
    imp += np.abs(m.predict(pn[FEATS].iloc[te], pred_contrib=True)[:, :-1]).mean(0) / 5
auc = roc_auc_score(y, oof)
boots = [roc_auc_score(y[b], oof[b]) for b in (RNG.integers(0, len(y), len(y)) for _ in range(1000)) if y[b].std() > 0]
perm = [roc_auc_score(RNG.permutation(y), oof) for _ in range(1000)]
shap_top = pd.Series(imp, index=FEATS).sort_values(ascending=False).head(12)
pn["painful"] = y
pnc = pn.dropna(subset=["diabetes", "hba1c", "bmi", "egfr", "crp", "smoke", "pir", "race"])
lr = smf.glm("painful ~ diabetes + hba1c + age + female + C(race) + bmi + egfr + crp + C(smoke) + pir", pnc, family=sm.families.Binomial(),
             var_weights=pnc.w).fit(cov_type="cluster", cov_kwds={"groups": pnc.cluster})
OUT["H3"] = dict(n_painless=int((y == 0).sum()), n_painful=int(y.sum()), auroc=auc, auroc_ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                 perm_p=float((np.array(perm) >= auc).mean()), shap_top=shap_top.round(4).to_dict(),
                 logistic_OR={k: [round(float(np.exp(v)), 3), round(float(lr.pvalues[k]), 4)] for k, v in lr.params.items() if k != "Intercept"})
print(json.dumps(OUT["H3"], indent=1))

# %% [markdown]
# ## H4 — Epigenetic aging (GrimAge2 acceleration): D vs A, C vs B

# %%
e = d[d.GRIMAGE2MORT.notna() & (d.WTDN4YR > 0)].copy()
e["eaa"] = sm.OLS(e.GRIMAGE2MORT, sm.add_constant(e.age)).fit().resid
e["eaa"] = (e.eaa - e.eaa.mean()) / e.eaa.std()
e["wd"] = e.WTDN4YR / e.WTDN4YR.mean()
h4 = {}
for lab, g1, g0 in [("D_vs_A", "D_hidden", "A_reference"), ("C_vs_B", "C_painful_PN", "B_painless_PN")]:
    x = e[e.group.isin([g1, g0])].copy()
    x["G"] = (x.group == g1).astype(int)
    x = x.dropna(subset=["race", "pir", "smoke", "bmi"])
    r = smf.wls("eaa ~ G + female + C(race) + pir + C(smoke) + bmi + diabetes", x, weights=x.wd).fit(cov_type="cluster", cov_kwds={"groups": x.cluster})
    h4[lab] = dict(n1=int(x.G.sum()), n0=int((x.G == 0).sum()), diff_SD=float(r.params["G"]), lo=float(r.conf_int().loc["G", 0]),
                   hi=float(r.conf_int().loc["G", 1]), p=float(r.pvalues["G"]))
OUT["H4"] = h4
print(json.dumps(h4, indent=1))

# %%
json.dump(OUT, open("hn_results.json", "w"), indent=1, default=float)
d[["SEQN", "group", "score_full", "score_nogly"]].to_parquet("hn_scores.parquet", index=False)
