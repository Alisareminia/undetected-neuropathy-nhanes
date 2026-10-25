# %% [markdown]
# # HN-11 — Revision data build and model scores
# Builds the analysis file for the co-author revision: STROBE flow counts, the six groups, covariates, mortality,
# exclusion flags (old and new: urate/gout drugs, neurotoxic drugs, hepatitis C, thyroid disease), eight unrelated
# self-reported conditions for the cosine reference range, and transfer-model scores for every sensitivity analysis.
# Design-based inference is done in HN-12 (R `survey`); the pipeline bootstrap is in HN-13.

# %%
import io
import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
S = requests.Session()
S.headers["User-Agent"] = "neuropath-research/0.1 (academic project)"
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"
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
    "OSQ": {1999: "OSQ", 2001: "OSQ_B", 2003: "OSQ_C"}, "HCV": {1999: "LAB02", 2001: "L02_B", 2003: "L02_C"},
    "TSH": {1999: "LAB18T4", 2001: "L40T4_B"},
}


def get_xpt(year, name):
    r = S.get(BASE.format(year=year, name=name), timeout=240)
    r.raise_for_status()
    df = pd.read_sas(io.BytesIO(r.content), format="xport")
    df["SEQN"] = df["SEQN"].round().astype(int)
    return df


def xpt(year, name):
    return get_xpt(year, name).drop_duplicates("SEQN")


T = {k: pd.concat([xpt(y, n) for y, n in v.items()], ignore_index=True) for k, v in FILES.items()}
demo = T["DEMO"][["SEQN", "SDDSRVYR", "RIDSTATR", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "DMDEDUC2", "INDFMPIR", "WTMEC2YR",
                  "WTMEC4YR", "SDMVPSU", "SDMVSTRA"]].copy()
d = demo.copy()
for k in ["LEXPN", "DIQ", "MPQ", "BMX", "BPX", "SMQ", "GHB", "BIO", "CBC", "CRP", "LEXAB", "MCQ", "ALQ", "OSQ", "HCV", "TSH"]:
    d = d.merge(T[k], on="SEQN", how="left", suffixes=("", f"__{k}"))
d = d.round(10)

MORT = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_{a}_{b}_MORT_2019_PUBLIC.dat"
mort = pd.concat([pd.read_fwf(io.StringIO(S.get(MORT.format(a=y, b=y + 1), timeout=120).text),
                              colspecs=[(0, 6), (14, 15), (15, 16), (16, 19), (45, 48)],
                              names=["SEQN", "ELIGSTAT", "MORTSTAT", "UCOD", "PERMTH_EXAM"], na_values=["."]) for y in (1999, 2001, 2003)])
d = d.merge(mort, on="SEQN", how="left")

# prescription medications (several rows per person): generic names, upper case
rx = []
for y, n, col in [(1999, "RXQ_RX", "RXD240B"), (2001, "RXQ_RX_B", "RXD240B"), (2003, "RXQ_RX_C", "RXDDRUG")]:
    x = get_xpt(y, n)
    v = x[col]
    v = v.str.decode("latin-1") if v.dtype == object and len(v.dropna()) and isinstance(v.dropna().iloc[0], bytes) else v
    rx.append(pd.DataFrame({"SEQN": x.SEQN, "drug": v.fillna("").astype(str).str.upper()}))
rx = pd.concat(rx, ignore_index=True)
print("prescription rows:", len(rx), " people:", rx.SEQN.nunique())

# vitamin B12 (as in HN-03/04)
b12 = []
for year, cands in {1999: ["LAB06"], 2001: ["L06_B"], 2003: ["L06BMT_C", "L06_C", "L06MH_C"]}.items():
    for nm in cands:
        try:
            x = xpt(year, nm)
            if any(c in x for c in ["LBXB12", "LBDB12"]):
                b12.append(x[["SEQN"] + [c for c in ["LBXB12", "LBDB12"] if c in x]])
                break
        except Exception:
            pass


def first(df, names):
    s = pd.Series(np.nan, index=df.index)
    for n in names:
        if n in df:
            s = s.fillna(df[n])
    return s


bb = pd.concat(b12, ignore_index=True).drop_duplicates("SEQN")
bb["b12"] = first(bb, ["LBXB12", "LBDB12"])
d = d.merge(bb[["SEQN", "b12"]], on="SEQN", how="left")

# %% [markdown]
# ## STROBE flow

# %%
L_, R_ = d.LEALPN.where(d.LEALPN >= 0), d.LEARPN.where(d.LEARPN >= 0)
d["pn"] = np.where((L_ >= 1) | (R_ >= 1), 1, np.where((L_ == 0) & (R_ == 0), 0, np.nan))
age40 = d.RIDAGEYR >= 40
exam = d.RIDSTATR == 2
mono = d.pn.notna()
sx = d.DIQ120.isin([1, 2])
flow = dict(
    nhanes_1999_2004=int(len(d)),
    aged_40_plus=int(age40.sum()),
    aged_40_not_examined=int((age40 & ~exam).sum()),
    examined_40_plus=int((age40 & exam).sum()),
    no_valid_monofilament=int((age40 & exam & ~mono).sum()),
    valid_monofilament=int((age40 & exam & mono).sum()),
    no_symptom_answer=int((age40 & exam & mono & ~sx).sum()),
    included=int((age40 & exam & mono & sx).sum()),
)
print(json.dumps(flow, indent=1))
OUT["flow"] = flow
assert flow["included"] == 7894, flow["included"]

# %% [markdown]
# ## Groups and variables (identical to HN-01/03/04/07)

# %%
d["foot_sx"] = np.where(d.DIQ120 == 1, d.DIQ130.isin([2, 3]).astype(float), np.where(d.DIQ120 == 2, 0.0, np.nan))
d["hand_only_sx"] = ((d.DIQ120 == 1) & (d.DIQ130 == 1)).astype(float)
d["back_pain"] = (d.MPQ070 == 1).astype(float)  # missing MPQ070 counts as no back pain
# pain sites, for the multi-site pain checks requested in the second review (DIQ130: 1 hands, 2 feet, 3 both)
d["sx_feet_only"] = ((d.DIQ120 == 1) & (d.DIQ130 == 2)).astype(int)
d["sx_hands_and_feet"] = ((d.DIQ120 == 1) & (d.DIQ130 == 3)).astype(int)
d["mpq070_missing"] = (~d.MPQ070.isin([1, 2])).astype(int)
d["n_pain_sites"] = d.foot_sx.fillna(0) + ((d.DIQ120 == 1) & d.DIQ130.isin([1, 3])) + d.back_pain
d["cohort"] = (age40 & exam & mono & sx).astype(int)
d["group"] = np.select(
    [(d.pn == 1) & (d.foot_sx == 1), (d.pn == 1) & (d.foot_sx == 0), (d.pn == 0) & (d.foot_sx == 1),
     (d.pn == 0) & (d.foot_sx == 0) & (d.hand_only_sx == 1),
     (d.pn == 0) & (d.foot_sx == 0) & (d.hand_only_sx == 0) & (d.back_pain == 1),
     (d.pn == 0) & (d.foot_sx == 0) & (d.hand_only_sx == 0) & (d.back_pain == 0)],
    ["C_painful_PN", "B_painless_PN", "D_hidden", "F_hands_only", "E_backpain_ctrl", "A_reference"], default="other")
d.loc[d.cohort == 0, "group"] = "not_in_cohort"
print(d.loc[d.cohort == 1, "group"].value_counts())

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
FP = [f for f in FEATS if f not in ("age", "female", "race_nhb", "race_mex", "race_oth_hisp", "race_other", "dm_self")]
# combined 6-year MEC weight (NCHS guidance)
d["w6"] = np.where(d.SDDSRVYR.isin([1, 2]), d.WTMEC4YR * 2 / 3, d.WTMEC2YR / 3)

# missingness of model inputs (for the Methods: LightGBM handles missing values natively; no imputation)
co = d[d.cohort == 1]
OUT["missing_pct_inputs"] = (co[FEATS].isna().mean() * 100).round(2).to_dict()
OUT["any_missing_input_pct"] = float(co[FEATS].isna().any(axis=1).mean() * 100)
print("participants with >=1 missing model input: %.1f%%" % OUT["any_missing_input_pct"])

# %% [markdown]
# ## Exclusion flags: earlier ones (HN-03/04) and new ones requested in review

# %%
abi = d[[c for c in ["LEXLABPI", "LEXRABPI"] if c in d]].min(axis=1)
d["abi_min"] = abi
d["abi_max"] = d[[c for c in ["LEXLABPI", "LEXRABPI"] if c in d]].max(axis=1)
d["abi_n_legs"] = d[[c for c in ["LEXLABPI", "LEXRABPI"] if c in d]].notna().sum(axis=1)
d["claud"] = ((d.DIQ140 == 1) & (d.DIQ150 == 1)).astype(int)
d["cvd_hist"] = d[[c for c in ["MCQ160B", "MCQ160C", "MCQ160E", "MCQ160F"] if c in d]].eq(1).any(axis=1).astype(int)
d["x_sciatica"] = (d.MPQ080 == 1).astype(int)
d["x_vascular"] = ((abi < 0.9) | (d.claud == 1)).astype(int)
d["x_lowb12"] = (d.b12 < 200).astype(int)
alc = d["ALQ130"].where(d["ALQ130"] < 50)
d["x_comorb"] = ((d.MCQ160A == 1) | (d.MCQ220 == 1) | (alc >= 4) | (d.egfr < 30)).astype(int)
d["foot_numb"] = ((d.DIQ100 == 1) & d.DIQ110.isin([2, 3])).astype(int)
d["foot_region"] = (d.get("MPQ120AE").notna() | d.get("MPQ120AF").notna()).astype(int)
d["corroborated"] = ((d.foot_numb == 1) | (d.foot_region == 1)).astype(int)


def on_drug(names):
    pat = "|".join(names)
    ids = rx.loc[rx.drug.str.contains(pat, regex=True), "SEQN"].unique()
    return d.SEQN.isin(ids).astype(int)


GOUT_RX = ["ALLOPURINOL", "COLCHICINE", "PROBENECID", "SULFINPYRAZONE"]  # febuxostat was approved only in 2009
NEURO_RX = [  # established neurotoxic medications
    "VINCRISTINE", "VINBLASTINE", "VINORELBINE", "CISPLATIN", "CARBOPLATIN", "OXALIPLATIN", "PACLITAXEL", "DOCETAXEL",
    "BORTEZOMIB", "THALIDOMIDE", "LENALIDOMIDE", "CAPECITABINE", "FLUOROURACIL", "CYTARABINE",
    "METRONIDAZOLE", "NITROFURANTOIN", "ISONIAZID", "AMIODARONE", "DAPSONE", "COLCHICINE", "PHENYTOIN", "DISULFIRAM",
    "STAVUDINE", "DIDANOSINE", "ZALCITABINE", "LINEZOLID", "HYDRALAZINE", "LEFLUNOMIDE",
    "CIPROFLOXACIN", "LEVOFLOXACIN", "MOXIFLOXACIN", "OFLOXACIN", "GATIFLOXACIN", "NORFLOXACIN", "LOMEFLOXACIN",
    "TROVAFLOXACIN", "GEMIFLOXACIN"]
THY_RX = ["LEVOTHYROXINE", "LIOTHYRONINE", "LIOTRIX", "THYROID", "METHIMAZOLE", "PROPYLTHIOURACIL"]

d["rx_gout"] = on_drug(GOUT_RX)
d["rx_neuro"] = on_drug(NEURO_RX)
d["rx_thyroid"] = on_drug(THY_RX)
d["urate_high"] = (d.uric > 6.8).astype(int)  # above monosodium urate saturation
d["x_urate"] = ((d.urate_high == 1) | (d.rx_gout == 1)).astype(int)
d["x_neurotox"] = d.rx_neuro
d["hcv_pos"] = (d.LBDHCV == 1).astype(int)
d["x_hcv"] = d.hcv_pos
thy_self = (d.get("MCQ160H") == 1) | (d.get("MCQ160I") == 1) | (d.get("MCD160M") == 1) | (d.get("MCQ160M") == 1)
d["thyroid_self"] = thy_self.astype(int)
d["tsh"] = d.LBXTSH
d["tsh_abn"] = ((d.tsh < 0.4) | (d.tsh > 4.5)).astype(int)
d["x_thyroid"] = ((d.thyroid_self == 1) | (d.rx_thyroid == 1) | (d.tsh_abn == 1)).astype(int)
d["x_new"] = ((d.x_urate == 1) | (d.x_neurotox == 1) | (d.x_hcv == 1) | (d.x_thyroid == 1)).astype(int)
d["x_old_all"] = ((d.x_sciatica == 1) | (d.x_vascular == 1) | (d.x_lowb12 == 1) | (d.x_comorb == 1)).astype(int)
d["x_everything"] = ((d.x_new == 1) | (d.x_old_all == 1)).astype(int)
co = d[d.cohort == 1]
avail = dict(hcv_measured=float(co.LBDHCV.isin([1, 2]).mean()), tsh_measured=float(co.tsh.notna().mean()),
             tsh_measured_1999_2002=float(co.loc[co.SDDSRVYR.isin([1, 2]), "tsh"].notna().mean()),
             urate_measured=float(co.uric.notna().mean()))
OUT["availability"] = avail
flags = ["x_sciatica", "x_vascular", "x_lowb12", "x_comorb", "x_urate", "x_neurotox", "x_hcv", "x_thyroid", "x_new",
         "x_old_all", "x_everything", "rx_gout", "rx_neuro", "rx_thyroid", "urate_high", "thyroid_self", "tsh_abn"]
OUT["flag_counts_by_group"] = co.groupby("group")[flags].sum().astype(int).to_dict()
print(json.dumps(avail, indent=1))
print(co.groupby("group")[flags].sum())
# which neurotoxic drugs were found in the cohort
hits = rx[rx.SEQN.isin(co.SEQN)]
OUT["neurotox_drug_counts"] = {n: int(hits.loc[hits.drug.str.contains(n), "SEQN"].nunique()) for n in NEURO_RX}
print({k: v for k, v in OUT["neurotox_drug_counts"].items() if v})

# %% [markdown]
# ## Unrelated self-reported conditions (cosine reference range)

# %%
CONDS = {"asthma": d.MCQ010 == 1, "arthritis": d.MCQ160A == 1, "emphysema": d.MCQ160G == 1,
         "chronic_bronchitis": d.MCQ160K == 1, "liver_condition": d.MCQ160L == 1, "cancer": d.MCQ220 == 1,
         "osteoporosis": d.OSQ060 == 1, "thyroid_disease": d.thyroid_self == 1}
for k, v in CONDS.items():
    d[f"cond_{k}"] = v.astype(int)
co = d[d.cohort == 1]
OUT["condition_counts"] = {k: dict(n_normal_exam_no_foot_sx=int(co.loc[co.group.isin(["A_reference", "E_backpain_ctrl", "F_hands_only"]),
                                                                       f"cond_{k}"].sum()),
                                   n_reference_without=int(((co.group == "A_reference") & (co[f"cond_{k}"] == 0)).sum()))
                           for k in CONDS}
print(json.dumps(OUT["condition_counts"], indent=1))

# %% [markdown]
# ## Transfer-model scores (main model and every sensitivity analysis)
# Training groups (insensate without symptoms vs reference) get 5-fold out-of-fold scores; the other groups get the
# average of the five fold models ("averaged"). A second score ("uniform") gives every non-training participant the
# score of one randomly assigned fold model, so that all groups are scored by a single model fitted to 80% of the
# training data.

# %%
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=8, min_child_samples=50, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=5, verbose=-1, random_state=2026)
TRAIN = ["A_reference", "B_painless_PN"]
SCORED = ["A_reference", "B_painless_PN", "C_painful_PN", "D_hidden", "E_backpain_ctrl", "F_hands_only"]
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def crossfit(df, feats, make_model=None, seed=2026, design=False):
    """design=True: survey weights as sample_weight and folds grouped by PSU (second-review sensitivity analysis)."""
    df = df[df.group.isin(SCORED)]
    train = df[df.group.isin(TRAIN)]
    y = (train.group == "B_painless_PN").astype(int).values
    oth = df[~df.group.isin(TRAIN)]
    oof, avg = np.zeros(len(train)), np.zeros(len(oth))
    fold_pred = np.zeros((5, len(oth)))
    if design:
        psu = (train.SDMVSTRA * 10 + train.SDMVPSU).values
        folds = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(train, y, psu)
        sw = (train.w6 / train.w6.mean()).values
    else:
        folds = StratifiedKFold(5, shuffle=True, random_state=seed).split(train, y)
    for k, (tr, te) in enumerate(folds):
        m = make_model() if make_model else lgb.LGBMClassifier(**PARAMS)
        m = m.fit(train[feats].iloc[tr], y[tr], sample_weight=sw[tr]) if design else m.fit(train[feats].iloc[tr], y[tr])
        oof[te] = m.predict_proba(train[feats].iloc[te])[:, 1]
        fold_pred[k] = m.predict_proba(oth[feats])[:, 1]
    avg = fold_pred.mean(0)
    pick = np.random.default_rng(seed).integers(0, 5, len(oth))
    uni = fold_pred[pick, np.arange(len(oth))]
    s_avg = pd.Series(np.r_[logit(oof), logit(avg)], index=np.r_[train.index, oth.index])
    s_uni = pd.Series(np.r_[logit(oof), logit(uni)], index=np.r_[train.index, oth.index])
    return s_avg, s_uni, roc_auc_score(y, oof)


lin = lambda C=0.1, **kw: (lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                LogisticRegression(C=C, max_iter=5000, **kw)))
co = d[d.cohort == 1].copy()
SPECS = {}
AUC = {}


def add(name, df, feats=FEATS, make_model=None, uniform=False, design=False):
    s_avg, s_uni, auc = crossfit(df, feats, make_model, design=design)
    d[f"s_{name}"] = s_avg
    if uniform:
        d[f"s_{name}_uniform"] = s_uni
    AUC[name] = auc
    print(f"{name:14s} n={len(df):5d}  OOF AUROC={auc:.3f}")


add("main", co, uniform=True)
add("nogly", co, FEATS_NO_GLY)
# body size: obesity can cause foot pain mechanically, so the model is refitted without body-size measures
BODY = ["height", "weight", "bmi", "waist"]
add("nobody", co, [f for f in FEATS if f not in BODY])
add("nobody_nogly", co, [f for f in FEATS_NO_GLY if f not in BODY])
add("nodm", co[co.diabetes == 0], FEATS_NO_GLY)
add("lr", co, make_model=lin())
add("en", co, make_model=lin(penalty="elasticnet", solver="saga", l1_ratio=0.5))
# second review: survey-weighted training with PSU-grouped folds; and a model without demographic inputs
add("design", co, design=True)
DEMO = ["age", "female", "race_nhb", "race_mex", "race_oth_hisp", "race_other"]
add("nodemo", co, [f for f in FEATS if f not in DEMO])
for x in ["x_sciatica", "x_lowb12", "x_vascular", "x_comorb", "x_old_all", "x_urate", "x_neurotox", "x_hcv", "x_thyroid",
          "x_new", "x_everything"]:
    add(x[2:], co[co[x] == 0])
add("sciatica_lowb12_vascular", co[(co.x_sciatica == 0) & (co.x_lowb12 == 0) & (co.x_vascular == 0)])

# held-out cycle: train on 1999–2002, score 2003–04 (single model, as in HN-03)
early, late = co[co.SDDSRVYR.isin([1, 2])], co[co.SDDSRVYR == 3]
trn = early[early.group.isin(TRAIN)]
m = lgb.LGBMClassifier(**PARAMS).fit(trn[FEATS], (trn.group == "B_painless_PN").astype(int))
late = late[late.group.isin(SCORED)]
d.loc[late.index, "s_heldout"] = logit(m.predict_proba(late[FEATS])[:, 1])
ab = late[late.group.isin(TRAIN)]
AUC["heldout_2003_04"] = roc_auc_score((ab.group == "B_painless_PN").astype(int), d.loc[ab.index, "s_heldout"])
print("held-out AUROC 2003–04 = %.3f" % AUC["heldout_2003_04"])
OUT["auroc"] = AUC

# corroborated-symptom definition of SMN (scores unchanged; group recoded)
d["group_strict"] = d.group.where(~((d.group == "D_hidden") & (d.corroborated == 0)), "other")

# %% [markdown]
# ## Export

# %%
keep = (["SEQN", "SDDSRVYR", "SDMVPSU", "SDMVSTRA", "w6", "WTMEC2YR", "WTMEC4YR", "cohort", "group", "group_strict",
         "age", "female", "race", "educ", "pir", "smoke", "diabetes", "dm_self", "ELIGSTAT", "MORTSTAT", "UCOD", "PERMTH_EXAM",
         "abi_min", "abi_max", "abi_n_legs", "claud", "DIQ140", "DIQ150", "back_pain", "sx_feet_only", "sx_hands_and_feet",
         "mpq070_missing", "n_pain_sites", "cvd_hist", "corroborated", "foot_numb", "tsh", "LBDHCV"] + flags +
        [c for c in d if c.startswith("cond_")] + FEATS + [c for c in d if c.startswith("s_")])
keep = list(dict.fromkeys(k for k in keep if k in d))
ex = d[(d.w6 > 0) & d.w6.notna()][keep]  # every examined participant (design rows); the cohort is a subpopulation
ex.to_csv("hn11_analysis.csv.gz", index=False)
print("exported", ex.shape, " cohort:", int(ex.cohort.sum()))
OUT["feature_list"] = FEATS
OUT["fingerprint_biomarkers"] = FP
OUT["neurotox_list"], OUT["gout_rx_list"], OUT["thyroid_rx_list"] = NEURO_RX, GOUT_RX, THY_RX
json.dump(OUT, open("hn11_results.json", "w"), indent=1, default=float)
