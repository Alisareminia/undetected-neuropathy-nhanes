# %% [markdown]
# # HN-02 — H5: movement signature of neuropathy from minute-level accelerometry (NHANES 2003–2004)
#
# Pre-specified in HN_PROTOCOL.md (H5): train B (painless PN) vs A (reference) on accelerometry-derived features
# (cross-fitted LightGBM), then score D (hidden), E (back-pain control), F (hands-only) and C (painful PN), and compare
# logit-scores adjusted for age, sex and race. Prediction: D resembles PN more than E does.

# %%
import gc
import io
import json
import warnings
import zipfile

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
S = requests.Session()
S.headers["User-Agent"] = "neuropath-research/0.1 (academic project)"
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/{}.xpt"
OUT = {}


def xpt(name):
    df = pd.read_sas(io.BytesIO(S.get(BASE.format(name), timeout=240).content), format="xport")
    df["SEQN"] = df["SEQN"].round().astype(int)
    return df.drop_duplicates("SEQN").round(10)


demo, lex, diq, mpq, bmx = (xpt(n) for n in ["DEMO_C", "LEXPN_C", "DIQ_C", "MPQ_C", "BMX_C"])
d = demo[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"]].merge(lex, on="SEQN") \
    .merge(diq, on="SEQN", how="left").merge(mpq, on="SEQN", how="left").merge(bmx[["SEQN", "BMXBMI"]], on="SEQN", how="left")
L_, R_ = d.LEALPN.where(d.LEALPN >= 0), d.LEARPN.where(d.LEARPN >= 0)
d["pn"] = np.where((L_ >= 1) | (R_ >= 1), 1, np.where((L_ == 0) & (R_ == 0), 0, np.nan))
d["foot_sx"] = np.where(d.DIQ120 == 1, d.DIQ130.isin([2, 3]).astype(float), np.where(d.DIQ120 == 2, 0.0, np.nan))
d["hand_only"] = ((d.DIQ120 == 1) & (d.DIQ130 == 1)).astype(float)
d["back"] = (d.MPQ070 == 1).astype(float)
d = d[(d.RIDAGEYR >= 40) & d.pn.notna() & d.DIQ120.isin([1, 2])].copy()
d["group"] = np.select(
    [(d.pn == 1) & (d.foot_sx == 1), (d.pn == 1) & (d.foot_sx == 0), (d.pn == 0) & (d.foot_sx == 1),
     (d.pn == 0) & (d.foot_sx == 0) & (d.hand_only == 1), (d.pn == 0) & (d.foot_sx == 0) & (d.hand_only == 0) & (d.back == 1),
     (d.pn == 0) & (d.foot_sx == 0) & (d.hand_only == 0) & (d.back == 0)],
    ["C_painful_PN", "B_painless_PN", "D_hidden", "F_hands_only", "E_backpain_ctrl", "A_reference"], default="other")
keep_ids = set(d.SEQN)
print(d.group.value_counts())

# %% [markdown]
# ## Stream the minute-level file and build per-person features

# %%
r = S.get("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/paxraw_c.zip", timeout=1800)
z = zipfile.ZipFile(io.BytesIO(r.content))
name = z.namelist()[0]
print("zip member:", name)
with z.open(name) as fh, open("/kaggle/tmp_paxraw.xpt", "wb") as out:
    while True:
        b = fh.read(1 << 24)
        if not b:
            break
        out.write(b)
del r, z
gc.collect()

parts = []
for chunk in pd.read_sas("/kaggle/tmp_paxraw.xpt", format="xport", chunksize=2_000_000):
    chunk["SEQN"] = chunk["SEQN"].round().astype(int)
    chunk = chunk[chunk.SEQN.isin(keep_ids) & (chunk.PAXSTAT == 1) & (chunk.PAXCAL == 1)]
    parts.append(chunk[["SEQN", "PAXDAY", "PAXHOUR", "PAXMINUT", "PAXINTEN"]].astype(
        {"PAXDAY": "int8", "PAXHOUR": "int8", "PAXMINUT": "int8", "PAXINTEN": "float32"}))
pax = pd.concat(parts, ignore_index=True)
del parts
gc.collect()
print("minutes kept:", len(pax), "people:", pax.SEQN.nunique())

# %%
def nonwear(counts):
    """Troiano 2008: ≥60 consecutive zero minutes (allowing up to 2 min of 1–100) = non-wear."""
    c = counts.values
    nw = np.zeros(len(c), bool)
    i = 0
    while i < len(c):
        if c[i] == 0:
            j, inter = i, 0
            while j < len(c) and (c[j] == 0 or (c[j] <= 100 and inter < 2)):
                inter += c[j] > 0
                j += 1
            if j - i >= 60:
                nw[i:j] = True
            i = max(j, i + 1)
        else:
            i += 1
    return nw


feats = []
for seqn, g in pax.groupby("SEQN", sort=False):
    g = g.sort_values(["PAXDAY", "PAXHOUR", "PAXMINUT"])
    wear = ~nonwear(g.PAXINTEN)
    g = g[wear]
    days = g.groupby("PAXDAY").size()
    valid_days = days[days >= 600].index
    g = g[g.PAXDAY.isin(valid_days)]
    if len(valid_days) < 4:
        continue
    c = g.PAXINTEN.values
    act = c >= 100
    # activity fragmentation: active-to-sedentary transition probability (ASTP)
    trans = np.sum(act[:-1] & ~act[1:]) / max(np.sum(act[:-1]), 1)
    satp = np.sum(~act[:-1] & act[1:]) / max(np.sum(~act[:-1]), 1)
    daily = g.groupby("PAXDAY").PAXINTEN
    hourly = g.groupby("PAXHOUR").PAXINTEN.mean().reindex(range(24))
    top30 = g.groupby("PAXDAY").PAXINTEN.apply(lambda s: np.sort(s.values)[-30:].mean()).mean()
    row = dict(SEQN=seqn, valid_days=len(valid_days), wear_min=len(g) / len(valid_days),
               mean_cpm=c.mean(), log_total=np.log1p(daily.sum().mean()), mvpa_min=(c >= 2020).sum() / len(valid_days),
               light_min=((c >= 100) & (c < 2020)).sum() / len(valid_days), sed_frac=(c < 100).mean(),
               astp=trans, satp=satp, top30_cpm=top30, cv_daily=daily.sum().std() / max(daily.sum().mean(), 1),
               peak_hour=float(hourly.idxmax()) if hourly.notna().any() else np.nan,
               am_pm_ratio=hourly.loc[7:12].mean() / max(hourly.loc[13:20].mean(), 1),
               last_hour_drop=hourly.loc[17:20].mean() / max(hourly.loc[9:12].mean(), 1))
    for h in range(6, 23):
        row[f"h{h:02d}"] = np.log1p(hourly.get(h, np.nan))
    feats.append(row)
F = pd.DataFrame(feats)
print("people with ≥4 valid days:", len(F))
d = d.merge(F, on="SEQN", how="inner")
d.to_parquet("movement_features.parquet", index=False)
print(d.group.value_counts())

# %% [markdown]
# ## H5 — train B vs A on movement only; score other groups

# %%
MOVE = [c for c in F.columns if c not in ("SEQN", "valid_days", "wear_min")]
d["cluster"] = (d.SDMVSTRA * 10 + d.SDMVPSU).astype(int)
d["w"] = d.WTMEC2YR / d.WTMEC2YR.mean()
d["female"] = (d.RIAGENDR == 2).astype(int)
d["race"] = d.RIDRETH1.astype(int).astype(str)


def run(feats, label):
    tr_df = d[d.group.isin(["A_reference", "B_painless_PN"])]
    y = (tr_df.group == "B_painless_PN").astype(int).values
    oof = np.zeros(len(tr_df))
    oth = d[~d.group.isin(["A_reference", "B_painless_PN", "other"])]
    op = np.zeros(len(oth))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=2026).split(tr_df, y):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=6, min_child_samples=40, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, verbose=-1, random_state=2026)
        m.fit(tr_df[feats].iloc[tr], y[tr])
        oof[te] = m.predict_proba(tr_df[feats].iloc[te])[:, 1]
        op += m.predict_proba(oth[feats])[:, 1] / 5
    sc = pd.concat([pd.Series(oof, index=tr_df.index), pd.Series(op, index=oth.index)]).clip(1e-4, 1 - 1e-4)
    d[f"ls_{label}"] = np.log(sc / (1 - sc))
    x = d[d.group != "other"].dropna(subset=[f"ls_{label}"])
    r = smf.wls(f"ls_{label} ~ C(group, Treatment('A_reference')) + RIDAGEYR + female + C(race)", x, weights=x.w).fit(
        cov_type="cluster", cov_kwds={"groups": x.cluster})
    tab = pd.DataFrame({"diff_vs_A": r.params, "lo": r.conf_int()[0], "hi": r.conf_int()[1], "p": r.pvalues})
    tab = tab[tab.index.str.startswith("C(group")]
    tab.index = tab.index.str.extract(r"T\.(\w+)\]")[0].values
    cv = pd.Series(0.0, index=r.params.index)
    cv[[i for i in cv.index if "D_hidden" in i]] = 1
    cv[[i for i in cv.index if "E_backpain_ctrl" in i]] = -1
    t = r.t_test(cv.values)
    res = dict(auroc_B_vs_A=roc_auc_score(y, oof), n_train=len(y), group_diffs=tab.round(5).to_dict("index"),
               D_minus_E=dict(diff=float(t.effect[0]), lo=float(t.conf_int()[0][0]), hi=float(t.conf_int()[0][1]), p=float(t.pvalue)))
    print(f"\n[{label}] AUROC={res['auroc_B_vs_A']:.3f}\n", tab.round(4), "\nD−E:", res["D_minus_E"])
    return res


OUT["H5_movement_only"] = run(MOVE, "move")
OUT["H5_movement_plus_age_sex"] = run(MOVE + ["RIDAGEYR", "female"], "move_as")
json.dump(OUT, open("hn_movement_results.json", "w"), indent=1, default=float)
