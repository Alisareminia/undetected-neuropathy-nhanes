# %% [markdown]
# # HN-14 — Figures for the co-author revision (plot-only; reads HN-11, HN-12 and HN-13 outputs)
# Main text: 1 STROBE flow · 2 transfer test (a, b) · 3 biomarker profile heatmap · 4 mortality.
# Supplement: S1 specification curves · S2 cosine similarity with bootstrap CIs and a reference range.
# House style as in HN-09 (millimetre grid, journal palette, Source Sans 3).
# Group palette: SMN #00897B · insensate without symptoms #3C5488 · insensate with symptoms #D6492F · reference #3A3F47 ·
# back pain #B09C85 · hands #8FA0C2.

# %%
import glob
import json
import os
import textwrap
import urllib.request

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.stats import gaussian_kde


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


H11 = json.load(open(find("hn11_results.json")))
H12 = json.load(open(find("hn12_results.json")))
H13 = json.load(open(find("hn13_results.json")))["stats"]
REP = pd.read_csv(find("hn13_replicates.csv.gz"))
AN = pd.read_csv(find("hn11_analysis.csv.gz"))
AN = AN[AN.cohort == 1]
KM = pd.read_csv(find("km_curves.csv"))
KR = pd.read_csv(find("km_at_risk.csv"))
FPW = pd.read_csv(find("fingerprint_weighted.csv"))
os.makedirs("figures", exist_ok=True)

# %% [markdown]
# ## Style and layout helpers

# %%
SANS = "DejaVu Sans"
try:
    os.makedirs("fonts", exist_ok=True)
    for w in ["400-normal", "600-normal", "700-normal", "400-italic"]:
        p = f"fonts/SourceSans3-{w}.ttf"
        if not os.path.exists(p):
            urllib.request.urlretrieve(f"https://cdn.jsdelivr.net/fontsource/fonts/source-sans-3@latest/latin-{w}.ttf", p)
        font_manager.fontManager.addfont(p)
    SANS = font_manager.FontProperties(fname="fonts/SourceSans3-400-normal.ttf").get_name()
except Exception as e:
    print("font fallback:", e)
    try:
        for w in ["Regular", "Bold", "Italic"]:
            p = f"fonts/Lato-{w}.ttf"
            if not os.path.exists(p):
                urllib.request.urlretrieve(f"https://github.com/google/fonts/raw/main/ofl/lato/Lato-{w}.ttf", p)
            font_manager.fontManager.addfont(p)
        SANS = "Lato"
    except Exception as e2:
        print("font fallback 2:", e2)
print("font:", SANS)

INK, INK2, INK3, GRID = "#1c1f24", "#4b5059", "#8b9099", "#e6e7ea"
C_HID, C_PNL, C_PFL, C_PUR = "#00897B", "#3C5488", "#D6492F", "#7E6148"
C_HID_TXT = "#00695F"
ORDER = ["A_reference", "E_backpain_ctrl", "F_hands_only", "D_hidden", "B_painless_PN", "C_painful_PN"]
GC = {"A_reference": "#3A3F47", "E_backpain_ctrl": "#B09C85", "F_hands_only": "#8FA0C2", "D_hidden": C_HID,
      "B_painless_PN": C_PNL, "C_painful_PN": C_PFL}
GLS = {"A_reference": "-", "E_backpain_ctrl": (0, (4, 1.6)), "F_hands_only": (0, (1, 1.2)), "D_hidden": "-",
       "B_painless_PN": "-", "C_painful_PN": "-"}
NAME = {"A_reference": "Reference", "E_backpain_ctrl": "Back pain only", "F_hands_only": "Hand symptoms only",
        "D_hidden": "SMN", "B_painless_PN": "Insensate, no symptoms", "C_painful_PN": "Insensate with symptoms"}
NM = {"waist": "Waist", "bmi": "BMI", "weight": "Weight", "height": "Height", "sbp": "Systolic BP", "dbp": "Diastolic BP",
      "hba1c": "HbA1c", "glucose": "Glucose", "egfr": "eGFR", "albumin": "Albumin", "alp": "ALP", "ast": "AST", "alt": "ALT",
      "bun": "BUN", "calcium": "Calcium", "chol": "Cholesterol", "bicarb": "Bicarbonate", "creat": "Creatinine", "ggt": "GGT",
      "iron": "Iron", "ldh": "LDH", "phos": "Phosphorus", "bili": "Bilirubin", "protein": "Total protein", "trig": "Triglycerides",
      "uric": "Uric acid", "sodium": "Sodium", "potassium": "Potassium", "chloride": "Chloride", "wbc": "WBC",
      "lymph_pct": "Lymphocytes %", "mono_pct": "Monocytes %", "neut_pct": "Neutrophils %", "eos_pct": "Eosinophils %",
      "hgb": "Hemoglobin", "mcv": "MCV", "rdw": "RDW", "platelets": "Platelets", "mpv": "MPV", "crp": "CRP",
      "dm_self": "Diabetes (self-report)"}
DOMAINS = [("Glucose & lipids", "#D1A32C", ["hba1c", "glucose", "trig", "chol", "uric"]),
           ("Body size", "#9A4F8F", ["waist", "bmi", "weight", "height"]),
           ("Blood pressure", "#EBA7BB", ["sbp", "dbp"]),
           ("Liver & protein", "#5F8F3E", ["alp", "ast", "alt", "ggt", "bili", "albumin", "protein", "ldh"]),
           ("Kidney & electrolytes", "#7E8C96", ["egfr", "bun", "creat", "potassium", "sodium", "chloride", "bicarb",
                                                 "calcium", "phos"]),
           ("Blood count & inflammation", "#8A5A3E", ["wbc", "lymph_pct", "mono_pct", "neut_pct", "eos_pct", "hgb", "mcv",
                                                      "rdw", "platelets", "mpv", "crp", "iron"])]
DOMC = {f: c for _, c, fs in DOMAINS for f in fs}
mpl.rcParams.update({
    "font.family": [SANS, "DejaVu Sans"], "font.size": 6.2, "axes.titlesize": 6.5, "axes.labelsize": 6.2, "xtick.labelsize": 5.8,
    "ytick.labelsize": 5.8, "legend.fontsize": 5.6, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2, "xtick.major.pad": 1.8, "ytick.major.pad": 1.8, "axes.labelpad": 2.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK,
    "ytick.labelcolor": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42, "svg.fonttype": "none",
    "lines.solid_capstyle": "butt", "mathtext.default": "regular",
})
MM = 1 / 25.4
WMM = 183.0


def newfig(hmm):
    fig = plt.figure(figsize=(WMM * MM, hmm * MM))
    fig._hmm = hmm
    return fig


def axmm(fig, x, top, w, h, **kw):
    """Axes placed in millimetres: x from the left edge, top from the top edge."""
    H = fig._hmm
    return fig.add_axes([x / WMM, 1 - (top + h) / H, w / WMM, h / H], **kw)


def ftext(fig, x, top, s, **kw):
    """Figure text placed in millimetres."""
    return fig.text(x / WMM, 1 - top / fig._hmm, s, **kw)


def letter(fig, x, top, s):
    ftext(fig, x, top, s, fontsize=9, fontweight="bold", va="top", ha="left")


def ptitle(ax, s, pad=3, color=INK):
    ax.set_title(s, loc="left", fontsize=6.6, fontweight="semibold", pad=pad, color=color)


def box(ax, x, y, w, h, fc="white", ec=INK, lw=0.6, r=0.25, z=2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc, edgecolor=ec, lw=lw,
                                zorder=z, ls=ls))


def arrow(ax, p0, p1, c=INK2, lw=0.6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>,head_length=2.2,head_width=1.3", color=c, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=1))


def tint(c, a):
    rgb = np.array(mpl.colors.to_rgb(c))
    return tuple(1 - a * (1 - rgb))


def ylabels_left(ax, labels, bold=()):
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for t in ax.get_yticklabels():
        if t.get_text() in bold:
            t.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def logx(ax, ticks, lim):
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(*lim)


def callouts(ax, items, x_text, ha, ylo, yhi, gap, fs=5.3):
    """Leader-line labels stacked in a column with a minimum vertical gap (no overlaps)."""
    if not items:
        return
    items = sorted(items, key=lambda t: t[1])
    ys = [min(max(t[1], ylo), yhi) for t in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    if ys[-1] > yhi:
        ys = [y - (ys[-1] - yhi) for y in ys]
    for (x, y, lab), yt in zip(items, ys):
        ax.annotate(lab, (x, y), (x_text, yt), ha=ha, va="center", fontsize=fs, color=INK2, zorder=6,
                    arrowprops=dict(arrowstyle="-", lw=0.35, color=INK3, shrinkA=0.5, shrinkB=1.2))


def save(fig, name):
    for a in fig.axes:
        if hasattr(a, "_colorbar") or a.get_label() == "<colorbar>":
            continue
        for side in ("left", "bottom"):
            sp = a.spines[side]
            if sp.get_visible() and a.axison:
                sp.set_position(("outward", 2.5))
    for ext, kw in [("png", dict(dpi=600)), ("pdf", {}), ("svg", {})]:
        fig.savefig(f"figures/{name}.{ext}", bbox_inches="tight", pad_inches=0.04, **kw)
    print("saved", name)


def pfmt(p, prefix="P "):
    """P value for display; values stored as 0 were rounded below 1e-5."""
    if p is None:
        return ""
    if p <= 0:
        return prefix + "< 0.001"
    if p < 0.001:
        m, e = f"{p:.0e}".split("e")
        return prefix + f"= {m} × 10$^{{{int(e)}}}$"
    if p < 0.01:
        return prefix + f"= {p:.3f}"
    return prefix + f"= {p:.2f}"


def pshort(p):
    if p is None:
        return "–"
    if p < 0.001:
        return "<.001"
    if p < 0.01:
        return f"{p:.3f}"[1:]
    return f"{p:.2f}"[1:] if p < 1 else "1.0"


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def fmt_ci(e, lo, hi, d=2):
    return f"{e:.{d}f} ({lo:.{d}f}–{hi:.{d}f})"


def _tr(log):
    return (np.log, np.exp) if log else ((lambda v: np.asarray(v, float)), (lambda v: np.asarray(v, float)))


def interval(ax, est, lo, hi, y, c, log=False, mk="o", ms=11, z=3, hollow=False, thick=2.0, vertical=False):
    """Point with 66% (thick) and 95% (thin) intervals, ggdist-style."""
    f, g = _tr(log)
    m, a, b = f(est), f(lo), f(hi)
    se = (b - a) / 3.92
    t66 = g(np.array([m - 0.954 * se, m + 0.954 * se]))
    if vertical:
        ax.plot([y, y], [lo, hi], color=c, lw=0.7, zorder=z, solid_capstyle="butt")
        ax.plot([y, y], t66, color=c, lw=thick, zorder=z, solid_capstyle="butt")
        ax.scatter(y, est, s=ms, marker=mk, facecolor="white" if hollow else c, edgecolor=c if hollow else "white", lw=0.8,
                   zorder=z + 1)
    else:
        ax.plot([lo, hi], [y, y], color=c, lw=0.7, zorder=z, solid_capstyle="butt")
        ax.plot(t66, [y, y], color=c, lw=thick, zorder=z, solid_capstyle="butt")
        ax.scatter(est, y, s=ms, marker=mk, facecolor="white" if hollow else c, edgecolor=c if hollow else "white", lw=0.8,
                   zorder=z + 1)


def gfill(ax, x, y0, y1, color, a_lo=0.55, a_hi=0.05, z=1):
    """Fill between y0 and y1 with a vertical colour gradient (opaque at the base, fading upwards); linear axes only."""
    x = np.asarray(x, float)
    y0 = np.broadcast_to(np.asarray(y0, float), x.shape)
    y1 = np.asarray(y1, float)
    poly = ax.fill_between(x, y0, y1, facecolor="none", edgecolor="none", zorder=z)
    lo_, hi_ = float(y0.min()), float(y1.max())
    img = np.zeros((256, 2, 4))
    img[..., :3] = mpl.colors.to_rgb(color)
    img[..., 3] = np.linspace(a_lo, a_hi, 256)[:, None]
    xl, yl = ax.get_xlim(), ax.get_ylim()
    auto = ax.get_autoscale_on()
    im = ax.imshow(img, extent=[x.min(), x.max(), lo_, hi_], origin="lower", aspect="auto", zorder=z, interpolation="bilinear")
    im.set_clip_path(poly.get_paths()[0], transform=ax.transData)
    if not auto:
        ax.set_xlim(xl)
        ax.set_ylim(yl)
    return im


def eye(ax, est, lo, hi, y, c, h=0.36, log=False, mk="o", ms=13, z=3, hollow=False):
    """Confidence distribution (normal density of the estimate) above a 66%/95% interval."""
    f, g = _tr(log)
    m, a, b = f(est), f(lo), f(hi)
    se = (b - a) / 3.92
    xs = np.linspace(m - 3.2 * se, m + 3.2 * se, 160)
    dens = np.exp(-0.5 * ((xs - m) / se) ** 2)
    if log:
        ax.fill_between(g(xs), y + 0.05, y + 0.05 + h * dens, color=tint(c, 0.3), lw=0, zorder=z - 1)
    else:
        gfill(ax, g(xs), y + 0.05, y + 0.05 + h * dens, c, a_lo=0.6, a_hi=0.12, z=z - 1)
    ax.plot(g(xs), y + 0.05 + h * dens, color=c, lw=0.5, zorder=z - 1)
    interval(ax, est, lo, hi, y, c, log=log, mk=mk, ms=ms, z=z, hollow=hollow)


def eye_handle(c, label):
    return Line2D([], [], color=c, lw=2.0, marker="o", ms=3.4, mfc=c, mec="white", label=label)


def smooth_ribbon(ax, x0, x1, a0, b0, a1, b1, color, alpha=1.0, z=1):
    t = np.linspace(0, 1, 80)
    sm = t * t * (3 - 2 * t)
    xs = x0 + (x1 - x0) * t
    ax.fill_between(xs, a0 + (a1 - a0) * sm, b0 + (b1 - b0) * sm, color=color, alpha=alpha, lw=0, zorder=z)


LEGENDS = {}
rng = np.random.default_rng(4)


# %%
NG = AN.group.value_counts().to_dict()
GK = dict(zip("BCDEF", ["B_painless_PN", "C_painful_PN", "D_hidden", "E_backpain_ctrl", "F_hands_only"]))
KEY = {v: k for k, v in GK.items()}
PREV = H12["prevalence"]
CX = H12["contrasts"]
COX = H12["cox"]


def bq(col, q):
    return float(REP[col].quantile(q))


def bstat(col):
    s = H13[col]
    return s["est"], s["lo"], s["hi"]


def ci_txt(e, lo, hi, d=2):
    f = lambda v: f"{v:.{d}f}".replace("-", "−")
    return f"{f(e)} ({f(lo)} to {f(hi)})" if lo < 0 else f"{f(e)} ({f(lo)}–{f(hi)})"


# %% [markdown]
# ## Figure 1 — STROBE participant flow

# %%
FL = H11["flow"]
FIG1_H = 4 + 4 * 15.0 + 8.0 + 4 + 16.0 + 29 + 3
fig = newfig(FIG1_H)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, WMM)
ax.set_ylim(FIG1_H, 0)
ax.axis("off")


def fbox(x, y, w, h, lines, fc="white", ec=INK, lw=0.6, bold_first=True, fs=6.0, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", facecolor=fc, edgecolor=ec, lw=lw,
                                zorder=2))
    n = len(lines)
    lh = 3.0
    y0 = y + h / 2 - (n - 1) * lh / 2
    for i, t in enumerate(lines):
        ax.text(x + (w / 2 if align == "center" else 1.6), y0 + i * lh, t, ha=align, va="center", fontsize=fs,
                fontweight="bold" if (i == 0 and bold_first) else "normal", color=INK, zorder=3)


def varrow(x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>,head_length=2.4,head_width=1.4", color=INK2, lw=0.6,
                                 shrinkA=0, shrinkB=0, zorder=1))


def harrow(x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>,head_length=2.4,head_width=1.4", color=INK2, lw=0.6,
                                 shrinkA=0, shrinkB=0, zorder=1))


MX, MW, BH, STEP = 40, 58, 8.0, 15.0
cx = MX + MW / 2
steps = [(["NHANES 1999–2004 participants", f"n = {FL['nhanes_1999_2004']:,}"],
          ["Aged <40 years", f"n = {FL['nhanes_1999_2004'] - FL['aged_40_plus']:,}"]),
         (["Aged ≥40 years", f"n = {FL['aged_40_plus']:,}"],
          ["Not examined in the mobile examination center", f"n = {FL['aged_40_not_examined']:,}"]),
         (["Examined", f"n = {FL['examined_40_plus']:,}"],
          ["No valid 10-g monofilament result for either foot", f"n = {FL['no_valid_monofilament']:,}"]),
         (["Valid monofilament examination", f"n = {FL['valid_monofilament']:,}"],
          ["Question on hand or foot pain or tingling", "not answered", f"n = {FL['no_symptom_answer']:,}"]),
         (["Included", f"n = {FL['included']:,}"], None)]
for i, (main_, side) in enumerate(steps):
    y = 4 + i * STEP
    last = i == len(steps) - 1
    fbox(MX, y, MW, BH, main_, fc="#f5f5f5" if last else "white", lw=0.9 if last else 0.6)
    if side:
        varrow(cx, y + BH, y + STEP)
        ym = y + BH + (STEP - BH) / 2
        harrow(cx, 112, ym)
        hb = 3.0 * len(side) + 2.2
        fbox(112, ym - hb / 2, 66, hb, side, ec=INK3, bold_first=False, fs=5.6, align="left")
y = 4 + (len(steps) - 1) * STEP + BH
yb = y + 4
n_norm = sum(NG[g] for g in ["D_hidden", "E_backpain_ctrl", "F_hands_only", "A_reference"])
n_ins = NG["C_painful_PN"] + NG["B_painless_PN"]
XN0, XN1, XI0, XI1 = 4.5, 119, 125, 179
ax.plot([cx, cx], [y, yb], color=INK2, lw=0.6, zorder=1)
ax.plot([(XN0 + XN1) / 2, (XI0 + XI1) / 2], [yb, yb], color=INK2, lw=0.6, zorder=1)
for xa, xb in [(XN0, XN1), (XI0, XI1)]:
    varrow((xa + xb) / 2, yb, yb + 3.0)
fbox(XN0, yb + 3.0, XN1 - XN0, 7.5, ["Monofilament normal (all sites felt)", f"n = {n_norm:,}"], fc="#fafafa")
fbox(XI0, yb + 3.0, XI1 - XI0, 7.5, ["≥1 insensate site", f"n = {n_ins:,}"], fc="#fafafa")
ybar = yb + 13.5
yg = yb + 16.0
GDEF = {"D_hidden": ("SMN", "Foot pain or tingling", "Study group"),
        "E_backpain_ctrl": ("Back pain only", "Low back pain; no foot or hand symptoms", "Negative control"),
        "F_hands_only": ("Hand symptoms only", "Hand pain or tingling; no foot symptoms", "Negative control"),
        "A_reference": ("Reference", "None of these symptoms", "Model training (negative)"),
        "C_painful_PN": ("Insensate with symptoms", "Foot pain or tingling", "Comparison group"),
        "B_painless_PN": ("Insensate, no symptoms", "No foot pain or tingling", "Model training (positive)")}
gap = 1.5
gwn = (XN1 - XN0 - 3 * gap) / 4
gwi = (XI1 - XI0 - gap) / 2
pos = {g: (XN0 + k * (gwn + gap), gwn) for k, g in enumerate(["D_hidden", "E_backpain_ctrl", "F_hands_only", "A_reference"])}
pos.update({g: (XI0 + k * (gwi + gap), gwi) for k, g in enumerate(["C_painful_PN", "B_painless_PN"])})
for xa, xb, gs in [(XN0, XN1, ["D_hidden", "A_reference"]), (XI0, XI1, ["C_painful_PN", "B_painless_PN"])]:
    ax.plot([(xa + xb) / 2] * 2, [yb + 10.5, ybar], color=INK3, lw=0.5, zorder=1)
    ax.plot([pos[gs[0]][0] + pos[gs[0]][1] / 2, pos[gs[1]][0] + pos[gs[1]][1] / 2], [ybar, ybar], color=INK3, lw=0.5, zorder=1)
for g, (x0, w_) in pos.items():
    xm = x0 + w_ / 2
    ax.plot([xm, xm], [ybar, yg], color=INK3, lw=0.5, zorder=1)
    nm, df_, role = GDEF[g]
    p = PREV[g]
    c = GC[g]
    ax.add_patch(FancyBboxPatch((x0, yg), w_, 29, boxstyle="round,pad=0,rounding_size=0.8", facecolor=tint(c, 0.12), edgecolor=c,
                                lw=1.3 if g == "D_hidden" else 0.6, zorder=2))
    ax.text(xm, yg + 3.3, nm, ha="center", va="center", fontsize=6.0, fontweight="bold", color=INK, zorder=3)
    ax.text(xm, yg + 8.0, "\n".join(textwrap.wrap(df_, 24)), ha="center", va="center", fontsize=4.9, color=INK2, zorder=3,
            linespacing=1.15)
    ax.text(xm, yg + 13.6, f"n = {NG[g]:,}", ha="center", va="center", fontsize=5.8, color=INK, zorder=3)
    ax.text(xm, yg + 17.4, f"{p['prop'] * 100:.1f}% ({p['lo'] * 100:.1f}–{p['hi'] * 100:.1f})", ha="center", va="center",
            fontsize=5.5, color=INK, zorder=3)
    km_ = H12["km_n"][g]
    ax.text(xm, yg + 21.4, f"Died by 2019: {km_['deaths']:,} / {km_['n']:,}", ha="center", va="center", fontsize=4.9,
            color=INK2, zorder=3)
    ax.text(xm, yg + 25.4, role, ha="center", va="center", fontsize=4.9, color=INK2, style="italic", zorder=3)
FIG1_BOTTOM = yg + 29
fs_share = PREV["share_of_foot_symptoms"]
save(fig, "Fig1_flow")
plt.show()
LEGENDS["Fig1_flow"] = (
    "**Figure 1 | Participant flow (STROBE).** Adults aged ≥40 years in NHANES 1999–2004 with a valid 10-g monofilament result "
    "for at least one foot and an answer to the question on pain or tingling in the hands or feet were assigned to six mutually "
    "exclusive groups by the monofilament result (any insensate site among three plantar sites on either foot) and by "
    "self-reported symptoms in the past 3 months. n, unweighted; %, survey-weighted percentage of US adults aged ≥40 years "
    "(95% CI, Taylor linearization with design degrees of freedom); deaths/n, participants eligible for mortality linkage "
    "with follow-up time, through 31 December 2019. SMN, symptomatic, monofilament-normal.")

# %% [markdown]
# ## Figure 2 — Transfer test (a, score distributions; b, survey-weighted contrasts with pipeline-bootstrap uncertainty)

# %%
sc = AN[(AN.cohort == 1) & AN.group.isin(ORDER)].copy()
sc["lg"] = sc.s_main
fig = newfig(80)
ax = axmm(fig, 25, 8, 62, 66)
xs = np.linspace(np.percentile(sc.lg, 0.5), np.percentile(sc.lg, 99.5), 400)
ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(-0.45, len(ORDER) - 0.3)
ax.set_autoscale_on(False)
ref_med = sc.loc[sc.group == "A_reference", "lg"].median()
for i, g in enumerate(ORDER):
    v = sc.loc[sc.group == g, "lg"].values
    dens = gaussian_kde(v, bw_method=0.3)(xs)
    dens = dens / dens.max() * 0.52
    y = len(ORDER) - 1 - i
    gfill(ax, xs, y, y + dens, GC[g], a_lo=0.55, a_hi=0.08, z=2)
    ax.plot(xs, y + dens, color=GC[g], lw=0.85, zorder=2)
    ax.plot([xs[0], xs[-1]], [y, y], color=GRID, lw=0.5, zorder=1)
    vv = v[(v > xs[0]) & (v < xs[-1])]
    ax.scatter(vv, y - 0.27 + rng.uniform(-0.04, 0.04, len(vv)), s=0.25, color=GC[g], alpha=0.35, lw=0, rasterized=True, zorder=1)
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([q1, q3], [y - 0.1, y - 0.1], color=GC[g], lw=1.4, zorder=3)
    ax.scatter(q2, y - 0.1, s=8, facecolor="white", edgecolor=GC[g], lw=0.8, zorder=4)
ax.axvline(ref_med, color=INK3, lw=0.5, ls=(0, (2, 2)), zorder=0)
lab_a = [f"{NAME[g]}\nn = {NG[g]:,}" for g in ORDER[::-1]]
ylabels_left(ax, lab_a, bold=(f"{NAME['D_hidden']}\nn = {NG['D_hidden']:,}",))
ax.set_ylim(-0.45, len(ORDER) - 0.3)
ax.set_xlim(xs[0], xs[-1])
ax.set_xlabel("Neuropathy-similarity score (logit)")
ptitle(ax, "Score distribution")
letter(fig, 0, 2, "a")

axb = axmm(fig, 124, 8, 40, 66)
axb.set_xlim(-0.12, 0.9)
axb.set_ylim(-0.55, 4.65)
axb.set_autoscale_on(False)
variants = [("No glycemic features", CX["nogly"], "s"), ("Adults without diabetes", CX["nodm"], "^"),
            ("Held-out cycle 2003–04", CX["heldout"], "D")]
rows = ["E_backpain_ctrl", "F_hands_only", "D_hidden", "B_painless_PN", "C_painful_PN"]
for i, g in enumerate(rows):
    y = len(rows) - 1 - i
    k = KEY[g]
    if g == "D_hidden":
        axb.axhspan(y - 0.47, y + 0.53, color="#f2f2f2", lw=0, zorder=0)
    r = REP[f"{k}_vs_A"].dropna().values
    est = H13[f"{k}_vs_A"]["est"]
    xx = np.linspace(np.percentile(r, 0.2), np.percentile(r, 99.8), 160)
    dens = gaussian_kde(r)(xx)
    dens = dens / dens.max() * 0.34
    gfill(axb, xx, y + 0.15, y + 0.15 + dens, GC[g], a_lo=0.6, a_hi=0.12, z=2)
    axb.plot(xx, y + 0.15 + dens, color=GC[g], lw=0.5, zorder=2)
    hs = H13[f"{k}_vs_A"]
    l95, h95, l66, h66 = hs["lo"], hs["hi"], hs["lo66"], hs["hi66"]
    axb.plot([l95, h95], [y + 0.1] * 2, color=GC[g], lw=0.7, zorder=3)
    axb.plot([l66, h66], [y + 0.1] * 2, color=GC[g], lw=2.0, zorder=3)
    axb.scatter(est, y + 0.1, s=11, color=GC[g], edgecolor="white", lw=0.8, zorder=4)
    axb.text(1.04, y + 0.1, ci_txt(est, l95, h95), transform=axb.get_yaxis_transform(), ha="left", va="center", fontsize=5.2,
             color=INK)
    for (lab, cx_, mk), dy in zip(variants, [-0.1, -0.22, -0.34]):
        w = cx_[f"{k}_vs_A"]
        axb.plot([w["lo"], w["hi"]], [y + dy] * 2, color=tint(GC[g], 0.6), lw=0.6, zorder=2)
        axb.scatter(w["est"], y + dy, s=7, marker=mk, facecolor="white", edgecolor=GC[g], lw=0.6, zorder=3)
de = H13["D_minus_E"]
pg = H13["DE_over_BA"]
yD = len(rows) - 1 - rows.index("D_hidden")
axb.text(1.04, yD - 0.2, f"SMN − back pain: {ci_txt(de['est'], de['lo'], de['hi'])}", transform=axb.get_yaxis_transform(),
         ha="left", va="center", fontsize=4.9, color=C_HID_TXT, style="italic")
axb.text(1.04, yD - 0.4, f"= {pg['est'] * 100:.0f}% ({pg['lo'] * 100:.0f}–{pg['hi'] * 100:.0f}%) of the insensate gap",
         transform=axb.get_yaxis_transform(), ha="left", va="center", fontsize=4.9, color=C_HID_TXT, style="italic")
axb.axvline(0, color=INK, lw=0.5)
ylabels_left(axb, [NAME[g] for g in rows[::-1]], bold=(NAME["D_hidden"],))
axb.set_ylim(-0.55, len(rows) - 0.35)
axb.set_xlim(-0.12, 0.9)
axb.set_xlabel("Difference vs reference (logit score)")
axb.legend(handles=[eye_handle(INK2, "Main model (bootstrap)")] +
           [Line2D([], [], marker=mk, color=tint(INK2, 0.6), mfc="white", mec=INK2, lw=0.6, ms=3, label=l) for l, _, mk in variants],
           loc="upper right", fontsize=5.3, handletextpad=0.3, borderaxespad=0.2, labelspacing=0.35, handlelength=1.6)
ptitle(axb, "Survey-weighted contrast vs reference")
axb.text(1.04, 4.62, "Estimate (95% CI)", transform=axb.get_yaxis_transform(), ha="left", va="bottom", fontsize=5.1,
         fontweight="semibold")
letter(fig, 92, 2, "b")
save(fig, "Fig2_transfer_test")
plt.show()
A11 = H11["auroc"]
LEGENDS["Fig2_transfer_test"] = (
    "**Figure 2 | A model trained only on insensate adults without symptoms scores the SMN group above the negative "
    "controls.** **a**, Neuropathy-similarity scores (logit) by group: density, individual adults, interquartile range and "
    "median; dashed line, reference median. The two training groups have five-fold out-of-fold scores; the other groups are "
    "scored by the average of the five fold models. **b**, Survey-weighted differences in mean score versus reference, adjusted "
    "for age, sex and race and ethnicity. Main model: point estimate, bootstrap distribution and 66% (thick) and 95% (thin) "
    f"bias-corrected percentile intervals from {H13['D_minus_E']['n_rep']:,} Rao–Wu bootstrap replicates of the whole pipeline (model "
    "refitting, scoring and regression; primary sampling units resampled within strata). Smaller symbols, design-based 95% CIs "
    "(Taylor linearization, scores fixed) for a model without glycemic features, adults without diabetes and the held-out "
    "2003–04 cycle (model trained on 1999–2002). The primary contrast, SMN minus back pain only, is also expressed as a "
    "percentage of the score gap between insensate adults without symptoms and the reference group. SMN, symptomatic, "
    "monofilament-normal.")

# %% [markdown]
# ## Figure 3 — Biomarker profile (survey-weighted SMD heatmap with bootstrap CIs)

# %%
FPG = FPW[FPW.profile.isin(ORDER)]
wide = FPG.pivot(index="profile", columns="feature", values="smd")
wlo = FPG.pivot(index="profile", columns="feature", values="lo")
whi = FPG.pivot(index="profile", columns="feature", values="hi")
sigm = (wlo > 0) | (whi < 0)
cols = wide.loc["D_hidden"].sort_values(ascending=False).index.tolist()
hm_rows = ["C_painful_PN", "B_painless_PN", "D_hidden", "F_hands_only", "E_backpain_ctrl"]
DIV = LinearSegmentedColormap.from_list("div", ["#2E4374", "#7D8FBF", "#F6F6F4", "#E79A8A", "#A8321F"])
norm = TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=0.6)
fig = newfig(72)
nc = len(cols)
HX, HW = 32, 126
cw = HW / nc
axt = axmm(fig, HX, 8, HW, 14)
dv = wide.loc["D_hidden", cols].values
axt.bar(np.arange(nc) + 0.5, dv, width=0.62, color=[DIV(norm(np.clip(v, -0.6, 0.6))) for v in dv], edgecolor="none")
axt.errorbar(np.arange(nc) + 0.5, dv, yerr=[dv - wlo.loc["D_hidden", cols].values, whi.loc["D_hidden", cols].values - dv],
             fmt="none", ecolor=INK, elinewidth=0.4, capsize=0)
axt.axhline(0, color=INK, lw=0.5)
axt.set_xlim(0, nc)
axt.set_xticks([])
axt.spines["bottom"].set_visible(False)
axt.set_ylabel("SMN", labelpad=2)
ymax = max(0.5, float(np.nanmax(whi.loc["D_hidden", cols])) + 0.05)
ymin = min(-0.4, float(np.nanmin(wlo.loc["D_hidden", cols])) - 0.05)
axt.set_ylim(ymin, ymax)
axt.set_yticks([-0.3, 0, 0.3])
ptitle(axt, "Adjusted standardized mean difference vs reference (SMD)")
axs = axmm(fig, HX, 23.6, HW, 2.0)
for j, f in enumerate(cols):
    axs.add_patch(Rectangle((j + 0.06, 0), 0.88, 1, facecolor=DOMC.get(f, INK3), edgecolor="none"))
axs.set_xlim(0, nc)
axs.set_ylim(0, 1)
axs.axis("off")
ftext(fig, HX - 1.5, 24.6, "Domain", ha="right", va="center", fontsize=5.4, color=INK2)
hh = cw * len(hm_rows)
ax = axmm(fig, HX, 27, HW, hh)
ri = hm_rows[::-1].index("D_hidden")
ax.axhspan(ri, ri + 1, color=tint(C_HID, 0.12), lw=0, zorder=0)
for i, g in enumerate(hm_rows[::-1]):
    ax.plot([0.5, nc - 0.5], [i + 0.5, i + 0.5], color=GRID, lw=0.4, zorder=1)
    for j, f in enumerate(cols):
        v = float(wide.loc[g, f])
        r = 0.1 + 0.36 * min(abs(v) / 0.5, 1.0)
        sig = bool(sigm.loc[g, f])
        ax.add_patch(Circle((j + 0.5, i + 0.5), r, facecolor=DIV(norm(np.clip(v, -0.6, 0.6))), edgecolor=INK if sig else "none",
                            lw=0.45, zorder=3))
ax.add_patch(Rectangle((0, ri), nc, 1, fill=False, edgecolor=C_HID, lw=0.9, zorder=5, clip_on=False))
ax.set_xlim(0, nc)
ax.set_ylim(0, len(hm_rows))
ax.set_xticks(np.arange(nc) + 0.5)
ax.set_xticklabels([NM.get(f, f) for f in cols], rotation=90, fontsize=5.4)
ax.set_yticks(np.arange(len(hm_rows)) + 0.5)
ax.set_yticklabels([NAME[g] for g in hm_rows[::-1]], fontsize=5.8)
for t, g in zip(ax.get_yticklabels(), hm_rows[::-1]):
    if g == "D_hidden":
        t.set_fontweight("bold")
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
axr = axmm(fig, HX + HW + 4, 27, 16, hh)
cnt = [int(sigm.loc[g, cols].sum()) for g in hm_rows[::-1]]
axr.barh(np.arange(len(hm_rows)) + 0.5, cnt, height=0.62, color=[GC[g] for g in hm_rows[::-1]])
for i, c in enumerate(cnt):
    axr.text(c + 0.6, i + 0.5, str(c), va="center", fontsize=5.3)
axr.set_ylim(0, len(hm_rows))
axr.set_yticks([])
axr.spines["left"].set_visible(False)
axr.set_xlim(0, nc * 0.75)
axr.set_xticks([0, 10, 20])
axr.set_xlabel("n, CI ≠ 0", labelpad=1.5)
cbx = axmm(fig, HX + HW + 4, 11, 16, 1.6)
cb = mpl.colorbar.ColorbarBase(cbx, cmap=DIV, norm=norm, orientation="horizontal", ticks=[-0.5, 0, 0.5])
cb.outline.set_linewidth(0.4)
cbx.tick_params(labelsize=5.2, length=1.5)
cbx.set_title("SMD", fontsize=5.3, pad=2, loc="left")
axk = axmm(fig, HX + HW + 4, 15, 16, 6)
axk.set_xlim(0, 16)
axk.set_ylim(0, 6)
axk.set_aspect("equal")
axk.axis("off")
for xk, vk in [(2.2, 0.1), (6.6, 0.3), (11.4, 0.5)]:
    rk = (0.1 + 0.36 * min(vk / 0.5, 1)) * cw
    axk.add_patch(Circle((xk, 3.9), rk, facecolor="#cfd2d7", edgecolor=INK, lw=0.45))
    axk.text(xk, 0.1, f"{vk:g}", ha="center", va="bottom", fontsize=4.8, color=INK2)
axleg = axmm(fig, HX, 27 + hh + 17, HW, 3)
axleg.axis("off")
axleg.legend(handles=[Line2D([], [], marker="s", color=c, lw=0, ms=3.6, label=d) for d, c, _ in DOMAINS], loc="center",
             ncol=6, fontsize=5.3, handletextpad=0.2, columnspacing=1.2, borderaxespad=0)
save(fig, "Fig3_biomarker_profile")
plt.show()
LEGENDS["Fig3_biomarker_profile"] = (
    "**Figure 3 | Biomarker profiles of the six groups.** Survey-weighted standardized mean difference (SMD) of each of "
    f"{nc} routine clinical and laboratory measures versus the reference group, after survey-weighted adjustment for age, sex "
    "and race and ethnicity. Top, SMN group with 95% CI; colored strip, biomarker domain; bubble heatmap, all groups, "
    "biomarkers ordered by the SMN group (color, SMD; circle area, |SMD|; outlined, 95% CI excludes 0); right, number of "
    "biomarkers whose 95% CI excludes 0. CIs are bias-corrected percentile intervals from the pipeline bootstrap (Fig. 2). SMN, symptomatic, "
    "monofilament-normal.")

# %% [markdown]
# ## Figure 4 — Mortality

# %%
fig = newfig(124)
ax = axmm(fig, 26, 8, 78, 54)
for g in ORDER:
    k = KM[(KM.group == g) & (KM.t <= 17)]
    if g in ("D_hidden", "B_painless_PN", "C_painful_PN", "A_reference"):
        ax.fill_between(k.t, k.lo, k.hi, color=GC[g], alpha=0.12, lw=0)
    ax.plot(k.t, k.surv, color=GC[g], lw=1.5 if g == "D_hidden" else 0.9, ls=GLS[g], label=NAME[g])
ax.set_xlim(0, 17)
ax.set_ylim(0.35, 1.01)
ax.set_xticks([0, 5, 10, 15])
ax.set_ylabel("Survival probability (survey-weighted)")
ax.set_xlabel("Years since examination")
ax.yaxis.grid(True, color=GRID, lw=0.4)
ax.set_axisbelow(True)
ax.legend(loc="lower left", fontsize=5.3, handlelength=2.2, ncol=2, columnspacing=1.0)
ptitle(ax, "Kaplan–Meier survival")
letter(fig, 0, 2, "a")
axr = axmm(fig, 26, 76, 78, 26)
axr.set_xlim(0, 17)
axr.set_ylim(-0.5, len(ORDER) - 0.5)
axr.axis("off")
for i, g in enumerate(ORDER):
    y = len(ORDER) - 1 - i
    axr.text(-1.4, y, NAME[g], ha="right", va="center", fontsize=5.4,
             color=C_HID_TXT if g == "D_hidden" else (GC[g] if g != "A_reference" else INK),
             fontweight="bold" if g == "D_hidden" else "normal")
    for yr in [0, 5, 10, 15]:
        n_ = KR[(KR.group == g) & (KR.year == yr)].at_risk
        axr.text(yr, y, f"{int(n_.iloc[0]):,}" if len(n_) else "", ha="center", va="center", fontsize=5.4)
    kn = H12["km_n"][g]
    axr.text(19.4, y, f"{kn['deaths']:,} / {kn['n']:,}", ha="right", va="center", fontsize=5.4, color=INK2)
axr.text(-1.4, len(ORDER) - 0.1, "Number at risk", ha="right", va="bottom", fontsize=5.6, fontweight="bold")
axr.text(19.4, len(ORDER) - 0.1, "Deaths / n", ha="right", va="bottom", fontsize=5.6, fontweight="bold")
rows = [g for g in ORDER if g != "A_reference"]
H2 = {g: COX["main"][KEY[g]] for g in rows}
axb = axmm(fig, 146, 8, 34, 54)
for i, g in enumerate(rows):
    y = len(rows) - 1 - i
    v = H2[g]
    eye(axb, v["HR"], v["lo"], v["hi"], y - 0.12, GC[g], h=0.42, log=True, mk="s", ms=15 if g == "D_hidden" else 12)
    axb.text(-0.06, y + 0.14, NAME[g], transform=axb.get_yaxis_transform(), ha="right", va="center", fontsize=5.6,
             fontweight="bold" if g == "D_hidden" else "normal")
    axb.text(-0.06, y - 0.2, f"{fmt_ci(v['HR'], v['lo'], v['hi'])} · {v['deaths']}/{v['n']}", transform=axb.get_yaxis_transform(),
             ha="right", va="center", fontsize=5.1, color=INK2)
    axb.text(-0.06, y - 0.44, pfmt(v["p"]), transform=axb.get_yaxis_transform(), ha="right", va="center", fontsize=4.9,
             color=INK2 if v["p"] >= 0.05 else INK, style="italic")
axb.axvline(1, color=INK, lw=0.5)
logx(axb, [0.8, 1, 1.5, 2, 3], (0.75, 3.0))
axb.set_yticks([])
axb.spines["left"].set_visible(False)
axb.set_ylim(-0.55, len(rows) - 0.3)
axb.set_xlabel("Adjusted hazard ratio vs reference")
ptitle(axb, "Cox model")
letter(fig, 120, 2, "b")
causes = [("cardiovascular", "Cardio-\nvascular"), ("cancer", "Cancer"), ("other_causes", "Other\ncauses")]
HRC = LinearSegmentedColormap.from_list("hr", [C_PNL, "#F6F6F4", C_PFL])
hnorm = TwoSlopeNorm(vmin=np.log(0.5), vcenter=0, vmax=np.log(3))
axc = axmm(fig, 146, 80, 34, 30)
for i, g in enumerate(rows):
    y = len(rows) - 1 - i
    for j, (ck, cl) in enumerate(causes):
        v = COX[f"cause_{ck}"][KEY[g]]
        sig = v["lo"] > 1 or v["hi"] < 1
        lv = np.log(v["HR"])
        axc.add_patch(Rectangle((j + 0.04, y + 0.06), 0.92, 0.88, facecolor=HRC(hnorm(np.clip(lv, np.log(0.5), np.log(3)))),
                                edgecolor=INK if sig else "none", lw=0.6))
        tc = "white" if abs(lv) > 0.55 else INK
        axc.text(j + 0.5, y + 0.62, f"{v['HR']:.2f}{stars(v['p'])}", ha="center", va="center", fontsize=5.5, color=tc,
                 fontweight="bold" if sig else "normal")
        axc.text(j + 0.5, y + 0.3, f"{v['lo']:.2f}–{v['hi']:.2f}", ha="center", va="center", fontsize=4.5, color=tc)
axc.set_xlim(0, 3)
axc.set_ylim(0, len(rows))
axc.set_xticks(np.arange(3) + 0.5)
axc.set_xticklabels([f"{cl}\n({COX['cause_' + ck]['deaths']:,} deaths)" for ck, cl in causes], fontsize=5.2, linespacing=1.1)
axc.set_yticks(np.arange(len(rows)) + 0.5)
axc.set_yticklabels([NAME[g] for g in rows[::-1]], fontsize=5.6)
for t in axc.get_yticklabels():
    if t.get_text() == NAME["D_hidden"]:
        t.set_fontweight("bold")
axc.tick_params(length=0)
for sp in axc.spines.values():
    sp.set_visible(False)
ptitle(axc, "Cause-specific hazard ratio")
letter(fig, 120, 70, "c")
save(fig, "Fig4_mortality")
plt.show()
EV = H12["evalue"]
LEGENDS["Fig4_mortality"] = (
    "**Figure 4 | Mortality through 2019.** **a**, Survey-weighted Kaplan–Meier survival by group (shaded, design-based 95% "
    "CI) with unweighted numbers at risk and deaths/n. **b**, Survey-weighted Cox hazard ratios versus reference adjusted for "
    "age, sex, race and ethnicity, education, income-to-poverty ratio, smoking status, BMI, diabetes and eGFR; design-based "
    "95% CIs (Taylor linearization, design degrees of freedom); thick bars, 66% CI; deaths/n in the model sample. E-value for "
    f"the SMN group {EV['E_point']:.2f} (CI limit {EV['E_ci']:.2f}). **c**, Cause-specific hazard ratios with 95% CIs "
    "(* P < 0.05, ** P < 0.01, *** P < 0.001); tiles colored by log hazard ratio; outlined and bold, 95% CI excludes 1. "
    "SMN, symptomatic, monofilament-normal.")

# %% [markdown]
# ## Supplementary Figure S1 — Sensitivity analyses as specification curves

# %%
EXL = {"sciatica": "Excl. sciatica", "lowb12": "Excl. low B12", "vascular": "Excl. PAD / claudication",
       "comorb": "Excl. arthritis, cancer, alcohol, CKD", "urate": "Excl. high urate / gout drugs",
       "neurotox": "Excl. neurotoxic drugs", "hcv": "Excl. hepatitis C", "thyroid": "Excl. thyroid disease"}
COMBO = {"sciatica_lowb12_vascular": ["sciatica", "lowb12", "vascular"], "old_all": ["sciatica", "lowb12", "vascular", "comorb"],
         "new": ["urate", "neurotox", "hcv", "thyroid"], "everything": list(EXL)}


def flags_for(name):
    if name in EXL:
        return [EXL[name]]
    if name in COMBO:
        return [EXL[x] for x in COMBO[name]]
    return {"nogly": ["No glycemic features"], "nodm": ["Adults without diabetes"], "heldout": ["Held-out cycle 2003–04"],
            "lr": ["Logistic regression (L2)"], "en": ["Elastic net"], "corroborated": ["Corroborated symptoms"],
            "uniform": ["Single-fold scoring"], "main": []}.get(name, [])


def dspec(name):
    v = CX[name]["D_minus_E"]
    return dict(name="Main analysis" if name == "main" else name, est=v["est"], lo=v["lo"], hi=v["hi"], p=v["p"],
                flags=set(flags_for(name)))


def hspec(name, extra_flags=None, ctrl=True):
    v = COX[name]
    return dict(name="Main analysis" if name == "main" else name, est=v["D"]["HR"], lo=v["D"]["lo"], hi=v["D"]["hi"],
                p=v["D"]["p"], ctrl=v["E"] if ctrl else None, flags=set(extra_flags if extra_flags is not None else flags_for(name)))


bio_specs = [dspec(n) for n in ["main", "uniform", "nogly", "nodm", "heldout", "corroborated", "sciatica", "lowb12", "vascular",
                                "comorb", "urate", "neurotox", "hcv", "thyroid", "sciatica_lowb12_vascular", "old_all", "new",
                                "everything", "lr", "en"]]
EXN = list(EXL.values())
bio_cats = [("Learner", ["Logistic regression (L2)", "Elastic net"]), ("Scoring", ["Single-fold scoring"]),
            ("Features", ["No glycemic features"]), ("Sample", ["Adults without diabetes", "Held-out cycle 2003–04"]),
            ("Case definition", ["Corroborated symptoms"]), ("Exclusions", EXN)]
mort_specs = ([hspec("main"), hspec("nodm", ["Adults without diabetes"]), hspec("excl2y", ["Excl. deaths < 2 y"]),
               hspec("p1999_2002", ["Cycles 1999–2002 only"]), hspec("p2003_2004", ["Cycle 2003–04 only"]),
               hspec("corroborated")] +
              [hspec(n) for n in ["sciatica", "lowb12", "vascular", "comorb", "urate", "neurotox", "hcv", "thyroid",
                                  "sciatica_lowb12_vascular", "old_all", "new", "everything"]] +
              [hspec("adj_abi_claud_cvd", ["+ ABI, claudication, CVD"], ctrl=False),
               hspec("normal_circulation_only", ["Normal circulation only"], ctrl=False)])
mort_cats = [("Sample", ["Adults without diabetes", "Cycles 1999–2002 only", "Cycle 2003–04 only", "Excl. deaths < 2 y"]),
             ("Case definition", ["Corroborated symptoms"]),
             ("Exclusions", [e for e in EXN if e != "Excl. PAD / claudication"]),
             ("Circulation", ["Excl. PAD / claudication", "+ ABI, claudication, CVD", "Normal circulation only"])]
VASC = {"Excl. PAD / claudication", "+ ABI, claudication, CVD", "Normal circulation only"}


def spec_curve(fig, x, top, w, hc, specs, cats, color, null, log, ylab, ylim, yticks, title, row_h=3.15):
    order = sorted(range(len(specs)), key=lambda k: specs[k]["est"])
    n = len(specs)
    axc = axmm(fig, x, top, w, hc)
    nsig = 0
    for j, k in enumerate(order):
        s_ = specs[k]
        main = s_["name"] == "Main analysis"
        vas = bool(s_["flags"] & VASC)
        sig = s_["lo"] > null or s_["hi"] < null
        nsig += sig
        if main:
            axc.axvspan(j - 0.5, j + 0.5, color="#f0f0f0", lw=0, zorder=0)
        e = s_.get("ctrl")
        if e is not None:
            interval(axc, e["HR"], e["lo"], e["hi"], j + 0.3, GC["E_backpain_ctrl"], log=log, ms=5, thick=1.0, vertical=True,
                     z=2)
        interval(axc, s_["est"], s_["lo"], s_["hi"], j - (0.08 if e is not None else 0), color, log=log,
                 mk="D" if main else "o", ms=14 if main else 10, hollow=not sig, thick=2.2, vertical=True)
        if vas:
            axc.scatter(j - (0.08 if e is not None else 0), ylim[0] * (ylim[1] / ylim[0]) ** 0.035 if log else
                        ylim[0] + (ylim[1] - ylim[0]) * 0.035, marker="^", s=9, color=INK, lw=0, zorder=4)
    axc.plot([j - (0.08 if specs[k].get("ctrl") is not None else 0) for j, k in enumerate(order)],
             [specs[k]["est"] for k in order], color=tint(color, 0.45), lw=0.6, zorder=2)
    axc.axhline(null, color=INK, lw=0.5, zorder=1)
    axc.axhline(specs[0]["est"], color=color, lw=0.5, ls=(0, (2, 2)), zorder=1)
    if log:
        axc.set_yscale("log")
        axc.minorticks_off()
    axc.set_ylim(*ylim)
    axc.set_yticks(yticks)
    axc.set_yticklabels([f"{t:g}" for t in yticks])
    axc.set_xlim(-0.6, n - 0.4)
    axc.set_xticks([])
    axc.spines["bottom"].set_visible(False)
    axc.set_ylabel(ylab)
    axc.text(0.02, 0.97, f"{nsig}/{n} intervals exclude {null:g}", transform=axc.transAxes, ha="left", va="top", fontsize=5.3,
             color=INK2)
    ptitle(axc, title)
    # choice matrix
    lines = []
    for cat, opts in cats:
        lines.append(("h", cat))
        lines += [("o", o) for o in opts]
    hm = row_h * len(lines)
    axp = axmm(fig, x, top + hc + 2.0, w, 6.5)
    axp.set_xlim(-0.6, n - 0.4)
    axp.set_ylim(0, 1)
    axp.axis("off")
    for j, k in enumerate(order):
        pk = specs[k]["p"]
        axp.text(j, 0.5, pshort(pk), rotation=90, ha="center", va="center", fontsize=4.6,
                 color=INK if (pk is not None and pk < 0.05) else INK3, fontweight="semibold" if (pk is not None and pk < 0.05) else "normal")
    axp.text(-0.012, 0.5, "P value", transform=axp.transAxes, ha="right", va="center", fontsize=5.3, fontweight="bold")
    axm = axmm(fig, x, top + hc + 9.5, w, hm)
    axm.set_xlim(-0.6, n - 0.4)
    axm.set_ylim(-0.5, len(lines) - 0.5)
    axm.axis("off")
    jm = [specs[k]["name"] for k in order].index("Main analysis")
    axm.axvspan(jm - 0.5, jm + 0.5, color="#f0f0f0", lw=0, zorder=0)
    for r, (kind, lab) in enumerate(lines):
        yy = len(lines) - 1 - r
        if kind == "h":
            axm.text(-0.012, yy, lab, transform=axm.get_yaxis_transform(), ha="right", va="center", fontsize=5.4,
                     fontweight="bold")
            axm.plot([-0.6, n - 0.4], [yy + 0.5, yy + 0.5], color=GRID, lw=0.5, zorder=0) if r else None
            continue
        axm.text(-0.012, yy, lab, transform=axm.get_yaxis_transform(), ha="right", va="center", fontsize=5.3,
                 color=C_PFL if lab in VASC else INK2)
        for j, k in enumerate(order):
            on = lab in specs[k]["flags"]
            axm.scatter(j, yy, s=10 if on else 2.5, color=(C_PFL if lab in VASC else INK) if on else "#cfcfcf", lw=0,
                        zorder=2)
    return axc, axm


fig = newfig(138)
spec_curve(fig, 52, 8, 40, 40, bio_specs, bio_cats, C_HID, 0, False, "SMN − back pain only (logit)", (-0.12, 0.6),
           [0, 0.2, 0.4], "Transfer-model contrast")
spec_curve(fig, 140, 8, 40, 40, mort_specs, mort_cats, C_HID, 1, True, "Hazard ratio, SMN vs reference", (0.55, 6.0),
           [0.6, 1, 1.5, 2, 3, 5], "All-cause mortality")
axlg = axmm(fig, 20, 131, 160, 4)
axlg.axis("off")
axlg.legend(handles=[Line2D([], [], color=INK2, lw=2.2, marker="o", ms=3.4, mec="white", label="SMN, 95% CI excludes null"),
                     Line2D([], [], color=INK2, lw=2.2, marker="o", ms=3.4, mfc="white", label="95% CI includes null"),
                     Line2D([], [], color=INK2, lw=0, marker="D", ms=3.2, label="Main analysis"),
                     Line2D([], [], color=GC["E_backpain_ctrl"], lw=1.0, marker="o", ms=2.4, mec="white", label="Back pain only"),
                     Line2D([], [], color=INK, lw=0, marker="^", ms=3, label="Circulation-related analysis")],
            loc="center", ncol=5, fontsize=5.2, handlelength=1.4, columnspacing=1.2, borderaxespad=0)
letter(fig, 0, 2, "a")
letter(fig, 96, 2, "b")
save(fig, "FigS1_sensitivity")
plt.show()
LEGENDS["FigS1_sensitivity"] = (
    "**Supplementary Figure S1 | Sensitivity analyses as specification curves.** Each column is one analysis, ordered by its "
    "estimate; the dot matrix below shows which analytic choices it makes (shaded column, main analysis). **a**, SMN minus back "
    "pain only in the transfer-model score; models are refitted within each restricted sample. **b**, Adjusted hazard ratio for "
    "SMN versus reference with back pain only alongside (grey); red choices mark analyses that exclude, adjust for or restrict "
    "on peripheral circulation (also marked by triangles). All intervals are design-based (Taylor linearization with design "
    "degrees of freedom; scores treated as fixed in a). Thick and thin bars, 66% and 95% CIs; hollow symbols, 95% CI includes "
    "the null; dashed line, main estimate; P values are printed between the curve and the choice matrix. High urate, serum "
    "urate >6.8 mg/dL or a urate-lowering or gout medication; neurotoxic drugs, a prescription in the past 30 days for an agent "
    "listed in Supplementary Table S2; hepatitis C, positive confirmed antibody; thyroid disease, self-reported thyroid disease, "
    "thyroid medication or a thyroid-stimulating hormone outside 0.4–4.5 mIU/L (measured in 1999–2002 only).")

# %% [markdown]
# ## Supplementary Figure S2 — Cosine similarity of biomarker profiles, reference range and leave-one-domain-out

# %%
fig = newfig(82)
# a — cosine matrix with bootstrap CIs
cos_order = ["D_hidden", "B_painless_PN", "C_painful_PN", "F_hands_only", "E_backpain_ctrl"]
short = {"D_hidden": "SMN", "B_painless_PN": "Insensate,\nno symptoms", "C_painful_PN": "Insensate\nwith symptoms",
         "F_hands_only": "Hand\nsymptoms", "E_backpain_ctrl": "Back pain"}


def ckey(a, b):
    a, b = sorted([KEY[a], KEY[b]])
    return f"cos_{a}{b}"


axd = axmm(fig, 22, 10, 50, 50)
cmap_c = LinearSegmentedColormap.from_list("c", ["#F2F7F6", "#A9D8CF", "#00A087", "#005C4F"])
for i in range(len(cos_order)):
    for j in range(i):
        e, lo, hi = bstat(ckey(cos_order[i], cos_order[j]))
        yy_ = len(cos_order) - 1 - i
        axd.add_patch(Rectangle((j, yy_), 1, 1, facecolor=cmap_c(np.clip((e - 0.3) / 0.7, 0, 1)), edgecolor="white", lw=0.8))
        tc = "white" if e > 0.8 else INK
        axd.text(j + 0.5, yy_ + 0.6, f"{e:.2f}", ha="center", va="center", fontsize=5.8, color=tc, fontweight="semibold")
        axd.text(j + 0.5, yy_ + 0.3, ci_txt(e, lo, hi)[ci_txt(e, lo, hi).index("(") + 1:-1], ha="center", va="center", fontsize=4.4, color=tc)
axd.add_patch(Rectangle((0, 0), 1, len(cos_order) - 1, fill=False, edgecolor=INK, lw=0.8, zorder=5))
axd.set_xlim(0, len(cos_order) - 1)
axd.set_ylim(0, len(cos_order) - 1)
axd.set_xticks(np.arange(len(cos_order) - 1) + 0.5)
axd.set_xticklabels([short[g] for g in cos_order[:-1]], fontsize=5.2, linespacing=1.0)
axd.set_yticks(np.arange(len(cos_order) - 1) + 0.5)
axd.set_yticklabels([short[g] for g in cos_order[1:][::-1]], fontsize=5.2, linespacing=1.0)
axd.tick_params(length=0)
for sp in axd.spines.values():
    sp.set_visible(False)
ptitle(axd, "Profile cosine similarity (95% CI)")
letter(fig, 0, 2, "a")

# b — reference range: insensate-without-symptoms profile vs SMN, controls and eight unrelated conditions
CN = {"cond_asthma": "Asthma", "cond_arthritis": "Arthritis", "cond_emphysema": "Emphysema",
      "cond_chronic_bronchitis": "Chronic bronchitis", "cond_liver_condition": "Liver condition", "cond_cancer": "Cancer",
      "cond_osteoporosis": "Osteoporosis", "cond_thyroid_disease": "Thyroid disease"}
CC = H11["condition_counts"]
items = [(f"{NAME[g]}", ckey("B_painless_PN", g), GC[g], NG[g]) for g in ["D_hidden", "C_painful_PN", "F_hands_only",
                                                                         "E_backpain_ctrl"]]
conds = sorted(CN, key=lambda c: -H13[f"cos_B_{c}"]["est"])
items += [(CN[c], f"cos_B_{c}", INK3, CC[c[5:]]["n_normal_exam_no_foot_sx"]) for c in conds]
axr = axmm(fig, 104, 10, 34, 58)
rng_lo = min(H13[f"cos_B_{c}"]["est"] for c in CN)
rng_hi = max(H13[f"cos_B_{c}"]["est"] for c in CN)
axr.axvspan(rng_lo, rng_hi, color="#efefef", lw=0, zorder=0)
for i, (lab, key, c, n_) in enumerate(items):
    y = len(items) - 1 - i - (0.6 if i >= 4 else 0)
    e, lo, hi = bstat(key)
    axr.plot([lo, hi], [y, y], color=c, lw=0.8, zorder=2)
    axr.scatter(e, y, s=14 if i == 0 else 10, color=c, edgecolor="white", lw=0.5, zorder=3, marker="D" if i == 0 else "o")
    axr.text(-0.04, y, lab, transform=axr.get_yaxis_transform(), ha="right", va="center", fontsize=5.3,
             fontweight="bold" if i == 0 else "normal")
    axr.text(1.03, y, ci_txt(e, lo, hi), transform=axr.get_yaxis_transform(), ha="left", va="center", fontsize=4.9)
    axr.text(1.62, y, f"{n_:,}", transform=axr.get_yaxis_transform(), ha="right", va="center", fontsize=4.9, color=INK2)
axr.text(1.03, len(items) - 0.4, "Cosine (95% CI)", transform=axr.get_yaxis_transform(), ha="left", va="bottom", fontsize=5.0,
         fontweight="semibold")
axr.text(1.62, len(items) - 0.4, "n", transform=axr.get_yaxis_transform(), ha="right", va="bottom", fontsize=5.0,
         fontweight="semibold")
axr.set_yticks([])
axr.spines["left"].set_visible(False)
axr.set_ylim(-1.2, len(items) - 0.3)
axr.set_xlim(-0.35, 1.0)
axr.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
axr.axvline(0, color=INK, lw=0.4)
axr.set_xlabel("Cosine similarity with the profile of\ninsensate adults without symptoms")
ptitle(axr, "Reference range from unrelated conditions", pad=12)
letter(fig, 76, 2, "b")
save(fig, "FigS2_similarity")
plt.show()
dd = H13["cosdiff_DB_minus_DE"]
LEGENDS["FigS2_similarity"] = (
    "**Supplementary Figure S2 | Similarity of biomarker profiles.** Each profile is the vector of survey-weighted, age-, "
    f"sex- and race-adjusted SMDs of the {nc} biomarkers in Fig. 3. **a**, Cosine similarity between group profiles with "
    "bootstrap 95% CIs (outlined column, similarity with the SMN profile). The similarity of SMN with insensate adults without "
    f"symptoms minus its similarity with back pain only was {ci_txt(dd['est'], dd['lo'], dd['hi'])}. **b**, Cosine similarity of "
    "the insensate-without-symptoms profile with the SMN and control profiles (top) and with the profiles of eight unrelated "
    "self-reported conditions (bottom; grey band, their range). Each condition profile compares normal-exam adults without "
    "foot symptoms who reported the condition with reference adults who did not; n, adults with the condition. Smaller groups "
    "give noisier profiles, which lowers their cosine similarity. 95% CIs are bias-corrected percentile intervals from the "
    "pipeline bootstrap.")

# %% [markdown]
# ## Supplementary Table S3 data — leave-one-domain-out similarity (printed for the supplement)

# %%
loo = []
for dom, _, _ in DOMAINS:
    a = H13[f"loo_DB_{dom}"]
    b = H13[f"loo_DE_{dom}"]
    loo.append(dict(domain_left_out=dom, SMN_vs_insensate=ci_txt(a["est"], a["lo"], a["hi"]),
                    SMN_vs_back_pain=ci_txt(b["est"], b["lo"], b["hi"])))
loo = pd.DataFrame(loo)
loo.to_csv("figures/loo_domains.csv", index=False)
print(loo.to_string(index=False))

# %%
with open("figures/figure_legends.md", "w") as fh:
    fh.write("\n\n".join(LEGENDS.values()) + "\n")
print(sorted(os.listdir("figures")))
