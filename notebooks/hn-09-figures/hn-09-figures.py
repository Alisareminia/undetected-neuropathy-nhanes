# %% [markdown]
# # HN-09 — Figures, journal palette edition (plot-only, reads saved results from HN-01…HN-07)
# Same panels as HN-08, restyled: muted Nature (NPG) palette — teal hidden, navy painless, brick red painful, neutral
# greys/sand for controls (every group pair ΔE ≥ 15 under protan, deutan and tritan simulation), Source Sans 3,
# gradient-shaded confidence distributions, bubble heatmap, offset axes.
# 1 Study concept and data (schematics) · 2 ML transfer test · 3 Biomarker fingerprint · 4 PU ranking validation ·
# 5 Mortality · 6 Sensitivity analyses.
# Every panel is placed on an explicit millimetre grid (`axmm`) so nothing collides; explanatory text lives in
# figures/figure_legends.md, not inside the figures.
# Group palette: hidden #00897B · painless #3C5488 · painful #D6492F · reference #3A3F47 · back pain #B09C85 · hands #8FA0C2.
# Biomarker domains use muted earth tones so they read as annotation, not as groups.

# %%
import glob
import json
import os
import urllib.request

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path
from scipy import ndimage
from scipy import stats as sst
from scipy.stats import gaussian_kde

try:
    from adjustText import adjust_text
except ImportError:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "adjustText"], check=False)
    from adjustText import adjust_text


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


R1, R3, R4, R5, R6 = (json.load(open(find(f))) for f in
                          ["hn_results.json", "hn_strengthen_results.json", "hn_fixes_results.json",
                           "hn_pu_results.json", "hn_pu2_results.json"])
KM = pd.read_csv(find("km_curves.csv"))
KR = pd.read_csv(find("km_at_risk.csv"))
KMS = pd.read_csv(find("km_summary.csv"), index_col=0)
FP = pd.read_csv(find("fingerprint.csv"))
COS = pd.read_csv(find("fingerprint_cosine.csv"), index_col=0)
DEC = pd.read_csv(find("pu_deciles.csv"))
EMB = pd.read_csv(find("embedding.csv"))
SC = pd.read_parquet(find("hn_scores.parquet"))
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
        "D_hidden": "Hidden neuropathy", "B_painless_PN": "Painless neuropathy", "C_painful_PN": "Painful neuropathy"}
NM = {"waist": "Waist", "bmi": "BMI", "weight": "Weight", "height": "Height", "sbp": "Systolic BP", "dbp": "Diastolic BP",
      "hba1c": "HbA1c", "glucose": "Glucose", "egfr": "eGFR", "albumin": "Albumin", "alp": "ALP", "ast": "AST", "alt": "ALT",
      "bun": "BUN", "calcium": "Calcium", "chol": "Cholesterol", "bicarb": "Bicarbonate", "creat": "Creatinine", "ggt": "GGT",
      "iron": "Iron", "ldh": "LDH", "phos": "Phosphorus", "bili": "Bilirubin", "protein": "Total protein", "trig": "Triglycerides",
      "uric": "Uric acid", "sodium": "Sodium", "potassium": "Potassium", "chloride": "Chloride", "wbc": "WBC",
      "lymph_pct": "Lymphocytes %", "mono_pct": "Monocytes %", "neut_pct": "Neutrophils %", "eos_pct": "Eosinophils %",
      "hgb": "Haemoglobin", "mcv": "MCV", "rdw": "RDW", "platelets": "Platelets", "mpv": "MPV", "crp": "CRP",
      "dm_self": "Diabetes (self-report)"}
NG = SC.group.value_counts().to_dict()
DOMAINS = [("Glucose & lipids", "#D1A32C", ["hba1c", "glucose", "trig", "chol", "uric"]),
           ("Body size", "#9A4F8F", ["waist", "bmi", "weight", "height"]),
           ("Blood pressure", "#EBA7BB", ["sbp", "dbp"]),
           ("Liver & protein", "#5F8F3E", ["alp", "ast", "alt", "ggt", "bili", "albumin", "protein", "ldh"]),
           ("Kidney & electrolytes", "#7E8C96", ["egfr", "bun", "creat", "potassium", "sodium", "chloride", "bicarb",
                                                 "calcium", "phos"]),
           ("Blood count & inflammation", "#8A5A3E", ["wbc", "lymph_pct", "mono_pct", "neut_pct", "eos_pct", "hgb", "mcv",
                                                      "rdw", "platelets", "mpv", "crp", "iron"])]
DOMC = {f: c for _, c, fs in DOMAINS for f in fs}
S2 = R3["S2"]
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

# %% [markdown]
# ## Figure 1 — Study concept and data (schematics)

# %%
fig = newfig(160)

# a — foot schematic with a skin cross-section inset
axf = axmm(fig, 0, 6, 46, 84)
axf.set_xlim(-3.0, 11.5)
axf.set_ylim(-3.8, 22)
axf.set_aspect("equal")
axf.axis("off")
sole = [(5.0, 0.6), (6.9, 1.3), (7.8, 3.6), (7.6, 7.5), (8.2, 11.5), (8.9, 14.6), (8.6, 16.6), (7.2, 17.4), (5.2, 17.8),
        (3.2, 17.6), (1.7, 16.4), (1.5, 13.6), (2.1, 10.2), (2.0, 6.3), (2.6, 2.6), (3.6, 1.0), (5.0, 0.6)]
pts = np.array(sole)
mids = (pts[:-1] + pts[1:]) / 2
verts, codes = [tuple(mids[0])], [Path.MOVETO]
for i in range(1, len(mids)):
    verts += [tuple(pts[i]), tuple(mids[i])]
    codes += [Path.CURVE3, Path.CURVE3]
verts += [tuple(pts[0]), tuple(mids[0])]
codes += [Path.CURVE3, Path.CURVE3]
FOOT = "#f3f0ea"
axf.add_patch(PathPatch(Path(verts, codes), facecolor=FOOT, edgecolor=INK2, lw=0.8, zorder=1))
for x, y, r in [(3.0, 19.4, 1.05), (5.0, 19.9, 0.72), (6.45, 19.6, 0.62), (7.6, 19.0, 0.55), (8.45, 18.1, 0.48)]:
    axf.add_patch(Circle((x, y), r, facecolor=FOOT, edgecolor=INK2, lw=0.8, zorder=1))
for _ in range(40):
    x0, y0 = rng.uniform(2.8, 7.4), rng.uniform(2.5, 15.5)
    ang = rng.uniform(0, 2 * np.pi)
    xe, ye = x0 + 0.55 * np.cos(ang), y0 + 0.55 * np.sin(ang)
    xm, ym = (x0 + xe) / 2 + 0.12 * np.sin(ang), (y0 + ye) / 2 - 0.12 * np.cos(ang)
    axf.plot([x0, xm, xe], [y0, ym, ye], color=C_HID, lw=0.55, alpha=0.8, zorder=2)
    axf.scatter(xe, ye, s=1.4, color=C_HID, lw=0, zorder=2)
for (x, y) in [(3.0, 19.4), (3.1, 15.8), (7.7, 15.0)]:
    axf.scatter(x, y, s=46, facecolor="white", edgecolor=INK, lw=1.0, zorder=5)
    axf.scatter(x, y, s=8, color=INK, zorder=6)
for xy, txt, xyt, ha, rp in [((3.0, 19.4), "Hallux", (-0.6, 21.6), "right", (1, 0.5)),
                             ((3.1, 15.8), "1st metatarsal\nhead", (-0.4, 12.6), "right", (1, 0.5)),
                             ((7.7, 15.0), "5th metatarsal\nhead", (9.2, 11.4), "left", (0, 0.5))]:
    axf.annotate(txt, xy, xyt, fontsize=5.6, color=INK, ha=ha, va="center", linespacing=1.1,
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2, shrinkA=1, shrinkB=3, relpos=rp))
axf.add_patch(Rectangle((8.6, 3.2), 2.7, 3.0, facecolor="#faf3ec", edgecolor=INK2, lw=0.5, zorder=1))
axf.plot([8.6, 11.3], [5.6, 5.6], color=INK3, lw=0.4, zorder=2)
for xx in np.linspace(8.9, 11.0, 6):
    axf.plot([xx, xx + 0.1, xx - 0.05], [4.1, 5.0, 6.0], color=C_HID, lw=0.5, zorder=3)
axf.plot([8.6, 11.3], [3.7, 3.7], color=INK, lw=1.5, zorder=3)
axf.text(9.95, 6.45, "Skin section", ha="center", va="bottom", fontsize=5.1, color=INK2)
axf.text(9.95, 2.85, "Large fibre", ha="center", va="top", fontsize=5.1, color=INK2)
axf.scatter(-2.2, -1.8, s=26, facecolor="white", edgecolor=INK, lw=0.9)
axf.scatter(-2.2, -1.8, s=4, color=INK)
axf.text(-1.3, -1.8, "Monofilament site (large fibres)", fontsize=5.5, va="center")
axf.plot([-2.6, -1.8], [-3.2, -3.2], color=C_HID, lw=0.8)
axf.text(-1.3, -3.2, "Small-fibre endings (pain, tingling)", fontsize=5.5, va="center")
letter(fig, 0, 0, "a")
ftext(fig, 5, 0, "Examination vs symptoms", fontsize=6.5, fontweight="bold", va="top")

# b — classification schematic (exam × foot symptoms)
axb = axmm(fig, 50, 6, 62, 84)
axb.set_xlim(0, 10)
axb.set_ylim(0, 12)
axb.axis("off")
letter(fig, 49, 0, "b")
ftext(fig, 54, 0, "Group definitions (NHANES 1999–2004, ≥40 y)", fontsize=6.5, fontweight="bold", va="top")
X0, XM, X1 = 2.1, 6.1, 10.0
YT, YM, YB = 10.6, 6.6, 0.3
axb.text((X0 + XM) / 2, YT + 0.35, "Monofilament normal", ha="center", va="bottom", fontsize=5.9, fontweight="bold")
axb.text((XM + X1) / 2, YT + 0.35, "≥1 insensate site", ha="center", va="bottom", fontsize=5.9, fontweight="bold")
axb.text(X0 - 0.25, (YT + YM) / 2, "Foot pain or\ntingling", ha="right", va="center", fontsize=5.9, fontweight="bold",
         linespacing=1.15)
axb.text(X0 - 0.25, (YM + YB) / 2, "No foot\nsymptoms", ha="right", va="center", fontsize=5.9, fontweight="bold", linespacing=1.15)


def cell(ax, x, y, w, h, g, hero=False, note=None):
    c = GC[g]
    box(ax, x, y, w, h, fc=tint(c, 0.14), ec=c, lw=1.3 if hero else 0.6, r=0.18)
    s = S2[g]
    ax.text(x + 0.22, y + h - 0.22, NAME[g], fontsize=5.9, fontweight="bold", va="top", color=INK)
    ax.text(x + 0.22, y + h - 0.7, f"n = {NG.get(g, 0):,}", fontsize=5.2, va="top", color=INK2)
    ax.text(x + 0.22, y + h - 1.04, f"{s['prevalence'] * 100:.1f}% ({s['lo'] * 100:.1f}–{s['hi'] * 100:.1f})", fontsize=5.2,
            va="top", color=INK2)
    if note:
        ax.text(x + 0.22, y + 0.12, note, fontsize=5.1, va="bottom", color=INK2, style="italic")


g_ = 0.12
cell(axb, X0 + g_, YM + g_, XM - X0 - 2 * g_, YT - YM - 2 * g_, "D_hidden", hero=True, note="Study group")
cell(axb, XM + g_, YM + g_, X1 - XM - 2 * g_, YT - YM - 2 * g_, "C_painful_PN", note="Exam-confirmed")
cell(axb, XM + g_, YB + g_, X1 - XM - 2 * g_, YM - YB - 2 * g_, "B_painless_PN", note="ML training positives")
hsub = (YM - YB - 2 * g_) / 3
for k, (g, nt) in enumerate([("A_reference", "ML training negatives"), ("E_backpain_ctrl", "Negative control"),
                             ("F_hands_only", "Negative control")]):
    cell(axb, X0 + g_, YM - g_ - (k + 1) * hsub + 0.06, XM - X0 - 2 * g_, hsub - 0.12, g, note=nt)
axb.plot([X0, X1], [YM, YM], color=INK3, lw=0.4, ls=(0, (2, 2)), zorder=0)
axb.plot([XM, XM], [YB, YT], color=INK3, lw=0.4, ls=(0, (2, 2)), zorder=0)

# c — ML transfer-test schematic
axc = axmm(fig, 118, 6, 65, 84)
axc.set_xlim(0, 10)
axc.set_ylim(0, 12)
axc.axis("off")
letter(fig, 116, 0, "c")
ftext(fig, 121, 0, "Machine-learning transfer test", fontsize=6.5, fontweight="bold", va="top")
nfeat = FP.feature.nunique()
box(axc, 0.4, 9.75, 9.2, 1.45, fc="#f5f5f5", ec=INK2)
axc.text(5, 10.47, f"Routine examination and laboratory data\n{nfeat} biomarkers + age, sex, race", ha="center", va="center",
         fontsize=5.6, linespacing=1.3)
arrow(axc, (5, 9.7), (5, 9.05))
box(axc, 0.4, 7.2, 9.2, 1.8, fc="white", ec=INK)
axc.text(5, 8.6, "Gradient boosting, 5-fold cross-fitted", ha="center", va="center", fontsize=5.8, fontweight="bold")
axc.scatter([1.3], [7.75], s=20, color=C_PNL, zorder=3)
axc.text(1.6, 7.75, "Painless neuropathy", fontsize=5.4, va="center")
axc.text(5.25, 7.75, "vs", fontsize=5.4, va="center", color=INK2)
axc.scatter([6.0], [7.75], s=20, color=GC["A_reference"], zorder=3)
axc.text(6.3, 7.75, "Reference", fontsize=5.4, va="center")
arrow(axc, (5, 7.15), (5, 6.5))
box(axc, 0.4, 4.15, 9.2, 2.3, fc="white", ec=INK)
axc.text(5, 6.05, "Score groups never seen in training", ha="center", va="center", fontsize=5.8, fontweight="bold")
for k, g in enumerate(["D_hidden", "E_backpain_ctrl", "F_hands_only"]):
    yy = 5.42 - k * 0.42
    axc.scatter([1.1], [yy], s=13, color=GC[g], zorder=3)
    axc.text(1.4, yy, NAME[g] + ("" if g == "D_hidden" else " (control)"), fontsize=5.4, va="center")
for k, (t1, t2) in enumerate([("Biological\nsimilarity", "Figs 2–3"), ("PU ranking of\nnormal-exam adults", "Fig. 4"),
                              ("17-year\nmortality", "Figs 5–6")]):
    xc = 1.75 + k * 3.25
    arrow(axc, (5, 4.1), (xc, 3.42), lw=0.5)
    box(axc, xc - 1.5, 1.35, 3.0, 2.02, fc="#f5f5f5", ec=INK2)
    axc.text(xc, 2.55, t1, ha="center", va="center", fontsize=5.4, linespacing=1.15)
    axc.text(xc, 1.65, t2, ha="center", va="center", fontsize=5.1, color=INK2, style="italic")
axc.text(5, 0.75, f"Out-of-fold AUROC, painless vs reference: {R1['H1_full']['auroc_B_vs_A']:.2f} "
                  f"(held-out cycle {R3['S1']['test_auroc_2003_04']:.2f})", ha="center", va="center", fontsize=5.1, color=INK2,
         style="italic")

# d — cohort flow over calendar time: examination (1999–2004) → vital status on 31 Dec 2019
axd = axmm(fig, 52, 104, 100, 46)
letter(fig, 0, 98, "d")
ftext(fig, 5, 98, "From examination to vital status (linked follow-up)", fontsize=6.5, fontweight="bold", va="top")
flow = ["C_painful_PN", "B_painless_PN", "D_hidden", "F_hands_only", "E_backpain_ctrl", "A_reference"]
ntot = float(KMS.loc[flow, "n"].sum())
dtot = float(KMS.loc[flow, "deaths"].sum())
GAP = 0.014
sc_ = (1 - GAP * (len(flow) - 1)) / ntot
XB0, XB1, XO0, XO1 = 1999.0, 2004.9, 2017.6, 2019.95
ytop, dcur, acur = 1.0, 1.0, 1.0 - dtot * sc_ - GAP * (len(flow) - 1)
lab_items = []
for g in flow:
    n_, dd = float(KMS.loc[g, "n"]), float(KMS.loc[g, "deaths"])
    hg, hd = n_ * sc_, dd * sc_
    c = GC[g]
    axd.add_patch(Rectangle((XB0, ytop - hd), XB1 - XB0, hd, facecolor=tint(c, 0.85), edgecolor="none", zorder=2))
    axd.add_patch(Rectangle((XB0, ytop - hg), XB1 - XB0, hg - hd, facecolor=tint(c, 0.3), edgecolor="none", zorder=2))
    axd.add_patch(Rectangle((XB0, ytop - hg), XB1 - XB0, hg, fill=False, edgecolor=c, lw=0.6 if g != "D_hidden" else 1.1, zorder=3))
    smooth_ribbon(axd, XB1, XO0, ytop - hd, ytop, dcur - hd, dcur, tint(c, 0.55), z=1)
    smooth_ribbon(axd, XB1, XO0, ytop - hg, ytop - hd, acur - (hg - hd), acur, tint(c, 0.16), z=1)
    axd.add_patch(Rectangle((XO0, dcur - hd), XO1 - XO0, hd, facecolor=tint(c, 0.85), edgecolor="white", lw=0.3, zorder=2))
    axd.add_patch(Rectangle((XO0, acur - (hg - hd)), XO1 - XO0, hg - hd, facecolor=tint(c, 0.3), edgecolor="white", lw=0.3,
                            zorder=2))
    lab_items.append((XB0, ytop - hg / 2, f"{NAME[g]}   n = {int(n_):,} · {dd / n_ * 100:.0f}% died"))
    ytop -= hg + GAP
    dcur -= hd
    acur -= hg - hd
callouts(axd, lab_items, 1997.9, "right", 0.0, 0.98, 0.075, fs=5.5)
dmid = 1.0 - dtot * sc_ / 2
amid = (1.0 - dtot * sc_ - GAP * (len(flow) - 1)) / 2
axd.text(2020.3, dmid, f"Died\n{int(dtot):,} ({dtot / ntot * 100:.0f}%)", ha="left", va="center", fontsize=5.6, linespacing=1.25,
         fontweight="bold")
axd.text(2020.3, amid, f"Alive on\n31 Dec 2019\n{int(ntot - dtot):,}", ha="left", va="center", fontsize=5.6, linespacing=1.25)
for x0, x1, t, held in [(1999, 2000.95, "1999–00", False), (2001, 2002.95, "2001–02", False), (2003, 2004.9, "2003–04", True)]:
    box(axd, x0 + 0.05, 1.035, x1 - x0 - 0.1, 0.075, fc="white" if held else tint(INK, 0.75), ec=INK, lw=0.6 if held else 0.4,
        r=0.03, ls=(0, (2, 1)) if held else "-")
    axd.text((x0 + x1) / 2, 1.0725, t, ha="center", va="center", fontsize=4.9, color=INK if held else "white")
axd.text(2001.95, 1.14, "Examination cycles (development · held-out)", ha="center", va="bottom", fontsize=5.1, color=INK2,
         style="italic")
axd.add_patch(FancyArrowPatch((XB1 + 0.2, 1.0725), (XO1, 1.0725), arrowstyle="-|>,head_length=2.2,head_width=1.3", color=INK2,
                              lw=0.6, shrinkA=0, shrinkB=0))
axd.text((XB1 + XO1) / 2, 1.1, "National Death Index linkage", ha="center", va="bottom", fontsize=5.1, color=INK2, style="italic")
axd.set_xlim(1998.6, 2020.2)
axd.set_ylim(-0.02, 1.2)
axd.set_yticks([])
axd.spines["left"].set_visible(False)
axd.set_xticks([1999, 2004, 2009, 2014, 2019])
axd.set_xlabel("Calendar year")
save(fig, "Fig1_study_concept")
plt.show()
LEGENDS["Fig1_study_concept"] = (
    "**Figure 1 | Study concept and data.** **a**, The 10-g monofilament tests large-fibre sensation at three plantar sites; "
    "pain, burning and tingling are conveyed by small intraepidermal fibres (inset) that it does not assess. **b**, Group "
    "definitions by monofilament result and self-reported foot pain or tingling (DIQ120/130). n, unweighted analysis sample; "
    "%, survey-weighted prevalence among US adults ≥40 y (95% CI). **c**, Transfer test: a cross-fitted gradient-boosting model "
    "trained to separate painless neuropathy from reference adults scores groups it never saw. **d**, Cohort flow over calendar "
    "time: adults examined in NHANES 1999–2004 (1999–2002 development, 2003–04 held-out) by group (darker band, died by 2019) "
    "and their vital status on 31 Dec 2019 from National Death Index linkage; ribbon widths are proportional to numbers of adults. "
    "Biochemistry, blood count and body size were measured at the same visit.")

# %% [markdown]
# ## Figure 2 — ML transfer test

# %%
sc = SC[SC.group.isin(ORDER)].copy()
p_ = sc.score_full.clip(1e-4, 1 - 1e-4)
sc["lg"] = np.log(p_ / (1 - p_))
fig = newfig(150)

# a — ridgelines + rug of the cross-fitted score
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
    ax.scatter(vv, y - 0.27 + rng.uniform(-0.04, 0.04, len(vv)), s=0.25, color=GC[g], alpha=0.35, lw=0, rasterized=True,
               zorder=1)
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([q1, q3], [y - 0.1, y - 0.1], color=GC[g], lw=1.4, zorder=3)
    ax.scatter(q2, y - 0.1, s=8, facecolor="white", edgecolor=GC[g], lw=0.8, zorder=4)
ax.axvline(ref_med, color=INK3, lw=0.5, ls=(0, (2, 2)), zorder=0)
lab_a = [f"{NAME[g]}\nn = {NG[g]:,}" for g in ORDER[::-1]]
ylabels_left(ax, lab_a, bold=(f"{NAME['D_hidden']}\nn = {NG['D_hidden']:,}",))
ax.set_ylim(-0.45, len(ORDER) - 0.3)
ax.set_xlim(xs[0], xs[-1])
ax.set_xlabel("Neuropathy-similarity score (logit, out-of-fold)")
ptitle(ax, "Score distribution")
letter(fig, 0, 2, "a")

# b — survey-weighted differences vs reference: confidence distribution (main) + three variants
axb = axmm(fig, 122, 8, 44, 66)
axb.set_xlim(-0.1, 0.78)
axb.set_ylim(-0.55, 4.65)
axb.set_autoscale_on(False)
variants = [("No glycaemic features", R1["H1_no_glycemia"]["group_diffs"], "s"),
            ("Adults without diabetes", R1["H1_no_diabetes"]["group_diffs"], "^"),
            ("Held-out cycle 2003–04", R3["S1"]["group_diffs_2003_04"], "D")]
rows = ["E_backpain_ctrl", "F_hands_only", "D_hidden", "B_painless_PN", "C_painful_PN"]
for i, g in enumerate(rows):
    y = len(rows) - 1 - i
    if g == "D_hidden":
        axb.axhspan(y - 0.47, y + 0.53, color="#f2f2f2", lw=0, zorder=0)
    v = R1["H1_full"]["group_diffs"][g]
    eye(axb, v["diff_vs_A"], v["lo"], v["hi"], y + 0.1, GC[g], h=0.34)
    axb.text(1.04, y + 0.1, pfmt(v["p"]), transform=axb.get_yaxis_transform(), ha="left", va="center", fontsize=5.2,
             color=INK, fontweight="semibold" if v["p"] < 0.05 else "normal")
    if g == "D_hidden":
        de_ = R1["H1_full"]["D_minus_E"]
        axb.text(1.04, y - 0.2, "vs back pain:", transform=axb.get_yaxis_transform(), ha="left", va="center", fontsize=4.9,
                 color=C_HID_TXT, style="italic")
        axb.text(1.04, y - 0.4, pfmt(de_["p"]), transform=axb.get_yaxis_transform(), ha="left", va="center", fontsize=4.9,
                 color=C_HID_TXT, style="italic")
    for (lab, gd, mk), dy in zip(variants, [-0.1, -0.22, -0.34]):
        w = gd[g]
        axb.plot([w["lo"], w["hi"]], [y + dy] * 2, color=tint(GC[g], 0.6), lw=0.6, zorder=2)
        axb.scatter(w["diff_vs_A"], y + dy, s=7, marker=mk, facecolor="white", edgecolor=GC[g], lw=0.6, zorder=3)
axb.axvline(0, color=INK, lw=0.5)
ylabels_left(axb, [NAME[g] for g in rows[::-1]], bold=(NAME["D_hidden"],))
axb.set_ylim(-0.55, len(rows) - 0.35)
axb.set_xlim(-0.1, 0.78)
axb.set_xlabel("Difference vs reference (logit score)")
axb.legend(handles=[eye_handle(INK2, "Main model")] +
           [Line2D([], [], marker=mk, color=tint(INK2, 0.6), mfc="white", mec=INK2, lw=0.6, ms=3, label=l) for l, _, mk in variants],
           loc="upper right", fontsize=5.3, handletextpad=0.3, borderaxespad=0.2, labelspacing=0.35, handlelength=1.6)
ptitle(axb, "Survey-weighted contrast vs reference")
axb.text(1.04, 4.62, "Main model", transform=axb.get_yaxis_transform(), ha="left", va="bottom", fontsize=5.1, fontweight="semibold")
letter(fig, 92, 2, "b")

# c — UMAP map (visualisation only); keep the main connected component
emb = EMB.merge(sc[["SEQN", "lg"]], on="SEQN", how="left")
Hh, xe, ye_ = np.histogram2d(emb.ux, emb.uy, bins=60)
lab_, _ = ndimage.label(ndimage.binary_dilation(Hh > 0))
ix = np.clip(np.searchsorted(xe, emb.ux, side="right") - 1, 0, 59)
iy = np.clip(np.searchsorted(ye_, emb.uy, side="right") - 1, 0, 59)
comp = lab_[ix, iy]
emb = emb[comp == np.bincount(comp[comp > 0]).argmax()]
dropped = 1 - len(emb) / len(EMB)
axc = axmm(fig, 6, 92, 46, 46)
cmap_s = LinearSegmentedColormap.from_list("s", ["#F3F4F7", "#C9D1E3", "#8495C0", C_PNL, "#1E2B4F"])
hb = axc.hexbin(emb.ux, emb.uy, C=emb.lg, gridsize=34, reduce_C_function=np.mean, cmap=cmap_s, mincnt=3, linewidths=0.15,
                edgecolors="white")
xx, yy = np.mgrid[emb.ux.min():emb.ux.max():140j, emb.uy.min():emb.uy.max():140j]
handles = []
for g, lw in [("E_backpain_ctrl", 0.9), ("B_painless_PN", 1.0), ("D_hidden", 1.3)]:
    sub = emb[emb.group == g]
    k = gaussian_kde(np.vstack([sub.ux, sub.uy]), bw_method=0.35)
    zz = k(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    lev = np.quantile(k(np.vstack([sub.ux, sub.uy])), 0.5)
    c = INK if g == "E_backpain_ctrl" else GC[g]
    ls = GLS[g] if g == "E_backpain_ctrl" else "-"
    axc.contour(xx, yy, zz, levels=[lev], colors=[c], linewidths=lw, linestyles=[ls])
    handles.append(Line2D([], [], color=c, lw=lw, ls=ls, label=NAME[g]))
axc.set_xticks([])
axc.set_yticks([])
axc.set_xlabel("UMAP 1")
axc.set_ylabel("UMAP 2")
axc.set_aspect("equal", adjustable="datalim")
ptitle(axc, "Biological map")
letter(fig, 0, 85, "c")
axl = axmm(fig, 55, 96, 24, 20)
axl.axis("off")
axl.legend(handles=handles, loc="upper left", fontsize=5.3, handlelength=1.8, borderaxespad=0, title="Densest 50%",
           title_fontsize=5.3, alignment="left")
cbx = axmm(fig, 56, 126, 20, 1.6)
cb = fig.colorbar(hb, cax=cbx, orientation="horizontal")
cb.outline.set_linewidth(0.4)
cbx.tick_params(labelsize=5.2, length=1.5)
cbx.set_title("Mean score (logit)", fontsize=5.3, pad=2, loc="left")

# d — what drives the hidden group's score (mean SHAP, D−A vs E−A)
s5 = R3["S5"]
feats = sorted(s5["D_minus_A"], key=lambda f: s5["D_minus_A"][f])
axd = axmm(fig, 106, 92, 28, 46)
for i, f in enumerate(feats):
    dv, ev = s5["D_minus_A"][f], s5["E_minus_A_same_features"][f]
    axd.plot([ev, dv], [i, i], color=GRID, lw=1.2, zorder=1)
    axd.scatter(ev, i, s=9, color=GC["E_backpain_ctrl"], zorder=3)
    axd.scatter(dv, i, s=11, color=C_HID, zorder=3)
ylabels_left(axd, [NM.get(f, f) for f in feats])
axd.axvline(0, color=INK, lw=0.5)
axd.set_xlim(-0.004, 0.06)
axd.set_xticks([0, 0.02, 0.04, 0.06])
axd.set_xticklabels(["0", "0.02", "0.04", "0.06"])
axd.set_ylim(-0.6, len(feats) - 0.4)
axd.set_xlabel("Mean SHAP shift vs reference")
axd.legend(handles=[Line2D([], [], marker="o", color=C_HID, lw=0, ms=3, label="Hidden"),
                    Line2D([], [], marker="o", color=GC["E_backpain_ctrl"], lw=0, ms=3, label="Back pain")],
           loc="lower right", fontsize=5.3, handletextpad=0.2, borderaxespad=0.1)
ptitle(axd, "Score drivers")
letter(fig, 82, 85, "d")

# e — discrimination of every model variant
aucs = [("LightGBM, main", R1["H1_full"]["auroc_B_vs_A"], C_PNL),
        ("No glycaemic features", R1["H1_no_glycemia"]["auroc_B_vs_A"], C_PNL),
        ("No diabetes", R1["H1_no_diabetes"]["auroc_B_vs_A"], C_PNL),
        ("Logistic (L2)", R4["F4"]["logistic_L2"]["auroc"], C_PNL),
        ("Elastic net", R4["F4"]["elastic_net"]["auroc"], C_PNL),
        ("Held-out 2003–04", R3["S1"]["test_auroc_2003_04"], C_PNL),
        ("PU, labelled vs unlabelled", R6["model"]["auroc_original"], C_PUR),
        ("PU, tuned", R6["model"]["auroc_big_tuned"], C_PUR)][::-1]
axe = axmm(fig, 158, 92, 17, 46)
for y0, y1, c in [(1.5, 7.5, C_PNL), (-0.5, 1.5, C_PUR)]:
    axe.axhspan(y0 + 0.06, y1 - 0.06, color=tint(c, 0.07), lw=0, zorder=0)
axe.axvline(R1["H1_full"]["auroc_B_vs_A"], color=C_PNL, lw=0.5, ls=(0, (2, 2)), zorder=0)
for i, (lab, v, c) in enumerate(aucs):
    axe.plot([0.5, v], [i, i], color=tint(c, 0.35), lw=1.0, zorder=1)
    axe.scatter(v, i, s=12, color=c, edgecolor="white", lw=0.5, zorder=3)
    axe.text(1.05, i, f"{v:.3f}", transform=axe.get_yaxis_transform(), ha="left", va="center", fontsize=5.0, color=INK2)
ylabels_left(axe, [a[0] for a in aucs])
axe.axvline(0.5, color=INK, lw=0.5)
axe.set_xlim(0.5, 0.8)
axe.set_xticks([0.5, 0.6, 0.7, 0.8])
axe.set_ylim(-0.6, len(aucs) - 0.4)
axe.set_xlabel("Out-of-fold AUROC")
ptitle(axe, "Discrimination")
axe.text(1.05, len(aucs) - 0.4, "AUROC", transform=axe.get_yaxis_transform(), ha="left", va="bottom", fontsize=5.0,
         fontweight="semibold")
letter(fig, 136, 85, "e")
save(fig, "Fig2_transfer_test")
plt.show()
de = R1["H1_full"]["D_minus_E"]
LEGENDS["Fig2_transfer_test"] = (
    "**Figure 2 | A model trained only on exam-confirmed painless neuropathy recognises the hidden group.** **a**, Out-of-fold "
    "neuropathy-similarity scores (logit) by group: density, individual adults (rug), interquartile range and median; dashed line, "
    "reference median. **b**, Survey-weighted differences in mean score versus reference: for the main model, the normal "
    "confidence distribution of the estimate with 66% (thick) and 95% (thin) cluster-robust intervals; smaller symbols, a model without glycaemic features, adults without diabetes and the held-out 2003–04 cycle (model trained on "
    f"1999–2002). Pre-specified contrast, hidden minus back-pain control: {fmt_ci(de['diff'], de['lo'], de['hi'])}, "
    f"P = {de['p']:.1e}. **c**, UMAP embedding of the same biomarkers (visualisation only), shaded by mean score; contours "
    f"enclose the densest 50% of each group; small satellite clusters ({dropped * 100:.1f}% of adults) not shown. **d**, Mean SHAP "
    "contribution of the twelve largest drivers in the hidden group (teal) and the back-pain control (grey), each minus reference. "
    "**e**, Out-of-fold AUROC of each model: transfer models (painless neuropathy vs reference, blue), positive-unlabeled models "
    "(brown); dashed line, main model.")

# %% [markdown]
# ## Figure 3 — Biomarker fingerprint

# %%
wide = FP.pivot(index="group", columns="feature", values="smd")
wlo = FP.pivot(index="group", columns="feature", values="lo")
whi = FP.pivot(index="group", columns="feature", values="hi")
sigm = (wlo > 0) | (whi < 0)
cols = wide.loc["D_hidden"].sort_values(ascending=False).index.tolist()
hm_rows = ["C_painful_PN", "B_painless_PN", "D_hidden", "F_hands_only", "E_backpain_ctrl"]
DIV = LinearSegmentedColormap.from_list("div", ["#2E4374", "#7D8FBF", "#F6F6F4", "#E79A8A", "#A8321F"])
norm = TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=0.6)
fig = newfig(214)
nc = len(cols)
HX, HW = 30, 128
cw = HW / nc

# a (top) — hidden-group SMD with 95% CI per biomarker, aligned with the heatmap columns
axt = axmm(fig, HX, 8, HW, 14)
dv = wide.loc["D_hidden", cols].values
axt.bar(np.arange(nc) + 0.5, dv, width=0.62, color=[DIV(norm(v)) for v in dv], edgecolor="none")
axt.errorbar(np.arange(nc) + 0.5, dv, yerr=[dv - wlo.loc["D_hidden", cols].values, whi.loc["D_hidden", cols].values - dv],
             fmt="none", ecolor=INK, elinewidth=0.4, capsize=0)
axt.axhline(0, color=INK, lw=0.5)
axt.set_xlim(0, nc)
axt.set_xticks([])
axt.spines["bottom"].set_visible(False)
axt.set_ylabel("Hidden", labelpad=2)
axt.set_ylim(-0.4, 0.5)
axt.set_yticks([-0.3, 0, 0.3])
ptitle(axt, "Adjusted standardized mean difference vs reference (SMD)")
letter(fig, 0, 2, "a")

# a (strip) — biomarker domain
axs = axmm(fig, HX, 23.6, HW, 2.0)
for j, f in enumerate(cols):
    axs.add_patch(Rectangle((j + 0.06, 0), 0.88, 1, facecolor=DOMC.get(f, INK3), edgecolor="none"))
axs.set_xlim(0, nc)
axs.set_ylim(0, 1)
axs.axis("off")
ftext(fig, HX - 1.5, 24.6, "Domain", ha="right", va="center", fontsize=5.4, color=INK2)

# a (main) — heatmap
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
        ax.add_patch(Circle((j + 0.5, i + 0.5), r, facecolor=DIV(norm(np.clip(v, -0.6, 0.6))),
                            edgecolor=INK if sig else "none", lw=0.45, zorder=3))
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

# a (right) — number of biomarkers with CI excluding 0, per group
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


axleg = axmm(fig, HX, 59.5, HW, 3)
axleg.axis("off")
axleg.legend(handles=[Line2D([], [], marker="s", color=c, lw=0, ms=3.6, label=d) for d, c, _ in DOMAINS], loc="center",
             ncol=6, fontsize=5.3, handletextpad=0.2, columnspacing=1.2, borderaxespad=0)

# b, c — biomarker-by-biomarker agreement with 95% CIs, labels for every biomarker whose CI excludes 0
def agree(ax, gx, title):
    x, y = wide.loc[gx, cols].values, wide.loc["D_hidden", cols].values
    xl, xh = wlo.loc[gx, cols].values, whi.loc[gx, cols].values
    yl, yh = wlo.loc["D_hidden", cols].values, whi.loc["D_hidden", cols].values
    lim = 0.62
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.plot([-lim, lim], [-lim, lim], color=INK3, lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax.axhline(0, color=GRID, lw=0.5, zorder=0)
    ax.axvline(0, color=GRID, lw=0.5, zorder=0)
    ax.errorbar(x, y, xerr=[x - xl, xh - x], yerr=[y - yl, yh - y], fmt="none", ecolor="#d9d9d9", elinewidth=0.35, capsize=0,
                zorder=1)
    n_ = len(x)
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    sres = np.sqrt((res ** 2).sum() / (n_ - 2))
    sxx = ((x - x.mean()) ** 2).sum()
    tq = sst.t.ppf(0.975, n_ - 2)
    blo, bhi = b - tq * sres / np.sqrt(sxx), b + tq * sres / np.sqrt(sxx)
    xx_ = np.linspace(-lim * 0.95, lim * 0.95, 60)
    band = tq * sres * np.sqrt(1 / n_ + (xx_ - x.mean()) ** 2 / sxx)
    ax.fill_between(xx_, a + b * xx_ - band, a + b * xx_ + band, color=tint(GC[gx], 0.18), lw=0, zorder=0)
    ax.plot(xx_, a + b * xx_, color=GC[gx], lw=0.8, zorder=2)
    sig_both = [bool(sigm.loc["D_hidden", f] and sigm.loc[gx, f]) for f in cols]
    ax.scatter(x, y, s=[16 if sb else 11 for sb in sig_both], color=[DOMC.get(f, INK3) for f in cols],
               lw=[0.6 if sb else 0.35 for sb in sig_both], edgecolor=[INK if sb else "white" for sb in sig_both], zorder=3)
    r, pr = sst.pearsonr(x, y)
    stat = ax.text(lim - 0.03, -lim + 0.03,
                   f"r = {r:.2f}, {pfmt(pr)}\nslope {b:.2f} (95% CI {blo:.2f}–{bhi:.2f})\n"
                   f"cosine similarity {float(COS.loc['D_hidden', gx]):.2f}\n{sum(sig_both)} biomarkers shifted in both",
                   ha="right", va="bottom", fontsize=5.0, color=INK, linespacing=1.35, zorder=6,
                   bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.15", fc="white", ec=GRID, lw=0.5))
    sel = [i for i, f in enumerate(cols) if sigm.loc["D_hidden", f]]
    # label rails: points above the identity line are labelled in a left column, the rest in a right column
    for side, xt, ha, lo_, hi_ in [("L", -lim + 0.03, "left", -lim + 0.05, lim - 0.12), ("R", lim - 0.03, "right", -0.27, lim - 0.05)]:
        its = sorted([i for i in sel if ((y[i] - x[i]) > 0) == (side == "L")], key=lambda i: y[i])
        if not its:
            continue
        gap = 0.047
        ys = [min(max(y[i], lo_), hi_) for i in its]
        for k in range(1, len(ys)):
            ys[k] = max(ys[k], ys[k - 1] + gap)
        if ys[-1] > hi_:
            ys = [v - (ys[-1] - hi_) for v in ys]
        for k in range(len(ys) - 2, -1, -1):
            ys[k] = min(ys[k], ys[k + 1] - gap)
        for i, yt in zip(its, ys):
            ax.annotate(NM.get(cols[i], cols[i]), (x[i], y[i]), (xt, yt), ha=ha, va="center", fontsize=4.8, color=INK2,
                        fontweight="semibold" if sig_both[i] else "normal", zorder=5,
                        arrowprops=dict(arrowstyle="-", lw=0.3, color="#b9bdc4", shrinkA=0.8, shrinkB=1.6,
                                        connectionstyle="arc3,rad=0"))
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_xlabel(f"SMD, {NAME[gx].lower()}")
    ax.set_ylabel("SMD, hidden neuropathy")
    ptitle(ax, title)


axb = axmm(fig, 14, 78, 64, 64)
agree(axb, "B_painless_PN", "Hidden vs painless neuropathy")
axc = axmm(fig, 104, 78, 64, 64)
agree(axc, "E_backpain_ctrl", "Hidden vs back-pain control")
letter(fig, 0, 70, "b")
letter(fig, 90, 70, "c")

# d — cosine similarity matrix
cos_order = ["D_hidden", "B_painless_PN", "C_painful_PN", "F_hands_only", "E_backpain_ctrl"]
axd = axmm(fig, 24, 166, 32, 32)
Cm = COS.loc[cos_order, cos_order].values.astype(float)
cmap_c = LinearSegmentedColormap.from_list("c", ["#F2F7F6", "#A9D8CF", "#00A087", "#005C4F"])
for i in range(len(cos_order)):
    for j in range(i):
        v = Cm[i, j]
        yy_ = len(cos_order) - 1 - i
        axd.add_patch(Rectangle((j, yy_), 1, 1, facecolor=cmap_c((v - 0.5) / 0.5), edgecolor="white", lw=0.8))
        axd.text(j + 0.5, yy_ + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=5.3, color="white" if v > 0.8 else INK)
axd.add_patch(Rectangle((0, 0), 1, len(cos_order) - 1, fill=False, edgecolor=INK, lw=0.8, zorder=5))
axd.set_xlim(0, len(cos_order) - 1)
axd.set_ylim(0, len(cos_order) - 1)
short = {"D_hidden": "Hidden", "B_painless_PN": "Painless", "C_painful_PN": "Painful", "F_hands_only": "Hands only",
         "E_backpain_ctrl": "Back pain"}
axd.set_xticks(np.arange(len(cos_order) - 1) + 0.5)
axd.set_xticklabels([short[g] for g in cos_order[:-1]], rotation=45, ha="right", rotation_mode="anchor", fontsize=5.4)
axd.set_yticks(np.arange(len(cos_order) - 1) + 0.5)
axd.set_yticklabels([short[g] for g in cos_order[1:][::-1]], fontsize=5.4)
axd.tick_params(length=0)
for sp in axd.spines.values():
    sp.set_visible(False)
ptitle(axd, "Profile cosine similarity")
letter(fig, 0, 158, "d")


# e — leave-one-domain-out similarity: is the match driven by a single biomarker domain?
def cosv(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


lodo = [("All biomarkers", list(cols))] + [(f"Without {d[0].lower() + d[1:]}", [f for f in cols if DOMC.get(f) != c])
                                          for d, c, _ in DOMAINS]
comp = ["B_painless_PN", "C_painful_PN", "F_hands_only", "E_backpain_ctrl"]
axe = axmm(fig, 112, 164, 52, 36)
for i, (lab, fs) in enumerate(lodo):
    y = len(lodo) - 1 - i
    vals = [cosv(wide.loc["D_hidden", fs].values, wide.loc[g, fs].values) for g in comp]
    if i == 0:
        axe.axhspan(y - 0.45, y + 0.45, color="#f2f2f2", lw=0, zorder=0)
        print("cosine check (all biomarkers):", dict(zip(comp, np.round(vals, 3))))
    axe.plot([min(vals), max(vals)], [y, y], color=GRID, lw=1.6, zorder=1, solid_capstyle="round")
    for g, v in zip(comp, vals):
        axe.scatter(v, y, s=16 if g == "B_painless_PN" else 12, color=GC[g], edgecolor="white", lw=0.5, zorder=3,
                    marker="D" if g == "B_painless_PN" else "o")
    axe.text(1.02, y, f"{len(fs)}", transform=axe.get_yaxis_transform(), ha="left", va="center", fontsize=5.0, color=INK2)
axe.text(1.02, len(lodo) - 0.4, "n", transform=axe.get_yaxis_transform(), ha="left", va="bottom", fontsize=5.0,
         fontweight="semibold")
ylabels_left(axe, [l for l, _ in lodo[::-1]])
axe.set_xlim(0.4, 1.0)
axe.set_xticks([0.4, 0.6, 0.8, 1.0])
axe.set_ylim(-0.6, len(lodo) - 0.4)
axe.xaxis.grid(True, color=GRID, lw=0.4)
axe.set_axisbelow(True)
axe.set_xlabel("Cosine similarity with the hidden-group profile")
axe.legend(handles=[Line2D([], [], marker="D" if g == "B_painless_PN" else "o", color=GC[g], lw=0, ms=3.2, label=NAME[g])
                    for g in comp], loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=4, fontsize=5.1, handletextpad=0.2,
           columnspacing=0.9, borderaxespad=0)
ptitle(axe, "Similarity with each domain left out")
letter(fig, 66, 158, "e")
save(fig, "Fig3_fingerprint")
plt.show()
LEGENDS["Fig3_fingerprint"] = (
    "**Figure 3 | The hidden group shares the biomarker profile of exam-confirmed neuropathy.** **a**, Standardized mean "
    "difference (SMD) of each biomarker versus reference after adjustment for age, sex and race. Top, hidden group with 95% CI; "
    "coloured strip, biomarker domain; bubble heatmap, all groups, biomarkers ordered by the hidden group (colour, SMD; circle "
    "area, |SMD|; outlined, 95% CI excludes 0); right, "
    "number of biomarkers whose CI excludes 0. **b,c**, Each point is one biomarker, coloured by domain (grey bars, 95% CIs): "
    "hidden-group SMD against painless neuropathy (**b**) and the back-pain control (**c**); dashed line, identity; solid line "
    "and band, least-squares fit with 95% confidence band; labelled, biomarkers whose 95% CI excludes 0 in the hidden group "
    "(bold and outlined, also in the comparison group). Pearson r P values are nominal (biomarkers are correlated). **d**, Cosine similarity between group "
    "profiles (outlined column: similarity with the hidden group). **e**, Cosine similarity with the hidden-group profile "
    "recomputed after leaving out one biomarker domain at a time (n, biomarkers retained); grey bar, range across groups.")

# %% [markdown]
# ## Figure 4 — Positive-unlabeled ranking

# %%
fig = newfig(136)
d_ = DEC.decile.values


STATLAB = {k: f"OR/SD {R5['V1'][k]['OR']:.2f}, {pfmt(R5['V1'][k]['p'])}"
           for k in ["foot_sx", "foot_numb", "back_pain", "hand_only_sx"]}
STATLAB["died"] = f"HR/SD {R5['V2']['age_sex_race']['HR']:.2f}, {pfmt(R5['V2']['age_sex_race']['p'])}"


def dec_plot(ax, keys, ylab, ylim):
    for k, c, mk, lab in keys:
        p, lo, hi = DEC[k] * 100, DEC[f"{k}_lo"] * 100, DEC[f"{k}_hi"] * 100
        ax.fill_between(d_, lo, hi, color=tint(c, 0.15), lw=0)
        ratio = DEC[k].iloc[-1] / DEC[k].iloc[0]
        ax.plot(d_, p, color=c, lw=1.0, marker=mk, ms=3.0, mfc="white" if mk == "o" else c, mec=c, mew=0.8,
                label=f"{lab}\nD10/D1 {ratio:.1f}× · {STATLAB[k]}")
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.6, 10.4)
    ax.set_ylim(*ylim)
    ax.set_xlabel("PU-score decile (within age band × sex)")
    ax.set_ylabel(ylab)
    ax.yaxis.grid(True, color=GRID, lw=0.4)
    ax.set_axisbelow(True)


axa = axmm(fig, 12, 8, 44, 44)
dec_plot(axa, [("foot_sx", C_HID, "o", "Foot pain or tingling"), ("foot_numb", C_PNL, "s", "Foot numbness")], "Prevalence (%)",
         (0, 26))
axa.legend(loc="upper left", fontsize=5.0, handlelength=1.5, labelspacing=0.6)
ptitle(axa, "Neuropathic foot symptoms")
letter(fig, 0, 2, "a")
axb = axmm(fig, 72, 8, 44, 44)
dec_plot(axb, [("back_pain", GC["E_backpain_ctrl"], "o", "Back pain"), ("hand_only_sx", GC["F_hands_only"], "s", "Hand-only symptoms")],
         "Prevalence (%)", (0, 70))
axb.legend(loc="upper left", fontsize=5.0, handlelength=1.5, ncol=1, labelspacing=0.6)
ptitle(axb, "Negative-control symptoms")
letter(fig, 60, 2, "b")
axc = axmm(fig, 132, 8, 44, 44)
dec_plot(axc, [("died", INK, "o", "Died")], "Died by 2019 (%)", (0, 60))
axc.legend(loc="upper left", fontsize=5.0, handlelength=1.5)
ptitle(axc, "All-cause mortality (crude)")
letter(fig, 120, 2, "c")

# d — balance across deciles (design check)
axd1 = axmm(fig, 12, 76, 44, 18)
axd1.plot(d_, DEC.mean_age, color=INK, lw=0.9, marker="o", ms=2.6, mfc="white")
axd1.set_ylim(50, 66)
axd1.set_xticks(range(1, 11))
axd1.set_xticklabels([])
axd1.set_xlim(0.6, 10.4)
axd1.set_ylabel("Mean age (y)")
axd1.yaxis.grid(True, color=GRID, lw=0.4)
ptitle(axd1, "Balance across deciles")
axd2 = axmm(fig, 12, 98, 44, 18)
axd2.plot(d_, DEC.pct_female * 100, color=INK, lw=0.9, marker="s", ms=2.6, mfc="white")
axd2.set_ylim(40, 62)
axd2.set_xticks(range(1, 11))
axd2.set_xlim(0.6, 10.4)
axd2.set_ylabel("Female (%)")
axd2.set_xlabel("PU-score decile")
axd2.yaxis.grid(True, color=GRID, lw=0.4)
axd2.text(10.3, 41, f"n per decile {DEC.n.min():,}–{DEC.n.max():,}", fontsize=5.1, ha="right", va="bottom", color=INK2)
letter(fig, 0, 70, "d")

# e — per-SD associations: all cycles (confidence distribution) and held-out cycle; mortality
axe = axmm(fig, 96, 76, 48, 40)
V1, V3, V2 = R5["V1"], R5["V3"], R5["V2"]
frows = [("Foot pain or tingling", "foot_sx", C_HID), ("Foot numbness", "foot_numb", C_PNL),
         ("Back pain", "back_pain", GC["E_backpain_ctrl"]), ("Hand-only symptoms", "hand_only_sx", GC["F_hands_only"])]
yl, ylab = [], []
y = 0
for lab, k, c in frows[::-1]:
    v, w = V1[k], V3[k]
    eye(axe, v["OR"], v["lo"], v["hi"], y + 0.08, c, h=0.3, log=True, ms=12)
    interval(axe, w["OR"], w["lo"], w["hi"], y - 0.2, c, log=True, mk="D", hollow=True, ms=8, thick=1.3)
    axe.text(1.04, y, f"{v['OR']:.2f}  /  {w['OR']:.2f}", transform=axe.get_yaxis_transform(), ha="left", va="center",
             fontsize=5.2, color=INK2)
    axe.text(1.42, y, f"{pshort(v['p'])}  /  {pshort(w['p'])}", transform=axe.get_yaxis_transform(), ha="left", va="center",
             fontsize=5.2, color=INK2)
    yl.append(y)
    ylab.append(lab)
    y += 1
y += 0.4
for lab, k in [("Mortality, + ABI/CVD", "plus_circulation_CVD"), ("Mortality", "age_sex_race")]:
    v = V2[k]
    eye(axe, v["HR"], v["lo"], v["hi"], y, INK, h=0.3, log=True, mk="s", ms=12)
    axe.text(1.04, y, fmt_ci(v["HR"], v["lo"], v["hi"]), transform=axe.get_yaxis_transform(), ha="left", va="center",
             fontsize=5.2, color=INK2)
    axe.text(1.42, y, pshort(v["p"]), transform=axe.get_yaxis_transform(), ha="left", va="center", fontsize=5.2, color=INK2)
    yl.append(y)
    ylab.append(lab)
    y += 1
axe.axhline(3.7, color=GRID, lw=0.5)
axe.axvline(1, color=INK, lw=0.5)
logx(axe, [0.8, 1, 1.25, 1.5, 2], (0.78, 2.2))
axe.set_yticks(yl)
axe.set_yticklabels(ylab)
axe.tick_params(axis="y", length=0)
axe.spines["left"].set_visible(False)
axe.set_ylim(-0.5, y - 0.35)
axe.text(1.04, 3.0 + 0.62, "OR: all / held-out", transform=axe.get_yaxis_transform(), ha="left", va="bottom", fontsize=5.1,
         fontweight="bold")
axe.text(1.42, 3.0 + 0.62, "P: all / held-out", transform=axe.get_yaxis_transform(), ha="left", va="bottom", fontsize=5.1,
         fontweight="bold")
axe.text(1.04, y - 0.55, "HR (95% CI)", transform=axe.get_yaxis_transform(), ha="left", va="bottom", fontsize=5.1,
         fontweight="bold")
axe.text(1.42, y - 0.55, "P", transform=axe.get_yaxis_transform(), ha="left", va="bottom", fontsize=5.1, fontweight="bold")
axe.set_xlabel("Odds or hazard ratio per SD of PU score")
axe.legend(handles=[eye_handle(INK2, "All cycles 1999–2004"),
                    Line2D([], [], marker="D", color=INK2, mfc="white", lw=1.3, ms=3, label="Held-out 2003–04"),
                    Line2D([], [], marker="s", color=INK, mec="white", lw=2.0, ms=3.4, label="Cox HR")],
           loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=3, fontsize=5.3, handletextpad=0.3, columnspacing=1.0,
           borderaxespad=0, handlelength=1.6)
ptitle(axe, "Per-SD associations (adjusted for age, sex, race)")
letter(fig, 72, 70, "e")
save(fig, "Fig4_pu_ranking")
plt.show()
LEGENDS["Fig4_pu_ranking"] = (
    "**Figure 4 | A positive-unlabeled model ranks normal-exam adults by neuropathy-like biology, and symptoms follow the "
    "ranking.** Normal-exam adults were split into deciles of the PU posterior, ranked within 10-year age band × sex. Symptoms were "
    "not model inputs. **a**, Foot pain/tingling and numbness. **b**, Negative-control symptoms. **c**, Deaths during follow-up to "
    "2019 (crude). Shaded bands, Wilson 95% CIs; labels, decile 10/decile 1 ratio. **d**, Mean age and percentage female by "
    "decile, confirming that the ranking is not an age–sex gradient. **e**, Odds ratios per SD of the PU score in all cycles "
    "(confidence distribution with 66%/95% intervals) and in the held-out 2003–04 cycle (diamonds; model trained on 1999–2002); "
    f"a larger tuned model gave foot pain/tingling OR {R6['V1_big']['foot_sx']['OR']:.2f}, back pain "
    f"{R6['V1_big']['back_pain']['OR']:.2f} and hand-only {R6['V1_big']['hand_only_sx']['OR']:.2f}. Cox hazard ratios for mortality without and with ankle–brachial index, claudication and cardiovascular "
    "history.")

# %% [markdown]
# ## Figure 5 — Mortality

# %%
fig = newfig(124)
H2 = R1["H2_main"]
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
    axr.text(19.4, y, f"{int(KMS.loc[g, 'deaths']):,} / {int(KMS.loc[g, 'n']):,}", ha="right", va="center", fontsize=5.4,
             color=INK2)
axr.text(-1.4, len(ORDER) - 0.1, "Number at risk", ha="right", va="bottom", fontsize=5.6, fontweight="bold")
axr.text(19.4, len(ORDER) - 0.1, "Deaths / n", ha="right", va="bottom", fontsize=5.6, fontweight="bold")

# b — main adjusted HRs (confidence distributions on the log scale)
rows = [g for g in ORDER if g != "A_reference"]
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

# c — cause-specific HRs as a tile matrix (colour = log HR; bold = 95% CI excludes 1)
s4 = R3["S4"]
causes = [("cardiovascular", "Cardio-\nvascular"), ("cancer", "Cancer"), ("other_causes", "Other\ncauses")]
HRC = LinearSegmentedColormap.from_list("hr", [C_PNL, "#F6F6F4", C_PFL])
hnorm = TwoSlopeNorm(vmin=np.log(0.5), vcenter=0, vmax=np.log(3))
axc = axmm(fig, 146, 80, 34, 30)
for i, g in enumerate(rows):
    y = len(rows) - 1 - i
    for j, (ck, cl) in enumerate(causes):
        v = s4[ck]["table"][g]
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
axc.set_xticklabels([f"{cl}\n({s4[ck]['deaths']:,} deaths)" for ck, cl in causes], fontsize=5.2, linespacing=1.1)
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
save(fig, "Fig5_mortality")
plt.show()
s6 = R3["S6"]
LEGENDS["Fig5_mortality"] = (
    "**Figure 5 | Mortality over 17 years of linked follow-up.** **a**, Survey-weighted Kaplan–Meier survival by group (shaded, "
    "95% CI) with unweighted numbers at risk and deaths/n. **b**, Survey-weighted Cox hazard ratios versus reference (confidence "
    "distribution on the log scale with 66%/95% intervals), adjusted for "
    "age, sex, race, education, income, smoking, BMI, diabetes and kidney function; 95% CIs from cluster-robust variance; "
    f"deaths/n in the model sample. E-value for the hidden group {s6['E_value_point']:.2f} (CI limit {s6['E_value_CI']:.2f}). "
    "**c**, Cause-specific hazard ratios with 95% CIs (* P < 0.05, ** P < 0.01, *** P < 0.001), tiles coloured by log hazard ratio (blue < 1 < red); outlined and bold, "
    "95% CI excludes 1. The hidden-group excess was not separable "
    "from peripheral vascular status (Fig. 6).")

# %% [markdown]
# ## Figure 6 — Sensitivity analyses as specification curves

# %%
H1, S1, S3, F1, F4 = R1["H1_full"], R3["S1"], R3["S3"], R4["F1_strict_D"], R4["F4"]
FN, FA = R4["F2_new_exclusions"], R4["F2_all_exclusions"]
EX_S3 = {"Excl. sciatica", "Excl. low B12", "Excl. PAD / claudication"}
EX_ALL = EX_S3 | {"Excl. arthritis, cancer, alcohol, CKD"}


def dspec(name, d, flags):
    return dict(name=name, est=d["diff"], lo=d["lo"], hi=d["hi"], p=d.get("p"), flags=set(flags))


def hspec(name, d, e, flags):
    return dict(name=name, est=d["HR"], lo=d["lo"], hi=d["hi"], p=d.get("p"), ctrl=e, flags=set(flags))


bio_specs = [
    dspec("Main analysis", H1["D_minus_E"], []),
    dspec("ng", R1["H1_no_glycemia"]["D_minus_E"], ["No glycaemic features"]),
    dspec("nd", R1["H1_no_diabetes"]["D_minus_E"], ["Adults without diabetes"]),
    dspec("ho", S1["D_minus_E_2003_04"], ["Held-out cycle 2003–04"]),
    dspec("co", F1["D_minus_E"], ["Corroborated symptoms"]),
    dspec("sc", S3["no_sciatica"]["D_minus_E"], ["Excl. sciatica"]),
    dspec("b12", S3["no_lowB12"]["D_minus_E"], ["Excl. low B12"]),
    dspec("pad", S3["no_vascular"]["D_minus_E"], ["Excl. PAD / claudication"]),
    dspec("s3", S3["all_excluded"]["D_minus_E"], EX_S3),
    dspec("cm", FN["D_minus_E"], ["Excl. arthritis, cancer, alcohol, CKD"]),
    dspec("all", FA["D_minus_E"], EX_ALL),
    dspec("lr", F4["logistic_L2"]["D_minus_E"], ["Logistic regression (L2)"]),
    dspec("en", F4["elastic_net"]["D_minus_E"], ["Elastic net"]),
]
bio_cats = [("Learner", ["Logistic regression (L2)", "Elastic net"]),
            ("Features", ["No glycaemic features"]),
            ("Sample", ["Adults without diabetes", "Held-out cycle 2003–04"]),
            ("Case definition", ["Corroborated symptoms"]),
            ("Exclusions", ["Excl. sciatica", "Excl. low B12", "Excl. PAD / claudication", "Excl. arthritis, cancer, alcohol, CKD"])]
H2m = R1["H2_main"]
mort_specs = [
    hspec("Main analysis", H2m["D_hidden"], H2m["E_backpain_ctrl"], []),
    hspec("nd", R1["H2_no_diabetes"]["D_hidden"], R1["H2_no_diabetes"]["E_backpain_ctrl"], ["Adults without diabetes"]),
    hspec("e2", R1["H2_excl_2y"]["D_hidden"], R1["H2_excl_2y"]["E_backpain_ctrl"], ["Excl. deaths < 2 y"]),
    hspec("c1", S1["cox_1999_2002"]["D_hidden"], S1["cox_1999_2002"]["E_backpain_ctrl"], ["Cycles 1999–2002 only"]),
    hspec("c2", S1["cox_2003_2004"]["D_hidden"], S1["cox_2003_2004"]["E_backpain_ctrl"], ["Cycle 2003–04 only"]),
    hspec("co", F1["HR_D"], F1["HR_E"], ["Corroborated symptoms"]),
    hspec("sc", S3["no_sciatica"]["HR_D"], S3["no_sciatica"]["HR_E"], ["Excl. sciatica"]),
    hspec("b12", S3["no_lowB12"]["HR_D"], S3["no_lowB12"]["HR_E"], ["Excl. low B12"]),
    hspec("pad", S3["no_vascular"]["HR_D"], S3["no_vascular"]["HR_E"], ["Excl. PAD / claudication"]),
    hspec("s3", S3["all_excluded"]["HR_D"], S3["all_excluded"]["HR_E"], EX_S3),
    hspec("cm", FN["HR_D"], FN["HR_E"], ["Excl. arthritis, cancer, alcohol, CKD"]),
    hspec("all", FA["HR_D"], FA["HR_E"], EX_ALL),
    hspec("abi", R4["F3"]["allcause_adj_ABI_claud_CVD"], None, ["+ ABI, claudication, CVD"]),
    hspec("nc", R4["F3"]["allcause_normal_circulation_only"], None, ["Normal circulation only"]),
]
mort_cats = [("Sample", ["Adults without diabetes", "Cycles 1999–2002 only", "Cycle 2003–04 only", "Excl. deaths < 2 y"]),
             ("Case definition", ["Corroborated symptoms"]),
             ("Exclusions", ["Excl. sciatica", "Excl. low B12", "Excl. arthritis, cancer, alcohol, CKD"]),
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


fig = newfig(119)
spec_curve(fig, 44, 8, 44, 40, bio_specs, bio_cats, C_HID, 0, False, "Hidden − back pain (logit)", (-0.02, 0.48),
           [0, 0.1, 0.2, 0.3, 0.4], "Biological similarity")
spec_curve(fig, 134, 8, 46, 40, mort_specs, mort_cats, C_HID, 1, True, "Hazard ratio, hidden vs reference", (0.55, 4.2),
           [0.6, 1, 1.5, 2, 3, 4], "All-cause mortality")
axlg = axmm(fig, 20, 111, 160, 4)
axlg.axis("off")
axlg.legend(handles=[Line2D([], [], color=INK2, lw=2.2, marker="o", ms=3.4, mec="white", label="Hidden group, 95% CI excludes null"),
                     Line2D([], [], color=INK2, lw=2.2, marker="o", ms=3.4, mfc="white", label="95% CI includes null"),
                     Line2D([], [], color=INK2, lw=0, marker="D", ms=3.2, label="Main analysis"),
                     Line2D([], [], color=GC["E_backpain_ctrl"], lw=1.0, marker="o", ms=2.4, mec="white", label="Back-pain control"),
                     Line2D([], [], color=INK, lw=0, marker="^", ms=3, label="Circulation-related analysis")],
            loc="center", ncol=5, fontsize=5.2, handlelength=1.4, columnspacing=1.2, borderaxespad=0)
letter(fig, 0, 2, "a")
letter(fig, 92, 2, "b")
save(fig, "Fig6_robustness")
plt.show()
LEGENDS["Fig6_robustness"] = (
    "**Figure 6 | Sensitivity analyses as specification curves.** Each column is one pre-specified analysis, ordered by its "
    "estimate (thin line connects the ordered estimates); the dot matrix below shows which analytic choices it makes (shaded column, main analysis). **a**, Pre-specified "
    "biological contrast, hidden minus back-pain control (out-of-fold logit score). **b**, Adjusted hazard ratio for the hidden group "
    "versus reference with the back-pain control alongside (grey); red choices mark analyses that exclude, "
    "adjust for or restrict on peripheral circulation (also marked by triangles in both panels). Thick and thin bars, 66% and 95% CIs; hollow symbols, 95% CI includes the null; "
    "dashed line, main estimate. P values for each analysis are printed between the curve and the choice matrix.")

# %%
with open("figures/figure_legends.md", "w") as fh:
    fh.write("\n\n".join(LEGENDS.values()) + "\n")
print(sorted(os.listdir("figures")))
