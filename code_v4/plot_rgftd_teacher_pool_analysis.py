import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import patches
import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - optional dependency
    stats = None


PAPER_DPI = 300
FIG_FONT = "DejaVu Sans"


def _apply_paper_style():
    plt.rcParams.update({
        "font.family": FIG_FONT,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _load_matrix(path: Path):
    df = pd.read_csv(path)
    if df.empty:
        return None
    index_col = df.columns[0]
    return df.set_index(index_col)


def _load_pair_df(analysis_dir: Path):
    return pd.read_csv(analysis_dir / "teacher_pool_pair_metrics.csv")


def _short_client(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    if name.startswith("client"):
        try:
            idx = int(name.replace("client", ""))
            return f"C{idx}"
        except ValueError:
            pass
    return name


def _teacher_label(name: str) -> str:
    if name.startswith("client"):
        try:
            idx = int(name.replace("client", ""))
            return f"T{idx}"
        except ValueError:
            pass
    return name


def _target_label(name: str) -> str:
    if name.startswith("client"):
        try:
            idx = int(name.replace("client", ""))
            return f"C{idx}"
        except ValueError:
            pass
    return name


def _save_figure(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=PAPER_DPI)
    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _debug_heatmap(df: pd.DataFrame, title: str, out_path: Path, cmap="viridis", fmt=".3f",
                   vmin=None, vmax=None):
    if df is None or df.empty:
        return
    values = df.values.astype(float)
    fig_w = max(6, 1.2 * df.shape[1] + 2)
    fig_h = max(4, 0.8 * df.shape[0] + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    im = ax.imshow(values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_xticklabels([_short_client(x) for x in df.columns], fontsize=9)
    ax.set_yticklabels([_short_client(x) for x in df.index], fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    threshold = (np.nanmax(values) + np.nanmin(values)) / 2.0
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = values[i, j]
            color = "white" if not math.isnan(val) and val > threshold else "black"
            ax.text(j, i, format(val, fmt), ha="center", va="center", color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    _save_figure(fig, out_path)


def _top_external_pairs(df: pd.DataFrame):
    rows = []
    ext = df[df["same_client"] == 0].copy()
    for target_name, sub_df in ext.groupby("target_name"):
        row = sub_df.sort_values("analysis_lambda_effective", ascending=False).iloc[0]
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("target_cid")
    return out.reset_index(drop=True)


def _paper_heatmap(ax, df: pd.DataFrame, title: str, cmap, vmin=None, vmax=None,
                   annotate_coords=None, annotate_fmt="{:.2f}",
                   highlight_coords=None, shade_diagonal=False, norm=None):
    values = df.values.astype(float)
    im = ax.imshow(values, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax, norm=norm)
    ax.set_title(title, pad=8, weight="bold")
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_xticklabels([_target_label(x) for x in df.columns])
    ax.set_yticklabels([_teacher_label(x) for x in df.index])
    ax.set_xlabel("Target client")
    ax.set_ylabel("Teacher client")
    ax.set_xticks(np.arange(-0.5, df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, df.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8, alpha=0.9)
    ax.tick_params(length=0)

    if shade_diagonal:
        for i in range(min(df.shape[0], df.shape[1])):
            rect = patches.Rectangle(
                (i - 0.5, i - 0.5),
                1.0,
                1.0,
                facecolor=(0.8, 0.8, 0.8, 0.55),
                edgecolor="gray",
                linewidth=0.8,
                hatch="///",
            )
            ax.add_patch(rect)

    if highlight_coords:
        for r, c in highlight_coords:
            rect = patches.Rectangle(
                (c - 0.5, r - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="black",
                linewidth=1.8,
            )
            ax.add_patch(rect)

    if annotate_coords:
        for r, c in annotate_coords:
            val = values[r, c]
            if np.isnan(val):
                continue
            ax.text(c, r, annotate_fmt.format(val), ha="center", va="center",
                    fontsize=8, color="black", weight="bold")
    return im


def _panel_label(ax, label: str):
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=13, weight="bold",
            va="top", ha="left")


def _utility_bucket(val: float) -> str:
    if val >= 0.015:
        return "high"
    if val >= 0.005:
        return "moderate"
    return "low"


def _margin_sign(val: float) -> str:
    if val > 0.05:
        return "+"
    if val < -0.05:
        return "-"
    return "~"


def _routing_decision(val_lambda: float, val_margin: float) -> str:
    if val_lambda < 0.001 or val_margin <= 0.0:
        return "silence"
    if val_lambda < 0.01:
        return "weak"
    return "distill"


def _failed_checks(row) -> str:
    failed = []
    if float(row["analysis_lambda_effective"]) < 0.001:
        failed.append("low util.")
    if float(row["teacher_seed_support_fg_margin_mean"]) <= 0.0:
        failed.append("neg. margin")
    if float(row["teacher_core_conflict"]) >= 0.20:
        failed.append("core risk")
    if failed:
        return " + ".join(failed)
    passed = ["util.", "margin"]
    if float(row["teacher_core_conflict"]) < 0.20:
        passed.append("safety")
    return "pass: " + "+".join(passed)


def _build_routing_table(top_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in top_df.iterrows():
        selected = (
            _teacher_label(row["teacher_name"])
            if row["analysis_lambda_effective"] >= 0.001 and row["teacher_seed_support_fg_margin_mean"] > 0.0
            else "none"
        )
        rows.append({
            "Target": _target_label(row["target_name"]),
            "Candidate": selected,
            "Checks": _failed_checks(row),
            "Decision": _routing_decision(float(row["analysis_lambda_effective"]), float(row["teacher_seed_support_fg_margin_mean"])),
        })
    return pd.DataFrame(rows)


def _load_case_image(case_dir: Path, name: str):
    path = case_dir / name
    if not path.exists():
        return None
    return mpimg.imread(path)


def _load_preferred_case(case_dir: Path, name: str):
    lean_name = "lean_" + name
    image = _load_case_image(case_dir, lean_name)
    if image is not None:
        return image
    return _load_case_image(case_dir, name)


def _crop_nonwhite(image):
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim == 2:
        mask = arr < 0.985
    else:
        mask = np.any(arr[..., :3] < 0.985, axis=-1)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return arr
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    pad_y = max(2, int(0.01 * (y1 - y0)))
    pad_x = max(2, int(0.01 * (x1 - x0)))
    y0 = max(0, y0 - pad_y)
    x0 = max(0, x0 - pad_x)
    y1 = min(arr.shape[0], y1 + pad_y)
    x1 = min(arr.shape[1], x1 + pad_x)
    return arr[y0:y1, x0:x1]


def _add_case_image(ax, image, title: str):
    ax.axis("off")
    if image is None:
        ax.text(0.5, 0.5, "Case image missing", ha="center", va="center")
        ax.set_title(title, pad=4, weight="bold", fontsize=10)
        return
    ax.imshow(_crop_nonwhite(image))
    ax.set_title(title, pad=4, weight="bold", fontsize=10)


def _make_figure1(analysis_dir: Path, out_dir: Path, pair_df: pd.DataFrame):
    lambda_df = _load_matrix(analysis_dir / "analysis_lambda_effective.csv")
    margin_df = _load_matrix(analysis_dir / "teacher_seed_support_fg_margin_mean.csv")
    top_df = _top_external_pairs(pair_df)
    routing_df = _build_routing_table(top_df)

    highlight_coords = []
    annotate_coords = []
    if not top_df.empty:
        for _, row in top_df.iterrows():
            r = lambda_df.index.get_loc(row["teacher_name"])
            c = lambda_df.columns.get_loc(row["target_name"])
            highlight_coords.append((r, c))
            annotate_coords.append((r, c))

    fig = plt.figure(figsize=(17.2, 5.3), dpi=PAPER_DPI)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.08, 1.08, 1.25], wspace=0.34)

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = _paper_heatmap(
        ax1,
        lambda_df,
        "Target-dependent teacher utility",
        cmap=colors.LinearSegmentedColormap.from_list("utility", ["#edf2f2", "#79b8b1", "#145b73"]),
        vmin=0.0,
        vmax=0.02,
        annotate_coords=annotate_coords,
        annotate_fmt="{:.2f}",
        highlight_coords=highlight_coords,
        shade_diagonal=True,
    )
    _panel_label(ax1, "a")
    c5_idx = lambda_df.columns.get_loc("client5")
    ax1.add_patch(patches.Rectangle((c5_idx - 0.5, -0.5), 1.0, lambda_df.shape[0],
                                    fill=False, edgecolor="#555555", linewidth=2.0, linestyle="--"))
    ax1.text(c5_idx, -1.15, "abstain target", fontsize=8, ha="center")
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.045, pad=0.03)
    cb1.set_label("lambda_eff", fontsize=9)
    cb1.ax.tick_params(labelsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = _paper_heatmap(
        ax2,
        margin_df,
        "Weak-support compatibility",
        cmap="coolwarm",
        annotate_coords=annotate_coords,
        annotate_fmt="{:.2f}",
        highlight_coords=highlight_coords,
        shade_diagonal=True,
        norm=colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
    )
    _panel_label(ax2, "b")
    c5_idx = margin_df.columns.get_loc("client5")
    ax2.add_patch(patches.Rectangle((c5_idx - 0.5, -0.5), 1.0, margin_df.shape[0],
                                    fill=False, edgecolor="#555555", linewidth=2.0, linestyle="--"))
    ax2.annotate(
        "no compatible\nexternal teacher",
        xy=(margin_df.columns.get_loc("client5"), 0.2),
        xytext=(margin_df.columns.get_loc("client5"), -0.88),
        ha="center",
        va="top",
        fontsize=7.5,
        arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8),
    )
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.045, pad=0.03)
    cb2.set_label("fg-bg margin", fontsize=9)
    cb2.ax.tick_params(labelsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    _panel_label(ax3, "c")
    ax3.set_title("Routing rule and decisions", pad=10, weight="bold")
    rule_text = (
        "Routing score = utility x compatibility x safety\n"
        "if score < tau -> silence\n\n"
        "Diagonal cells are masked for reference only.\n"
        "External routing excludes self-teachers."
    )
    ax3.text(0.02, 0.93, rule_text, transform=ax3.transAxes, fontsize=8.5,
             ha="left", va="top",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#f7f9f9", edgecolor="#c9d6d6"))
    y0 = 0.55
    row_h = 0.085
    headers = ["Target", "Candidate", "Checks", "Decision"]
    xs = [0.04, 0.24, 0.47, 0.86]
    for x, header in zip(xs, headers):
        ax3.text(x, y0 + 0.055, header, transform=ax3.transAxes, fontsize=8.2,
                 weight="bold", ha="left", va="center")
    for idx, row in routing_df.reset_index(drop=True).iterrows():
        y = y0 - idx * row_h
        decision = str(row["Decision"])
        face = "#f2f6f6" if idx % 2 == 0 else "white"
        if decision == "silence":
            face = "#eeeeee"
        rect = patches.FancyBboxPatch(
            (0.02, y - 0.045), 0.96, 0.072,
            transform=ax3.transAxes,
            boxstyle="round,pad=0.004,rounding_size=0.006",
            facecolor=face,
            edgecolor="#c8c8c8",
            linewidth=0.6,
        )
        ax3.add_patch(rect)
        ax3.text(xs[0], y - 0.010, str(row["Target"]), transform=ax3.transAxes, fontsize=8.0, ha="left", va="center")
        ax3.text(xs[1], y - 0.010, str(row["Candidate"]), transform=ax3.transAxes, fontsize=8.0, ha="left", va="center")
        ax3.text(xs[2], y - 0.010, str(row["Checks"]), transform=ax3.transAxes, fontsize=7.2, ha="left", va="center")
        color = "#555555" if decision == "silence" else "#145b73"
        ax3.text(xs[3], y - 0.010, decision, transform=ax3.transAxes, fontsize=8.0,
                 color=color, weight="bold", ha="left", va="center")

    fig.suptitle("Teacher utility is pairwise and target-dependent", fontsize=13.5, weight="bold", y=1.01)
    _save_figure(fig, out_dir / "figure1_teacher_availability.png")


def _make_figure2(analysis_dir: Path, out_dir: Path, pair_df: pd.DataFrame, case_dir: Path):
    conflict_df = _load_matrix(analysis_dir / "teacher_core_conflict.csv")
    ext = pair_df[pair_df["same_client"] == 0].copy()

    fig = plt.figure(figsize=(15, 8.6), dpi=PAPER_DPI)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], width_ratios=[1.0, 1.25], hspace=0.28, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    high_conflict = ext.nlargest(5, "teacher_core_conflict")
    high_coords = []
    for _, row in high_conflict.iterrows():
        high_coords.append((conflict_df.index.get_loc(row["teacher_name"]), conflict_df.columns.get_loc(row["target_name"])))
    im1 = _paper_heatmap(
        ax1,
        conflict_df,
        "Core safety conflict",
        cmap=colors.LinearSegmentedColormap.from_list("risk", ["#f7f7f7", "#f6c2b8", "#a61d1d"]),
        vmin=0.0,
        vmax=max(0.25, float(conflict_df.values.max())),
        annotate_coords=high_coords[:4],
        annotate_fmt="{:.2f}",
        highlight_coords=high_coords[:4],
        shade_diagonal=True,
    )
    _panel_label(ax1, "a")
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.045, pad=0.03)
    cb1.set_label("core conflict\n(lower is safer)", fontsize=9)
    cb1.ax.tick_params(labelsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axvspan(-1.0, 0.0, color="#eeeeee", alpha=0.55, zorder=0)
    ax2.axvspan(0.0, 1.05, ymin=0.0, ymax=0.5, color="#f4eadf", alpha=0.42, zorder=0)
    ax2.axvspan(0.0, 1.05, ymin=0.5, ymax=1.0, color="#e8f3f1", alpha=0.42, zorder=0)
    sc = ax2.scatter(
        ext["teacher_seed_support_fg_margin_mean"],
        ext["active_fg_near_seed_ratio"],
        s=80 + 2800 * ext["analysis_lambda_effective"],
        c=ext["teacher_core_conflict"],
        cmap=colors.LinearSegmentedColormap.from_list("risk", ["#f1f1f1", "#f2a693", "#8f1010"]),
        alpha=0.92,
        edgecolor="black",
        linewidth=0.45,
    )
    ax2.axvline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    label_pairs = {
        ("client5", "client4"),
        ("client1", "client6"),
        ("client3", "client5"),
        ("client5", "client6"),
        ("client2", "client3"),
        ("client1", "client5"),
    }
    for _, row in ext.iterrows():
        key = (row["teacher_name"], row["target_name"])
        if key in label_pairs:
            ax2.text(
                row["teacher_seed_support_fg_margin_mean"] + 0.03,
                row["active_fg_near_seed_ratio"] + 0.02,
                f"{_teacher_label(row['teacher_name'])}->{_target_label(row['target_name'])}",
                fontsize=8,
            )
    ax2.text(0.72, 0.91, "eligible", transform=ax2.transAxes, fontsize=9, ha="center")
    ax2.text(0.73, 0.08, "compatible but\nneeds refinement", transform=ax2.transAxes, fontsize=8.5, ha="center")
    ax2.text(0.16, 0.12, "reject /\nsilence", transform=ax2.transAxes, fontsize=8.5, ha="center")
    ax2.text(0.15, 0.88, "redder points:\nunsafe", transform=ax2.transAxes, fontsize=8.5, ha="center")
    ax2.set_title("Raw teacher release is not always target-anchored", pad=8, weight="bold")
    ax2.set_xlabel("Seed fg-bg margin")
    ax2.set_ylabel("Raw near-seed alignment")
    ax2.set_xlim(-1.10, 1.12)
    ax2.set_ylim(-0.04, 0.99)
    ax2.grid(alpha=0.15, linewidth=0.6)
    _panel_label(ax2, "b")
    cb2 = fig.colorbar(sc, ax=ax2, fraction=0.045, pad=0.03)
    cb2.set_label("core conflict", fontsize=9)
    cb2.ax.tick_params(labelsize=8)

    ax3 = fig.add_subplot(gs[1, 0])
    _panel_label(ax3, "c")
    _add_case_image(
        ax3,
        _load_preferred_case(case_dir, "client5_to_client6_rescue.png"),
        "Rescue case: foreground-compatible but spatially off-target",
    )

    ax4 = fig.add_subplot(gs[1, 1])
    _panel_label(ax4, "d")
    _add_case_image(
        ax4,
        _load_preferred_case(case_dir, "client1_to_client5_silence.png"),
        "Silence case: no compatible external teacher",
    )

    fig.suptitle("Teacher strength is not enough: safety and placement both matter", fontsize=13.5, weight="bold", y=0.985)
    _save_figure(fig, out_dir / "figure2_teacher_safety_alignment.png")


def _paired_p_value(x, y):
    if stats is None:
        return None
    try:
        return float(stats.ttest_rel(y, x).pvalue)
    except Exception:
        return None


def _make_figure3(analysis_dir: Path, out_dir: Path, pair_df: pd.DataFrame, case_dir: Path):
    ext = pair_df[pair_df["same_client"] == 0].copy()
    raw_fg = ext["active_foreground_pixels"].replace(0, np.nan).astype(float)
    ext["foreground_retained_ratio"] = (ext["refine_q_fg_mass"].astype(float) / raw_fg).replace([np.inf, -np.inf], np.nan)
    ext["foreground_retained_ratio"] = ext["foreground_retained_ratio"].fillna(0.0).clip(0.0, 1.0)

    fig = plt.figure(figsize=(16.2, 5.1), dpi=PAPER_DPI)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.08, 0.98, 1.24], wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    sizes = 35 + 230 * ext["foreground_retained_ratio"]
    sc = ax1.scatter(
        ext["active_fg_near_seed_ratio"],
        ext["refine_q_near_seed_ratio"],
        s=sizes,
        c=ext["teacher_core_conflict"],
        cmap=colors.LinearSegmentedColormap.from_list("risk", ["#e9f2f1", "#79b8b1", "#8f1010"]),
        alpha=0.9,
        edgecolor="black",
        linewidth=0.45,
    )
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0, alpha=0.8)
    label_pairs = {
        ("client5", "client6"),
        ("client1", "client5"),
        ("client2", "client3"),
        ("client6", "client1"),
    }
    for _, row in ext.iterrows():
        if (row["teacher_name"], row["target_name"]) in label_pairs:
            ax1.text(
                row["active_fg_near_seed_ratio"] + 0.015,
                row["refine_q_near_seed_ratio"] + 0.015,
                f"{_teacher_label(row['teacher_name'])}->{_target_label(row['target_name'])}",
                fontsize=8,
            )
    raw = ext["active_fg_near_seed_ratio"].to_numpy(dtype=float)
    refined = ext["refine_q_near_seed_ratio"].to_numpy(dtype=float)
    improved = int(np.sum(refined > raw + 1e-8))
    total = int(len(ext))
    mean_delta = float(np.mean(refined - raw))
    p_val = _paired_p_value(raw, refined)
    stat_lines = [
        f"Improved: {improved}/{total} pairs",
        f"Mean Delta = {mean_delta:+.2f}",
    ]
    if p_val is not None:
        stat_lines.append(f"paired t-test p = {p_val:.2e}")
    ax1.text(
        0.03,
        0.97,
        "\n".join(stat_lines),
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    )
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlabel("Raw teacher near-seed alignment")
    ax1.set_ylabel("Refined q near-seed alignment")
    ax1.set_title("Refinement increases near-seed alignment", pad=8, weight="bold")
    ax1.grid(alpha=0.15, linewidth=0.6)
    _panel_label(ax1, "a")
    cb1 = fig.colorbar(sc, ax=ax1, fraction=0.045, pad=0.03)
    cb1.set_label("core conflict", fontsize=9)
    cb1.ax.tick_params(labelsize=8)
    for retained in [0.25, 0.50, 0.75]:
        ax1.scatter([], [], s=35 + 230 * retained, color="#d7e9e7", edgecolor="black",
                    linewidth=0.45, label=f"{retained:.2f}")
    leg = ax1.legend(title="fg retained", fontsize=7.5, title_fontsize=8,
                     loc="lower right", frameon=True, borderpad=0.4)
    leg.get_frame().set_alpha(0.9)

    ax2 = fig.add_subplot(gs[0, 1])
    strong = ext[ext["analysis_lambda_effective"] >= 0.01].copy()
    if strong.empty:
        strong = ext.copy()
    metrics = {
        "Near-seed\ngain": strong["refine_q_near_seed_ratio"] - strong["active_fg_near_seed_ratio"],
        "Candidate\ngain": strong["refine_q_candidate_ratio"] - strong["active_fg_candidate_ratio"],
        "Unsupported\nreduction": (1.0 - strong["active_fg_candidate_ratio"]) - strong["refine_unsupported_fg_ratio"],
        "Foreground\nretained": strong["foreground_retained_ratio"],
    }
    names = list(metrics.keys())
    means = np.asarray([float(np.nanmean(metrics[name])) for name in names])
    sems = np.asarray([float(np.nanstd(metrics[name]) / max(np.sqrt(len(metrics[name])), 1.0)) for name in names])
    bar_colors = ["#2f6f7e", "#64a69f", "#b25d39", "#9aa8ad"]
    x = np.arange(len(names))
    ax2.axhline(0.0, color="#666666", linewidth=0.8)
    ax2.bar(x, means, yerr=sems, color=bar_colors, edgecolor="black", linewidth=0.5, capsize=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=12, ha="right", fontsize=8.3)
    ax2.set_ylim(min(-0.15, float(np.nanmin(means - sems)) - 0.05), max(1.05, float(np.nanmax(means + sems)) + 0.05))
    ax2.set_ylabel("Mean ratio / delta")
    ax2.set_title("Correction without foreground collapse", pad=8, weight="bold")
    ax2.grid(axis="y", alpha=0.18, linewidth=0.6)
    ax2.text(0.02, 0.96, f"strong external pairs: n={len(strong)}", transform=ax2.transAxes,
             va="top", ha="left", fontsize=8.3)
    _panel_label(ax2, "b")

    ax3 = fig.add_subplot(gs[0, 2])
    _panel_label(ax3, "c")
    _add_case_image(
        ax3,
        _load_preferred_case(case_dir, "client5_to_client6_rescue.png"),
        "Rescue case: refined q re-anchors released foreground",
    )

    fig.suptitle("Refinement is not identity: it re-anchors teacher supervision", fontsize=13.5, weight="bold", y=1.01)
    _save_figure(fig, out_dir / "figure3_refinement_pullback.png")


def _plot_top_teacher_summary(df: pd.DataFrame, out_path: Path):
    if df.empty:
        return

    order = list(df["target_name"])
    x = np.arange(len(order))
    width = 0.34

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=220)
    axes = axes.ravel()

    ax = axes[0]
    ax.bar(x - width / 2, df["analysis_lambda_effective"], width, label="lambda_eff")
    ax.bar(x + width / 2, df["teacher_core_conflict"], width, label="core_conflict")
    ax.set_title("Top External Teacher: Strength vs Safety")
    ax.set_xticks(x)
    ax.set_xticklabels([_target_label(v) for v in order])
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x - width / 2, df["teacher_seed_support_fg_prob_mean"], width, label="seed_fgp")
    ax.bar(x + width / 2, df["teacher_seed_support_fg_margin_mean"], width, label="seed_fgm")
    ax.set_title("Top External Teacher: Seed Foreground Ability")
    ax.set_xticks(x)
    ax.set_xticklabels([_target_label(v) for v in order])
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar(x - width / 2, df["active_fg_candidate_ratio"], width, label="raw candidate")
    ax.bar(x + width / 2, df["active_fg_near_seed_ratio"], width, label="raw near-seed")
    ax.set_title("Raw Release Alignment")
    ax.set_xticks(x)
    ax.set_xticklabels([_target_label(v) for v in order])
    ax.legend(fontsize=8)

    ax = axes[3]
    ax.bar(x - width / 2, df["refine_q_candidate_ratio"], width, label="q candidate")
    ax.bar(x + width / 2, df["refine_q_near_seed_ratio"], width, label="q near-seed")
    ax.set_title("Refined q Alignment")
    ax.set_xticks(x)
    ax.set_xticklabels([_target_label(v) for v in order])
    ax.legend(fontsize=8)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    _save_figure(fig, out_path)


def _plot_refine_delta(df: pd.DataFrame, out_path: Path):
    if df.empty:
        return
    order = list(df["target_name"])
    x = np.arange(len(order))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5), dpi=220)
    ax.bar(x - width, df["refine_teacher_q_kl"], width, label="teacher-q KL")
    ax.bar(x, df["refine_q_fg_delta"], width, label="q_fg_delta")
    ax.bar(x + width, df["refine_unsupported_fg_ratio"], width, label="unsupported_fg")
    ax.set_title("Refinement Changes Teacher Prior")
    ax.set_xticks(x)
    ax.set_xticklabels([_target_label(v) for v in order])
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, out_path)


def _plot_pair_scatter(df: pd.DataFrame, out_path: Path):
    ext = df[df["same_client"] == 0].copy()
    if ext.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6), dpi=220)
    sc = ax.scatter(
        ext["active_fg_near_seed_ratio"],
        ext["refine_q_near_seed_ratio"],
        s=40 + 1600 * ext["analysis_lambda_effective"],
        c=ext["teacher_core_conflict"],
        cmap="coolwarm",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
    )
    for _, row in ext.iterrows():
        ax.text(
            row["active_fg_near_seed_ratio"] + 0.005,
            row["refine_q_near_seed_ratio"] + 0.005,
            f"{_teacher_label(row['teacher_name'])}->{_target_label(row['target_name'])}",
            fontsize=7,
        )
    ax.set_xlabel("Raw active_fg_near_seed_ratio")
    ax.set_ylabel("Refined q_near_seed_ratio")
    ax.set_title("Refinement Pulls Teacher Mass Toward Target Support")
    ax.grid(alpha=0.2)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("teacher_core_conflict", fontsize=8)
    fig.tight_layout()
    _save_figure(fig, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_dir", type=str, required=True)
    parser.add_argument("--case_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--paper_only", action="store_true")
    args = parser.parse_args()

    _apply_paper_style()

    analysis_dir = Path(args.analysis_dir)
    debug_dir = analysis_dir / "figures"
    debug_dir.mkdir(parents=True, exist_ok=True)

    if args.output_dir is None:
        paper_dir = analysis_dir / "paper_figures"
    else:
        paper_dir = Path(args.output_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)

    if args.case_dir is None:
        case_dir = analysis_dir.parent / "case_visualizations_v35"
    else:
        case_dir = Path(args.case_dir)

    pair_df = _load_pair_df(analysis_dir)

    if not args.paper_only:
        heatmaps = [
            ("analysis_lambda_effective.csv", "Teacher -> Target Lambda Effective", "magma", ".3f", 0.0, 0.02),
            ("teacher_seed_support_fg_prob_mean.csv", "Teacher -> Target Seed FG Probability", "viridis", ".3f", 0.0, 1.0),
            ("teacher_seed_support_fg_margin_mean.csv", "Teacher -> Target Seed FG Margin", "coolwarm", ".3f", -1.0, 1.0),
            ("teacher_core_conflict.csv", "Teacher -> Target Core Conflict", "coolwarm_r", ".3f", 0.0, 0.35),
            ("active_fg_candidate_ratio.csv", "Raw Release Candidate Alignment", "viridis", ".3f", 0.0, 1.0),
            ("active_fg_near_seed_ratio.csv", "Raw Release Near-Seed Alignment", "viridis", ".3f", 0.0, 1.0),
        ]

        for filename, title, cmap, fmt, vmin, vmax in heatmaps:
            matrix = _load_matrix(analysis_dir / filename)
            _debug_heatmap(matrix, title, debug_dir / filename.replace(".csv", ".png"), cmap=cmap, fmt=fmt, vmin=vmin, vmax=vmax)

        top_df = _top_external_pairs(pair_df)
        if not top_df.empty:
            top_df.to_csv(debug_dir / "top_external_pairs.csv", index=False)
            _plot_top_teacher_summary(top_df, debug_dir / "top_external_teacher_summary.png")
            _plot_refine_delta(top_df, debug_dir / "refinement_delta_summary.png")
            _plot_pair_scatter(pair_df, debug_dir / "refinement_pull_scatter.png")

    _make_figure1(analysis_dir, paper_dir, pair_df)
    _make_figure2(analysis_dir, paper_dir, pair_df, case_dir)
    _make_figure3(analysis_dir, paper_dir, pair_df, case_dir)

    print(f"paper_figure_dir={paper_dir}")


if __name__ == "__main__":
    main()
