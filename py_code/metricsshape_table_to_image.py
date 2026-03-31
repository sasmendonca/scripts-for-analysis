import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLUMNS = [
    "Score",
    "Query",
    "AUC",
    "TOP1_EF",
    "TOP1_BEDROC",
    "TOP1_CCR",
    "TOP5_EF",
    "TOP5_BEDROC",
    "TOP5_CCR",
    "TOP10_EF",
    "TOP10_BEDROC",
    "TOP10_CCR",
]

RAW_ROWS = [
    ["SF1", "Q1", "0.67", "2.75", "0.08", "0.5", "2.58", "0.11", "0.52", "2.01", "0.16", "0.53"],
    ["SF1", "Q2", "0.88", "34.76", "0.65", "0.63", "3.17", "0.62", "0.71", "7.57", "0.65", "0.74"],
    ["SF2", "Q1", "0.75", "11.01", "0.33", "0.53", "4.16", "0.23", "0.56", "3.32", "0.29", "0.59"],
    ["SF2", "Q2", "0.88", "35.45", "0.64", "0.62", "13.42", "0.66", "0.7", "7.5", "0.69", "0.73"],
    ["SF3", "Q1", "0.52", "0.86", "0.02", "0.5", "0.92", "0.04", "0.5", "0.87", "0.07", "0.5"],
    ["SF3", "Q2", "0.79", "9.81", "0.19", "0.52", "8.33", "0.34", "0.6", "5.59", "0.43", "0.63"],
    ["SF4", "Q1", "0.82", "20.65", "0.45", "0.54", "8.57", "0.43", "0.6", "5.57", "0.48", "0.64"],
    ["SF4", "Q2", "0.76", "13.42", "0.29", "0.53", "7.12", "0.31", "0.57", "4.97", "0.39", "0.6"],
    ["SF5", "Q1", "0.56", "1.2", "0.03", "0.5", "0.96", "0.05", "0.5", "1.17", "0.08", "0.51"],
    ["SF5", "Q2", "0.53", "0.02", "0.02", "0.5", "0.75", "0.03", "0.5", "0.79", "0.06", "0.5"],
]


def parse_pt_float(value: str) -> float:
    return float(value.replace(",", "."))


def build_metrics_list(raw_rows):
    metrics = []
    for row in raw_rows:
        item = {
            "Score": row[0],
            "Query": row[1],
        }
        for idx, key in enumerate(COLUMNS[2:], start=2):
            item[key] = parse_pt_float(row[idx])
        metrics.append(item)
    return metrics


def format_pt(value):
    if isinstance(value, float):
        text = f"{value:.2f}".rstrip("0").rstrip(".")
        return text
    return str(value)


def annotate_bars(ax, bars):
    y_max = ax.get_ylim()[1]
    top_margin = y_max * 0.03
    for bar in bars:
        value = bar.get_height()
        offset = max(0.01, value * 0.015)
        y_pos = value + offset
        va = "bottom"

        # Keep labels inside the plotting area when bars approach the axis top.
        if y_pos > (y_max - top_margin):
            y_pos = y_max - top_margin
            va = "top"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            format_pt(value),
            ha="center",
            va=va,
            fontsize=8,
            rotation=90,
            clip_on=True,
        )


def axis_upper(values, factor=1.22, min_upper=1.0):
    max_value = max(values)
    return max(max_value * factor, min_upper)


def style_axis(ax):
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#8c8c8c")
        spine.set_linewidth(0.8)


def add_compact_legend(ax, *, outside=False):
    legend_kwargs = {
        "title": "Percentage",
        "fontsize": 6,
        "title_fontsize": 6,
        "frameon": True,
    }
    if outside:
        legend_kwargs.update({
            "loc": "upper left",
            "bbox_to_anchor": (1.01, 1.0),
            "borderaxespad": 0.0,
        })
    else:
        legend_kwargs.update({"loc": "upper right"})
    ax.legend(**legend_kwargs)


def render_metrics_dashboard(metrics, output_file: Path):
    labels = [f"{item['Score']}_{item['Query']}" for item in metrics]
    x = np.arange(len(labels))

    auc = [item["AUC"] for item in metrics]
    top1_ef = [item["TOP1_EF"] for item in metrics]
    top5_ef = [item["TOP5_EF"] for item in metrics]
    top10_ef = [item["TOP10_EF"] for item in metrics]
    top1_bedroc = [item["TOP1_BEDROC"] for item in metrics]
    top5_bedroc = [item["TOP5_BEDROC"] for item in metrics]
    top10_bedroc = [item["TOP10_BEDROC"] for item in metrics]
    top1_ccr = [item["TOP1_CCR"] for item in metrics]
    top5_ccr = [item["TOP5_CCR"] for item in metrics]
    top10_ccr = [item["TOP10_CCR"] for item in metrics]

    colors = {
        "top1": "#d97721",
        "top5": "#e59b5b",
        "top10": "#f2c28f",
        "auc": "#e8aa83",
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 7,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), facecolor="#ececec")
    for axis in axes.flat:
        axis.set_facecolor("#f5f5f5")

    bar_w = 0.24

    ax_auc = axes[0, 0]
    bars_auc = ax_auc.bar(x, auc, color=colors["auc"], edgecolor="white")
    ax_auc.set_title("AUC")
    ax_auc.set_ylabel("AUC")
    ax_auc.set_xlabel("Score_Query")
    ax_auc.set_xticks(x)
    ax_auc.set_xticklabels(labels, rotation=50, ha="right")
    ax_auc.set_ylim(0, axis_upper(auc, factor=1.22, min_upper=1.0))
    style_axis(ax_auc)
    annotate_bars(ax_auc, bars_auc)

    ax_ef = axes[0, 1]
    bars_ef1 = ax_ef.bar(x - bar_w, top1_ef, width=bar_w, color=colors["top1"], label="TOP 1%")
    bars_ef5 = ax_ef.bar(x, top5_ef, width=bar_w, color=colors["top5"], label="TOP 5%")
    bars_ef10 = ax_ef.bar(x + bar_w, top10_ef, width=bar_w, color=colors["top10"], label="TOP 10%")
    ax_ef.set_title("EF")
    ax_ef.set_ylabel("EF")
    ax_ef.set_xlabel("Score_Query")
    ax_ef.set_xticks(x)
    ax_ef.set_xticklabels(labels, rotation=50, ha="right")
    ax_ef.set_ylim(0, axis_upper(top1_ef + top5_ef + top10_ef, factor=1.22, min_upper=1.0))
    add_compact_legend(ax_ef)
    style_axis(ax_ef)
    annotate_bars(ax_ef, bars_ef1)
    annotate_bars(ax_ef, bars_ef5)
    annotate_bars(ax_ef, bars_ef10)

    ax_bedroc = axes[1, 0]
    bars_b1 = ax_bedroc.bar(x - bar_w, top1_bedroc, width=bar_w, color=colors["top1"], label="TOP 1%")
    bars_b5 = ax_bedroc.bar(x, top5_bedroc, width=bar_w, color=colors["top5"], label="TOP 5%")
    bars_b10 = ax_bedroc.bar(x + bar_w, top10_bedroc, width=bar_w, color=colors["top10"], label="TOP 10%")
    ax_bedroc.set_title("BEDROC")
    ax_bedroc.set_ylabel("BEDROC")
    ax_bedroc.set_xlabel("Score_Query")
    ax_bedroc.set_xticks(x)
    ax_bedroc.set_xticklabels(labels, rotation=50, ha="right")
    ax_bedroc.set_ylim(0, axis_upper(top1_bedroc + top5_bedroc + top10_bedroc, factor=1.25, min_upper=0.8))
    add_compact_legend(ax_bedroc)
    style_axis(ax_bedroc)
    annotate_bars(ax_bedroc, bars_b1)
    annotate_bars(ax_bedroc, bars_b5)
    annotate_bars(ax_bedroc, bars_b10)

    ax_ccr = axes[1, 1]
    bars_c1 = ax_ccr.bar(x - bar_w, top1_ccr, width=bar_w, color=colors["top1"], label="TOP 1%")
    bars_c5 = ax_ccr.bar(x, top5_ccr, width=bar_w, color=colors["top5"], label="TOP 5%")
    bars_c10 = ax_ccr.bar(x + bar_w, top10_ccr, width=bar_w, color=colors["top10"], label="TOP 10%")
    ax_ccr.set_title("CCR")
    ax_ccr.set_ylabel("CCR")
    ax_ccr.set_xlabel("Score_Query")
    ax_ccr.set_xticks(x)
    ax_ccr.set_xticklabels(labels, rotation=50, ha="right")
    ax_ccr.set_ylim(0, axis_upper(top1_ccr + top5_ccr + top10_ccr, factor=1.22, min_upper=0.8))
    add_compact_legend(ax_ccr)
    style_axis(ax_ccr)
    annotate_bars(ax_ccr, bars_c1)
    annotate_bars(ax_ccr, bars_c5)
    annotate_bars(ax_ccr, bars_c10)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    metrics = build_metrics_list(RAW_ROWS)

    output_dir = Path("output")
    output_image = output_dir / "metrics_table.png"
    output_json = output_dir / "metrics_list.json"

    render_metrics_dashboard(metrics, output_image)
    output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Imagem salva em: {output_image}")
    print(f"Lista salva em: {output_json}")


if __name__ == "__main__":
    main()
