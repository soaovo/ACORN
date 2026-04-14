#!/usr/bin/env python3
"""
固定索引参数，改变 efSearch —— 绘制 Recall-延迟 曲线

输入: parse_acorn_logs.py 生成的 summary.csv
输出: recall_latency.png (以及可选的 recall_latency.pdf)

用法:
    python scripts/plot_recall_latency.py --csv logs/acorn_experiments/fig1_efsearch_recall_latency/summary.csv
    python scripts/plot_recall_latency.py --csv summary.csv --out result.png
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def load_csv(csv_path: str):
    """读取 summary.csv，返回按 efSearch 排序的行列表。"""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def extract_series(rows, index_type: str):
    """
    从 CSV 行中提取 (recall, latency_ms, efsearch) 三元组列表。
    index_type: "hnsw" 或 "acorn"
    """
    recall_key = f"{index_type}_recall_at_10"
    latency_key = f"{index_type}_ms_per_query"
    efsearch_key = "efsearch"

    series = []
    for row in rows:
        recall_str = row.get(recall_key, "")
        latency_str = row.get(latency_key, "")
        efs_str = row.get(efsearch_key, "")

        if not recall_str or not latency_str or not efs_str:
            continue

        recall = float(recall_str)
        latency = float(latency_str)
        efs = int(float(efs_str))
        series.append((recall, latency, efs))

    series.sort(key=lambda t: t[2])
    return series


def plot(hnsw_series, acorn_series, out_path: str, title: str = ""):
    """绘制 Recall@10 vs Latency (ms/query) 曲线。"""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # ---- HNSW ----
    if hnsw_series:
        recalls = [s[0] for s in hnsw_series]
        latencies = [s[1] for s in hnsw_series]
        efs_vals = [s[2] for s in hnsw_series]
        ax.plot(recalls, latencies, "o-", color="#1f77b4", linewidth=2,
                markersize=7, label="HNSW (base)", zorder=3)
        for r, l, e in zip(recalls, latencies, efs_vals):
            ax.annotate(f"ef={e}", (r, l), textcoords="offset points",
                        xytext=(6, 6), fontsize=7, color="#1f77b4", alpha=0.8)

    # ---- ACORN ----
    if acorn_series:
        recalls = [s[0] for s in acorn_series]
        latencies = [s[1] for s in acorn_series]
        efs_vals = [s[2] for s in acorn_series]
        ax.plot(recalls, latencies, "s-", color="#d62728", linewidth=2,
                markersize=7, label="ACORN", zorder=3)
        for r, l, e in zip(recalls, latencies, efs_vals):
            ax.annotate(f"ef={e}", (r, l), textcoords="offset points",
                        xytext=(6, -10), fontsize=7, color="#d62728", alpha=0.8)

    ax.set_xlabel("Recall@10", fontsize=13)
    ax.set_ylabel("Latency (ms / query)", fontsize=13)
    ax.set_title(title or "Fixed Index — Recall vs Latency (varying efSearch)", fontsize=14)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Plot saved to {out_path}")

    # 也保存一份 PDF
    pdf_path = str(Path(out_path).with_suffix(".pdf"))
    fig.savefig(pdf_path)
    print(f"PDF  saved to {pdf_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="绘制固定索引、改变 efSearch 的 Recall-延迟 曲线")
    parser.add_argument("--csv", required=True, help="parse_acorn_logs.py 输出的 summary.csv")
    parser.add_argument("--out", default=None, help="输出图片路径 (默认与 csv 同目录)")
    parser.add_argument("--title", default="", help="图表标题")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    rows = load_csv(str(csv_path))
    if not rows:
        sys.exit("CSV is empty — no data rows found.")

    hnsw_series = extract_series(rows, "hnsw")
    acorn_series = extract_series(rows, "acorn")

    if not hnsw_series and not acorn_series:
        print("WARNING: no recall/latency data found in CSV. Available columns:")
        print("  ", list(rows[0].keys()))
        sys.exit(1)

    out_path = args.out or str(csv_path.parent / "recall_latency.png")
    plot(hnsw_series, acorn_series, out_path, title=args.title)


if __name__ == "__main__":
    main()
