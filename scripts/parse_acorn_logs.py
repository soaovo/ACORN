#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path


EXP_RE = re.compile(r"^(EXP_[A-Z0-9_]+)=(.*)$")
KV_INT_RE = re.compile(r"^(N|gamma|M|M_beta|pathwise_width|pathwise_growth_interval|efSearch|edgewise_nt|n1|n2|n3 \(number distance comps at level 0\)|ndis|nreorder):\s+(.+)$")
QUERY_N_RE = re.compile(r"query vecs data loaded, with dim:\s*\d+,\s*nb=(\d+)")
QUERY_TIME_RE = re.compile(r"\*\*\* Query time:\s*([0-9.]+)")
RECALL_RE = re.compile(r"^(HNSW|ACORN) Recall@(\d+):\s*([0-9.]+)")
MAX_NEIGH_RE = re.compile(r"\* stats on level (\d+), max (\d+) neighbors per vertex:")
NEIGH_PER_NODE_RE = re.compile(r"2\.\s+neighbors per node:\s*([0-9.]+)\s+\((\d+)\)")
AVG_DIST_RE = re.compile(r"average distance computations per query:\s*([0-9.]+)")


def maybe_number(value: str):
    value = value.strip()
    if value == "":
        return value
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_log(path: Path):
    data = {
        "log_path": str(path),
        "file_name": path.name,
    }

    in_pre_search_stats = True
    index_stats_ctx = None
    level_stats_ctx = None
    query_ctx = None
    profile_ctx = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            exp_match = EXP_RE.match(line)
            if exp_match:
                key, value = exp_match.groups()
                data[key[4:].lower()] = maybe_number(value)
                continue

            if "====================Search Results====================" in line:
                in_pre_search_stats = False

            qn = QUERY_N_RE.search(line)
            if qn:
                data["nq"] = int(qn.group(1))

            if in_pre_search_stats:
                if line == "============ BASE INDEX =============":
                    index_stats_ctx = "base"
                    level_stats_ctx = None
                    continue
                if line == "============ ACORN INDEX =============":
                    index_stats_ctx = "acorn"
                    level_stats_ctx = None
                    continue

                max_neigh_match = MAX_NEIGH_RE.search(line)
                if max_neigh_match and index_stats_ctx:
                    level = int(max_neigh_match.group(1))
                    max_neighbors = int(max_neigh_match.group(2))
                    level_stats_ctx = level
                    data[f"{index_stats_ctx}_level{level}_max_neighbors"] = max_neighbors
                    continue

                neigh_per_node_match = NEIGH_PER_NODE_RE.search(line)
                if neigh_per_node_match and index_stats_ctx is not None and level_stats_ctx is not None:
                    data[f"{index_stats_ctx}_level{level_stats_ctx}_neighbors_per_node"] = float(
                        neigh_per_node_match.group(1)
                    )
                    data[f"{index_stats_ctx}_level{level_stats_ctx}_total_neighbors"] = int(
                        neigh_per_node_match.group(2)
                    )
                    continue

            if line == "====================HNSW INDEX====================":
                query_ctx = "hnsw"
                continue
            if line == "==================== ACORN INDEX ====================":
                query_ctx = "acorn"
                continue

            query_time_match = QUERY_TIME_RE.search(line)
            if query_time_match and query_ctx:
                data[f"{query_ctx}_query_time_s"] = float(query_time_match.group(1))
                continue

            recall_match = RECALL_RE.search(line)
            if recall_match:
                family = recall_match.group(1).lower()
                rank = int(recall_match.group(2))
                recall = float(recall_match.group(3))
                data[f"{family}_recall_at_{rank}"] = recall
                if rank == 10:
                    data[f"{family}_recall_at_10"] = recall
                continue

            if line == "============= BASE HNSW QUERY PROFILING STATS =============":
                profile_ctx = "hnsw"
                continue
            if line == "============= ACORN QUERY PROFILING STATS =============":
                profile_ctx = "acorn"
                continue

            if profile_ctx:
                int_match = KV_INT_RE.match(line)
                if int_match:
                    key_name = int_match.group(1)
                    value = int_match.group(2)
                    normalized_key = key_name
                    normalized_key = normalized_key.replace(" (number distance comps at level 0)", "")
                    normalized_key = normalized_key.replace(" ", "_").lower()
                    data[f"{profile_ctx}_{normalized_key}"] = maybe_number(value)
                    continue

                avg_dist_match = AVG_DIST_RE.search(line)
                if avg_dist_match:
                    data[f"{profile_ctx}_avg_distance_comps_per_query"] = float(avg_dist_match.group(1))
                    continue

    nq = data.get("nq")
    if isinstance(nq, int) and nq > 0:
        for prefix in ("hnsw", "acorn"):
            query_time_s = data.get(f"{prefix}_query_time_s")
            n3_total = data.get(f"{prefix}_n3")
            if isinstance(query_time_s, (int, float)):
                data[f"{prefix}_ms_per_query"] = (query_time_s * 1000.0) / nq
            if isinstance(n3_total, (int, float)):
                data[f"{prefix}_n3_per_query"] = float(n3_total) / nq

    return data


def collect_logs(log_dir: Path):
    return sorted(p for p in log_dir.rglob("*.log") if p.is_file())


def main():
    parser = argparse.ArgumentParser(description="Parse test_acorn experiment logs into a CSV table.")
    parser.add_argument("--log-dir", required=True, help="Directory containing .log files.")
    parser.add_argument("--csv", required=True, help="Output CSV path.")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise SystemExit(f"Log directory does not exist: {log_dir}")

    rows = [parse_log(path) for path in collect_logs(log_dir)]
    if not rows:
        raise SystemExit(f"No .log files found under: {log_dir}")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    output_path = Path(args.csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
