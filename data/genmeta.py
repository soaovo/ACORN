#!/usr/bin/env python3
import argparse
import numpy as np
import pathlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_base", type=int, default=1_000_000,
                        help="基向量数量，默认 1,000,000")
    parser.add_argument("--n_query", type=int, default=10_000,
                        help="查询向量数量，默认 10,000")
    parser.add_argument("--low", type=int, default=1,
                        help="随机整数的最小值（含）")
    parser.add_argument("--high", type=int, default=100,
                        help="随机整数的最大值（含）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机数种子，便于复现")
    parser.add_argument("--out_base", type=pathlib.Path,
                        default=pathlib.Path("~/ACORN/data/metadata.txt"),
                        help="base metadata 输出路径")
    parser.add_argument("--out_query", type=pathlib.Path,
                        default=pathlib.Path("~/ACORN/data/query_meta.txt"),
                        help="query metadata 输出路径")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    base_meta = rng.integers(args.low, args.high + 1,
                             size=args.n_base, dtype=np.int16)
    query_meta = rng.integers(args.low, args.high + 1,
                              size=args.n_query, dtype=np.int16)

    out_base = args.out_base.expanduser()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_base, base_meta, fmt="%d")

    out_query = args.out_query.expanduser()
    out_query.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_query, query_meta, fmt="%d")

    print(f"Saved base metadata to {out_base}")
    print(f"Saved query metadata to {out_query}")


if __name__ == "__main__":
    main()
