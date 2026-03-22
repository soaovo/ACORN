#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None


def ivecs_mmap(path: str) -> np.ndarray:
    data = np.memmap(path, dtype=np.int32, mode="r")
    dim = int(data[0])
    return data.reshape(-1, dim + 1)[:, 1:]


def fvecs_mmap(path: str) -> np.ndarray:
    return ivecs_mmap(path).view(np.float32)


def xbin_mmap(path: str) -> np.ndarray:
    header = np.fromfile(path, dtype=np.uint32, count=2)
    if header.size != 2:
        raise ValueError("invalid xbin header: %s" % path)
    n, d = int(header[0]), int(header[1])
    if path.endswith(".fbin"):
        return np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(n, d))
    if path.endswith(".u8bin"):
        return np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(n, d))
    raise ValueError("unsupported xbin suffix: %s" % path)


def vectors_mmap(path: str) -> np.ndarray:
    if path.endswith(".fvecs"):
        return fvecs_mmap(path)
    if path.endswith(".fbin") or path.endswith(".u8bin"):
        return xbin_mmap(path)
    raise ValueError("unsupported vector file: %s" % path)


def ivecs_write(path: str, matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix, dtype=np.int32)
    n, d = matrix.shape
    out = np.empty((n, d + 1), dtype=np.int32)
    out[:, 0] = d
    out[:, 1:] = matrix
    out.tofile(path)


def splitmix64_labels(ids: np.ndarray, gamma: int, seed: int) -> np.ndarray:
    x = ids.astype(np.uint64) + np.uint64(seed)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return np.remainder(x, np.uint64(gamma)).astype(np.int32)


def write_txt_labels(path: str, labels: np.ndarray) -> None:
    with open(path, "w", encoding="ascii") as fout:
        for value in labels:
            fout.write(f"{int(value)}\n")


def merge_topk(
        best_d: np.ndarray,
        best_i: np.ndarray,
        cand_d: np.ndarray,
        cand_i: np.ndarray,
        k: int) -> Tuple[np.ndarray, np.ndarray]:
    all_d = np.hstack((best_d, cand_d))
    all_i = np.hstack((best_i, cand_i))
    pick = np.argpartition(all_d, kth=k - 1, axis=1)[:, :k]
    row = np.arange(all_d.shape[0])[:, None]
    new_d = all_d[row, pick]
    new_i = all_i[row, pick]
    order = np.argsort(new_d, axis=1)
    return new_d[row, order], new_i[row, order]


def exact_search_chunk(xq: np.ndarray, xb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if faiss is None:
        raise RuntimeError(
            "Python faiss module is required for GT generation. "
            "Install faiss or run in an environment where it is available.")
    index = faiss.IndexFlatL2(xb.shape[1])
    index.add(np.ascontiguousarray(xb, dtype="float32"))
    return index.search(np.ascontiguousarray(xq, dtype="float32"), k)


def build_groundtruth(
        base_path: str,
        query_path: str,
        base_labels_path: str,
        query_labels_path: str,
        gt_path: str,
        gamma: int,
        k: int,
        base_seed: int,
        query_seed: int,
        base_block: int) -> None:
    xb = vectors_mmap(base_path)
    xq = vectors_mmap(query_path)
    nb, d = xb.shape
    nq, qd = xq.shape
    if d != qd:
        raise ValueError(f"dimension mismatch: base={d}, query={qd}")

    print(f"base vectors: {nb}, query vectors: {nq}, dim: {d}")

    query_ids = np.arange(nq, dtype=np.uint64)
    query_labels = splitmix64_labels(query_ids, gamma, query_seed)
    write_txt_labels(query_labels_path, query_labels)
    print(f"wrote query labels to {query_labels_path}")

    with open(base_labels_path, "w", encoding="ascii") as fout:
        for start in range(0, nb, base_block):
            end = min(start + base_block, nb)
            ids = np.arange(start, end, dtype=np.uint64)
            labels = splitmix64_labels(ids, gamma, base_seed)
            for value in labels:
                fout.write(f"{int(value)}\n")
            done = end / nb * 100.0
            print(f"base labels: {end}/{nb} ({done:.2f}%)")
    print(f"wrote base labels to {base_labels_path}")

    gt = np.full((nq, k), -1, dtype=np.int32)

    for label in range(gamma):
        qids = np.flatnonzero(query_labels == label)
        if qids.size == 0:
            continue

        xq_label = np.ascontiguousarray(xq[qids], dtype="float32")
        best_d = np.full((qids.size, k), np.inf, dtype="float32")
        best_i = np.full((qids.size, k), -1, dtype=np.int64)

        print(f"label {label}: {qids.size} queries")

        for start in range(0, nb, base_block):
            end = min(start + base_block, nb)
            ids = np.arange(start, end, dtype=np.uint64)
            labels = splitmix64_labels(ids, gamma, base_seed)
            mask = labels == label
            if not np.any(mask):
                continue

            local_ids = np.arange(start, end, dtype=np.int64)[mask]
            xb_label = np.ascontiguousarray(xb[start:end][mask], dtype="float32")
            topk = min(k, xb_label.shape[0])
            cand_d, cand_i = exact_search_chunk(xq_label, xb_label, topk)
            cand_i = local_ids[cand_i]

            if topk < k:
                pad_d = np.full((cand_d.shape[0], k - topk), np.inf, dtype="float32")
                pad_i = np.full((cand_i.shape[0], k - topk), -1, dtype=np.int64)
                cand_d = np.hstack((cand_d, pad_d))
                cand_i = np.hstack((cand_i, pad_i))

            best_d, best_i = merge_topk(best_d, best_i, cand_d, cand_i, k)

            if (start // base_block) % 10 == 0 or end == nb:
                done = end / nb * 100.0
                print(f"label {label}: scanned {end}/{nb} ({done:.2f}%)")

        gt[qids] = best_i.astype(np.int32)

    ivecs_write(gt_path, gt)
    print(f"wrote filtered GT to {gt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ACORN-style custom metadata, query filters, and filtered exact GT.")
    parser.add_argument("--base-fvecs", required=True)
    parser.add_argument("--query-fvecs", required=True)
    parser.add_argument("--base-meta-out", required=True)
    parser.add_argument("--query-meta-out", required=True)
    parser.add_argument("--gt-out", required=True)
    parser.add_argument("--gamma", type=int, default=12)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--query-seed", type=int, default=54321)
    parser.add_argument("--base-block", type=int, default=200000)
    args = parser.parse_args()

    if args.gamma <= 0:
        raise ValueError("--gamma must be > 0")
    if args.k <= 0:
        raise ValueError("--k must be > 0")
    if args.base_block <= 0:
        raise ValueError("--base-block must be > 0")

    for path in (args.base_meta_out, args.query_meta_out, args.gt_out):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    build_groundtruth(
        base_path=args.base_fvecs,
        query_path=args.query_fvecs,
        base_labels_path=args.base_meta_out,
        query_labels_path=args.query_meta_out,
        gt_path=args.gt_out,
        gamma=args.gamma,
        k=args.k,
        base_seed=args.base_seed,
        query_seed=args.query_seed,
        base_block=args.base_block)


if __name__ == "__main__":
    main()
