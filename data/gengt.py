#!/usr/bin/env python3
import argparse, pathlib, numpy as np, faiss, h5py
from faiss.contrib.datasets import fvecs_read   # 新增

def load_fvecs(path):
    return fvecs_read(str(path))                # 替换调用

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=pathlib.Path, required=True)
parser.add_argument("--query", type=pathlib.Path, required=True)
parser.add_argument("--base-meta", type=pathlib.Path, required=True)
parser.add_argument("--query-meta", type=pathlib.Path, required=True)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--out", type=pathlib.Path, required=True)
args = parser.parse_args()

xb = load_fvecs(args.base.expanduser())
xq = load_fvecs(args.query.expanduser())
base_meta = np.loadtxt(args.base_meta.expanduser(), dtype=np.int32)
query_meta = np.loadtxt(args.query_meta.expanduser(), dtype=np.int32)

index = faiss.IndexFlatL2(xb.shape[1])
with h5py.File(args.out.expanduser(), "w") as h5:
    ds_I = h5.create_dataset("neighbors", (xq.shape[0], args.k), dtype=np.int32, fillvalue=-1)
    ds_D = h5.create_dataset("distances", (xq.shape[0], args.k), dtype=np.float32, fillvalue=np.inf)

    for qi in range(xq.shape[0]):
        mask = (base_meta == query_meta[qi])
        cand_idx = np.where(mask)[0]
        if cand_idx.size == 0:
            continue
        xb_sub = xb[cand_idx]
        index.reset()
        index.add(xb_sub)
        D, I = index.search(xq[qi:qi+1], min(args.k, xb_sub.shape[0]))
        ds_I[qi, :I.shape[1]] = cand_idx[I[0]]
        ds_D[qi, :I.shape[1]] = D[0]
print(f"Saved filtered GT to {args.out}")
