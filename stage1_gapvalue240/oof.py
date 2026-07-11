from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .errors import ValidationError
from .util import atomic_write_json, sha256_file

RAW_RE = re.compile(r"fold_(?P<fold>[0-9]{2}).*epoch_(?P<epoch>[0-9]{3})_predictions\.csv$", re.I)

@dataclass(frozen=True)
class OOFCache:
    matrix_path: Path
    metadata_path: Path
    sample_ids_path: Path
    shape: tuple[int, int]
    dtype: str = "float64"

    def open(self, mode: str = "r") -> np.memmap:
        return np.memmap(self.matrix_path, dtype=self.dtype, mode=mode, shape=self.shape)


def _resolve_raw_files(raw_root: Path, source_manifest: Path) -> list[tuple[int, str, Path]]:
    mf = pd.read_csv(source_manifest, dtype={"relative_path":"string"})
    if len(mf) != 2000: raise ValidationError(f"Expected 2000 raw OOF files, got {len(mf)}")
    found: list[tuple[int,str,Path]] = []
    for rel in mf["relative_path"]:
        rel_norm = str(rel).replace("\\", "/")
        m = re.search(r"fold_(?P<fold>[0-9]{2})/epoch_(?P<epoch>[0-9]{3})_predictions\.csv$", rel_norm, re.I)
        if not m: raise ValidationError(f"Cannot parse fold/epoch: {rel}")
        path = raw_root / rel_norm
        if not path.exists():
            candidates = list(raw_root.rglob(Path(rel_norm).name))
            candidates = [p for p in candidates if f"fold_{m.group('fold')}" in p.as_posix()]
            if len(candidates) != 1: raise FileNotFoundError(f"Raw OOF file not found uniquely: {rel}")
            path = candidates[0]
        found.append((int(m.group("epoch")), m.group("fold"), path))
    keys={(e,f) for e,f,_ in found}
    expected={(e,f"{i:02d}") for e in range(1,201) for i in range(10)}
    if keys != expected: raise ValidationError(f"Raw OOF coverage mismatch: missing={sorted(expected-keys)[:5]}, extra={sorted(keys-expected)[:5]}")
    return sorted(found)


def build_oof_memmap(raw_root: str | Path, source_manifest: str | Path, master_index: str | Path,
                     cache_dir: str | Path, overwrite: bool = False) -> OOFCache:
    raw_root, cache_dir = Path(raw_root).resolve(), Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = cache_dir / "oof_probabilities_float64.mmap"
    metadata_path = cache_dir / "oof_probabilities_metadata.json"
    sample_ids_path = cache_dir / "sample_ids.csv"
    if matrix_path.exists() and not overwrite:
        meta=json.loads(metadata_path.read_text(encoding="utf-8"))
        return OOFCache(matrix_path, metadata_path, sample_ids_path, tuple(meta["shape"]), meta["dtype"])
    master = pd.read_csv(master_index, usecols=["sample_id","y_true","oof_fold"], dtype={"sample_id":"string","oof_fold":"string"})
    master["oof_fold"] = master["oof_fold"].str.zfill(2)
    if len(master)!=120000 or master.sample_id.duplicated().any(): raise ValidationError("Invalid master index")
    id_to_idx = pd.Series(np.arange(len(master), dtype=np.int64), index=master.sample_id).to_dict()
    files = _resolve_raw_files(raw_root, Path(source_manifest))
    tmp_path = matrix_path.with_suffix(".mmap.tmp")
    if tmp_path.exists(): tmp_path.unlink()
    matrix = np.memmap(tmp_path, dtype="float64", mode="w+", shape=(200, len(master)))
    matrix[:] = np.nan
    seen = np.zeros((200, len(master)), dtype=np.bool_)
    for epoch, fold, path in files:
        df = pd.read_csv(path, usecols=["sample_id","y_true","p_defect_raw"], dtype={"sample_id":"string"})
        idx = df["sample_id"].map(id_to_idx)
        if idx.isna().any(): raise ValidationError(f"Unknown sample IDs in {path}")
        idx_arr=idx.to_numpy(np.int64)
        expected_fold=master.iloc[idx_arr]["oof_fold"].to_numpy()
        if not np.all(expected_fold == fold): raise ValidationError(f"Fold mismatch in {path}")
        labels=master.iloc[idx_arr]["y_true"].to_numpy(np.int8)
        if not np.array_equal(labels,df["y_true"].to_numpy(np.int8)): raise ValidationError(f"Label mismatch in {path}")
        if seen[epoch-1,idx_arr].any(): raise ValidationError(f"Duplicate predictions in epoch {epoch} fold {fold}")
        matrix[epoch-1,idx_arr]=df["p_defect_raw"].to_numpy(np.float64)
        seen[epoch-1,idx_arr]=True
    matrix.flush()
    if not seen.all() or np.isnan(matrix).any(): raise ValidationError("OOF cache incomplete")
    del matrix
    tmp_path.replace(matrix_path)
    master[["sample_id","y_true","oof_fold"]].to_csv(sample_ids_path,index=False)
    meta={
        "shape":[200,len(master)],"dtype":"float64","raw_file_count":len(files),"prediction_rows":int(seen.sum()),
        "source_manifest_sha256":sha256_file(source_manifest),"master_index_sha256":sha256_file(master_index),
        "epoch_base":1,"fold_values":[f"{i:02d}" for i in range(10)]
    }
    atomic_write_json(metadata_path,meta,overwrite=overwrite)
    return OOFCache(matrix_path,metadata_path,sample_ids_path,(200,len(master)))


def compute_epoch_gap_metrics(matrix: np.ndarray, labels: np.ndarray, folds: np.ndarray | None = None,
                              exclude_fold_epoch: tuple[str,int] | None = None) -> pd.DataFrame:
    rows=[]
    for e in range(matrix.shape[0]):
        keep=np.ones(matrix.shape[1],dtype=bool)
        if exclude_fold_epoch and e+1==exclude_fold_epoch[1]: keep &= folds != exclude_fold_epoch[0]
        p=matrix[e,keep]; y=labels[keep]
        n=p[y==0]; d=p[y==1]
        rows.append({"epoch":e+1,"row_count":len(p),"normal_count":len(n),"defect_count":len(d),
                     "normal_q68":float(np.quantile(n,.68)),"normal_q90":float(np.quantile(n,.90)),"normal_mean":float(n.mean()),
                     "defect_q50":float(np.quantile(d,.50)),"defect_q05":float(np.quantile(d,.05)),"defect_mean":float(d.mean()),
                     "gap_q68_q050":float(np.quantile(d,.50)-np.quantile(n,.68)),
                     "tail_gap_q90_q05":float(np.quantile(d,.05)-np.quantile(n,.90))})
    return pd.DataFrame(rows)


def score_from_windows(matrix: np.ndarray, good_epochs: Iterable[int], bad_epochs: Iterable[int]) -> np.ndarray:
    good=np.array(sorted(set(int(e) for e in good_epochs)),dtype=int)-1
    bad=np.array(sorted(set(int(e) for e in bad_epochs)),dtype=int)-1
    if len(good)==0 or len(bad)==0: raise ValueError("Empty epoch window")
    return np.nanmean(matrix[bad],axis=0)-np.nanmean(matrix[good],axis=0)
