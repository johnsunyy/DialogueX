"""
prepare_splits.py — Phase 4: CSV Split Preparation

Official split strategy:
  - Test  : official 'test' subset from nslt_100.json (UNTOUCHED)
  - Train : official 'train' subset → stratified 90/10 → train + val

Each CSV row: feature_path,label_index
(feature_path points to the pre-extracted .npy file)

Run: python preprocessing/prepare_splits.py
"""

import json
import csv
import random
from pathlib import Path
from collections import defaultdict

MODULE_DIR  = Path(__file__).resolve().parent.parent
ROOT        = MODULE_DIR.parent
NSLT_JSON   = ROOT / "nslt_100.json"
WLASL_JSON  = ROOT / "WLASL_v0.3.json"
VIDEOS_DIR  = ROOT / "videos"
FEATURES_DIR= MODULE_DIR / "data" / "features"
SPLITS_DIR  = MODULE_DIR / "data" / "splits"
META_DIR    = MODULE_DIR / "data" / "metadata"

SEED = 42


def load_label_map():
    with open(WLASL_JSON) as f:
        data = json.load(f)
    return {i: e["gloss"] for i, e in enumerate(data)}


def safe_gloss(gloss):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in gloss)


def stratified_split(by_class, val_ratio=0.10, seed=SEED):
    """Stratified split: {cls: [video_ids]} → (train_pairs, val_pairs)"""
    rng = random.Random(seed)
    train, val = [], []
    for cls_idx, vids in by_class.items():
        shuffled = list(vids); rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_ratio))
        val   += [(v, cls_idx) for v in shuffled[:n_val]]
        train += [(v, cls_idx) for v in shuffled[n_val:]]
    return train, val


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_path", "label_index"])
        for r in rows:
            w.writerow(r)
    print(f"  → {path.name}: {len(rows)} rows")


def prepare_splits():
    print("=" * 60)
    print("  NSLT-100 Official Splits (feature_path CSVs)")
    print("=" * 60)

    with open(NSLT_JSON) as f:
        nslt = json.load(f)
    idx2gloss = load_label_map()

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    # Partition — only include videos whose .npy exists
    train_by_class = defaultdict(list)
    test_rows      = []
    skipped_nofile = 0
    skipped_nonpy  = 0

    for vid_id, info in nslt.items():
        cls_idx = info["action"][0]
        subset  = info["subset"]
        gloss   = idx2gloss.get(cls_idx, f"class_{cls_idx}")
        npy     = FEATURES_DIR / safe_gloss(gloss) / f"{vid_id}.npy"

        if not (VIDEOS_DIR / f"{vid_id}.mp4").exists():
            skipped_nofile += 1
            continue
        if not npy.exists():
            skipped_nonpy += 1
            continue

        if subset == "train":
            train_by_class[cls_idx].append(vid_id)
        elif subset == "test":
            test_rows.append((str(npy), cls_idx))
        # val from JSON is intentionally ignored — we create our own from train

    print(f"\n  Skipped (no video)  : {skipped_nofile}")
    print(f"  Skipped (no .npy)   : {skipped_nonpy}")
    print(f"  Train pool          : {sum(len(v) for v in train_by_class.values())}")
    print(f"  Test  pool          : {len(test_rows)}")

    train_pairs, val_pairs = stratified_split(train_by_class)

    def pairs_to_rows(pairs):
        rows = []
        for vid_id, cls_idx in pairs:
            gloss = idx2gloss.get(cls_idx, f"class_{cls_idx}")
            npy   = FEATURES_DIR / safe_gloss(gloss) / f"{vid_id}.npy"
            rows.append((str(npy), cls_idx))
        return rows

    train_rows = pairs_to_rows(train_pairs)
    val_rows   = pairs_to_rows(val_pairs)

    print(f"\n  Train: {len(train_rows)} | Val: {len(val_rows)} | Test: {len(test_rows)}")

    write_csv(SPLITS_DIR / "train.csv", train_rows)
    write_csv(SPLITS_DIR / "val.csv",   val_rows)
    write_csv(SPLITS_DIR / "test.csv",  test_rows)

    # Label map
    all_cls = set(train_by_class) | {r[1] for r in test_rows}
    lm      = {str(k): idx2gloss.get(k, f"class_{k}") for k in sorted(all_cls)}
    with open(META_DIR / "label_map.json", "w") as f:
        json.dump(lm, f, indent=2)
    print(f"  label_map.json: {len(lm)} classes")
    print("=" * 60)


if __name__ == "__main__":
    prepare_splits()
