"""
prepare_splits.py — Phase 4: CSV Split Preparation

Uses the OFFICIAL train/val/test split from nslt_100.json directly.
No re-splitting — val stays val, test stays test.

Each CSV has columns: video_path,label_index

Run: python preprocessing/prepare_splits.py
"""

import json
import csv
from pathlib import Path

# ─── Paths ──────────────────────────────────────────────────────────────────
MODULE_DIR  = Path(__file__).resolve().parent.parent          # word_model_nslt300/
ROOT        = MODULE_DIR.parent                               # d:\haha
NSLT_JSON   = ROOT / "nslt_100.json"
WLASL_JSON  = ROOT / "WLASL_v0.3.json"
VIDEOS_DIR  = ROOT / "videos"
SPLITS_DIR  = MODULE_DIR / "data" / "splits"
META_DIR    = MODULE_DIR / "data" / "metadata"


def load_label_map():
    with open(WLASL_JSON, "r") as f:
        data = json.load(f)
    return {i: entry["gloss"] for i, entry in enumerate(data)}


def load_nslt():
    with open(NSLT_JSON, "r") as f:
        return json.load(f)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "label_index"])
        for row in rows:
            writer.writerow(row)
    print(f"  → Saved {len(rows)} rows to {path.relative_to(ROOT)}")


def prepare_splits():
    print("=" * 60)
    print("  Preparing Official Train / Val / Test Splits (NSLT-100)")
    print("=" * 60)

    nslt      = load_nslt()
    idx2gloss = load_label_map()

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    # ── Use official subsets directly ─────────────────────────
    splits = {"train": [], "val": [], "test": []}
    skipped = 0

    for vid_id, info in nslt.items():
        cls_idx = info["action"][0]
        subset  = info["subset"]
        vid_path = VIDEOS_DIR / f"{vid_id}.mp4"

        if not vid_path.exists():
            skipped += 1
            continue

        splits[subset].append((str(vid_path), cls_idx))

    print(f"\n  Skipped {skipped} missing video files")
    print(f"  Train (official) : {len(splits['train'])} videos")
    print(f"  Val   (official) : {len(splits['val'])} videos")
    print(f"  Test  (official) : {len(splits['test'])} videos")

    # ── Write CSVs ────────────────────────────────────────────
    write_csv(SPLITS_DIR / "train.csv", splits["train"])
    write_csv(SPLITS_DIR / "val.csv",   splits["val"])
    write_csv(SPLITS_DIR / "test.csv",  splits["test"])

    # ── Save label map ────────────────────────────────────────
    all_classes = set(row[1] for rows in splits.values() for row in rows)
    label_map   = {str(k): idx2gloss.get(k, f"class_{k}") for k in sorted(all_classes)}
    label_map_path = META_DIR / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\n  label_map.json saved → {label_map_path.relative_to(ROOT)}")
    print(f"  Total classes in splits: {len(all_classes)}")
    print("=" * 60)


if __name__ == "__main__":
    prepare_splits()
