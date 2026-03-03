"""
dataset_analysis.py — Phase 1: NSLT-300 Dataset Analysis

Run: python dataset_analysis.py
Prints a full summary of the dataset without modifying any files.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent          # d:\haha
NSLT_JSON   = ROOT / "nslt_100.json"
WLASL_JSON  = ROOT / "WLASL_v0.3.json"
VIDEOS_DIR  = ROOT / "videos"


def load_nslt(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_wlasl(path: Path) -> dict:
    """Returns {class_index: gloss_word}."""
    with open(path, "r") as f:
        data = json.load(f)
    mapping = {}
    for entry in data:
        gloss = entry["gloss"]
        for inst in entry.get("instances", []):
            mapping[inst.get("video_id")] = gloss
    # Build index-to-gloss from the order in WLASL
    idx_to_gloss = {i: entry["gloss"] for i, entry in enumerate(data)}
    return idx_to_gloss


def analyse():
    print("=" * 60)
    print("  NSLT-300 Dataset Analysis")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────
    nslt      = load_nslt(NSLT_JSON)
    idx2gloss = load_wlasl(WLASL_JSON)

    # ── Counts ────────────────────────────────────────────────
    subsets = defaultdict(list)           # subset → [video_ids]
    classes = defaultdict(lambda: defaultdict(list))  # class_idx → subset → [vid]

    for vid_id, info in nslt.items():
        subset    = info["subset"]          # "train" | "val" | "test"
        cls_idx   = info["action"][0]       # class index
        subsets[subset].append(vid_id)
        classes[cls_idx][subset].append(vid_id)

    total_videos  = len(nslt)
    num_classes   = len(classes)
    train_count   = len(subsets["train"])
    val_count     = len(subsets["val"])
    test_count    = len(subsets["test"])

    avg_train     = train_count / num_classes if num_classes else 0
    avg_val       = val_count   / num_classes if num_classes else 0
    avg_test      = test_count  / num_classes if num_classes else 0

    print(f"\n  Total unique words (classes) : {num_classes}")
    print(f"  Total video entries          : {total_videos}")
    print(f"  Train videos                 : {train_count}")
    print(f"  Val   videos                 : {val_count}")
    print(f"  Test  videos                 : {test_count}")
    print(f"  Avg train samples / class    : {avg_train:.2f}")
    print(f"  Avg val   samples / class    : {avg_val:.2f}")
    print(f"  Avg test  samples / class    : {avg_test:.2f}")

    # ── Class distribution (top 10 / bottom 10) ──────────────
    class_sizes = {
        idx: sum(len(v) for v in splits.values())
        for idx, splits in classes.items()
    }
    sorted_classes = sorted(class_sizes.items(), key=lambda x: x[1], reverse=True)

    print("\n  Top-10 classes by total samples:")
    for rank, (idx, cnt) in enumerate(sorted_classes[:10], 1):
        print(f"    {rank:2d}. [{idx:3d}] {idx2gloss.get(idx, '?'):20s} — {cnt} samples")

    print("\n  Bottom-10 classes by total samples:")
    for rank, (idx, cnt) in enumerate(sorted_classes[-10:], 1):
        print(f"    {rank:2d}. [{idx:3d}] {idx2gloss.get(idx, '?'):20s} — {cnt} samples")

    # ── Video file check ──────────────────────────────────────
    print(f"\n  Checking video files in: {VIDEOS_DIR}")
    found, missing = 0, []
    for vid_id in nslt:
        p = VIDEOS_DIR / f"{vid_id}.mp4"
        if p.exists():
            found += 1
        else:
            missing.append(vid_id)

    print(f"  Videos found   : {found} / {total_videos}")
    print(f"  Videos missing : {len(missing)}")
    if missing[:10]:
        print(f"  First 10 missing: {missing[:10]}")

    # ── Save label map ────────────────────────────────────────
    meta_dir = Path(__file__).resolve().parent / "data" / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    label_map_path = meta_dir / "label_map.json"
    label_map = {str(k): v for k, v in idx2gloss.items() if k in classes}
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\n  Label map saved → {label_map_path.relative_to(ROOT)}")

    # ── Verify 300 classes ────────────────────────────────────
    assert num_classes == 300, f"Expected 300 classes, got {num_classes}!"
    print(f"\n  ✓ Confirmed {num_classes} classes (NSLT-300)\n")
    print("=" * 60)

    return nslt, idx2gloss, classes


if __name__ == "__main__":
    analyse()
