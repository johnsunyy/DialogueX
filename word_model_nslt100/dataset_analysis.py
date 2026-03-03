"""
dataset_analysis.py — Phase 1: NSLT-100 Dataset Analysis

Run: python dataset_analysis.py
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
NSLT_JSON  = ROOT / "nslt_100.json"
WLASL_JSON = ROOT / "WLASL_v0.3.json"
VIDEOS_DIR = ROOT / "videos"


def analyse():
    print("=" * 60)
    print("  NSLT-100 Dataset Analysis")
    print("=" * 60)

    with open(NSLT_JSON) as f:
        nslt = json.load(f)
    with open(WLASL_JSON) as f:
        wlasl = json.load(f)

    idx2gloss = {i: e["gloss"] for i, e in enumerate(wlasl)}

    subsets = defaultdict(list)
    classes = defaultdict(lambda: defaultdict(list))

    for vid_id, info in nslt.items():
        subset  = info["subset"]
        cls_idx = info["action"][0]
        subsets[subset].append(vid_id)
        classes[cls_idx][subset].append(vid_id)

    total   = len(nslt)
    n_cls   = len(classes)
    tr      = len(subsets["train"])
    va      = len(subsets["val"])
    te      = len(subsets["test"])

    print(f"\n  Classes          : {n_cls}")
    print(f"  Total entries    : {total}")
    print(f"  Train (official) : {tr}")
    print(f"  Val   (official) : {va}")
    print(f"  Test  (official) : {te}")
    print(f"  Avg train/class  : {tr/n_cls:.2f}")

    sizes = {idx: sum(len(v) for v in sp.values()) for idx, sp in classes.items()}
    ranked = sorted(sizes.items(), key=lambda x: x[1], reverse=True)

    print("\n  Top-10 classes:")
    for rank, (idx, cnt) in enumerate(ranked[:10], 1):
        print(f"    {rank:2d}. [{idx:3d}] {idx2gloss.get(idx,'?'):20s} — {cnt}")

    # Video file check
    found = sum(1 for v in nslt if (VIDEOS_DIR / f"{v}.mp4").exists())
    print(f"\n  Videos on disk   : {found} / {total}")

    # Save label map
    meta = Path(__file__).resolve().parent / "data" / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    lm = {str(k): idx2gloss.get(k, f"class_{k}") for k in sorted(classes)}
    with open(meta / "label_map.json", "w") as f:
        json.dump(lm, f, indent=2)
    print(f"  label_map.json   : saved ({len(lm)} classes)")

    assert n_cls == 100, f"Expected 100 classes, got {n_cls}"
    print(f"\n  ✓ Confirmed {n_cls} classes (NSLT-100)")
    print("=" * 60)


if __name__ == "__main__":
    analyse()
