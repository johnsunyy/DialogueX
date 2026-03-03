"""
utils.py — Phase 9: Integration API

Clean, dependency-free interface for loading the model and running predictions.

Usage:
    from word_model_nslt100.utils import load_model, predict_sequence
    model, label_map = load_model()
    result = predict_sequence(sequence_np, model, label_map)
    # {"word": "drink", "confidence": 0.91, "top3": [...]}
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

MODULE_DIR = Path(__file__).resolve().parent
CKPT_PATH  = MODULE_DIR / "checkpoints" / "best_model.pt"
META_DIR   = MODULE_DIR / "data" / "metadata"
WLASL_JSON = MODULE_DIR.parent / "WLASL_v0.3.json"


def get_label_map() -> Dict[int, str]:
    lm_path = META_DIR / "label_map.json"
    if lm_path.exists():
        with open(lm_path) as f:
            return {int(k): v for k, v in json.load(f).items()}
    if WLASL_JSON.exists():
        with open(WLASL_JSON) as f:
            data = json.load(f)
        return {i: e["gloss"] for i, e in enumerate(data)}
    raise FileNotFoundError("No label map found. Run dataset_analysis.py first.")


def load_model(
    checkpoint_path: Union[str, Path] = None,
    device: Union[str, torch.device] = "auto",
) -> Tuple[torch.nn.Module, Dict[int, str]]:
    """
    Load the trained SignModel from checkpoint.
    Returns (model_in_eval_mode, label_map).
    """
    import sys
    sys.path.insert(0, str(MODULE_DIR))
    from model.sign_model import SignModel

    if checkpoint_path is None:
        checkpoint_path = CKPT_PATH
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("cfg", {})

    model = SignModel(
        num_classes  = cfg.get("num_classes",  100),
        seq_len      = cfg.get("seq_len",       40),
        feat_dim     = cfg.get("feat_dim",      324),
        lstm1_hidden = cfg.get("lstm1_hidden",  256),
        lstm2_hidden = cfg.get("lstm2_hidden",  128),
        dropout1     = cfg.get("dropout1",      0.3),
        dropout2     = cfg.get("dropout2",      0.4),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, get_label_map()


def predict_sequence(
    sequence: Union[np.ndarray, torch.Tensor],
    model:     torch.nn.Module            = None,
    label_map: Dict[int, str]             = None,
    device:    Union[str, torch.device]   = "auto",
    top_k:     int                        = 3,
) -> Dict:
    """
    Run inference on a single feature sequence.

    Args:
        sequence  : (40, 324) numpy array or tensor
        model     : loaded SignModel (or None → auto-load from checkpoint)
        label_map : {int: str} (or None → auto-load)
        device    : "auto" | "cpu" | "cuda"
        top_k     : number of top predictions to return

    Returns:
        {"word": str, "confidence": float, "top3": [(word, prob), ...]}
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if model is None or label_map is None:
        _m, _lm = load_model(device=device)
        model     = model     or _m
        label_map = label_map or _lm

    if isinstance(sequence, np.ndarray):
        sequence = torch.from_numpy(sequence.astype(np.float32))
    if sequence.ndim == 2:
        sequence = sequence.unsqueeze(0)   # (1, 40, 324)

    with torch.no_grad():
        probs = F.softmax(model(sequence.to(device)), dim=-1)[0]

    top_k   = min(top_k, probs.size(0))
    vals, idxs = probs.topk(top_k)
    top_list: List[Tuple[str, float]] = [
        (label_map.get(int(i), f"class_{i}"), float(v))
        for i, v in zip(idxs.cpu().numpy(), vals.cpu().numpy())
    ]

    return {"word": top_list[0][0], "confidence": top_list[0][1], "top3": top_list}


def predict_batch(
    sequences: Union[np.ndarray, torch.Tensor],
    model:     torch.nn.Module,
    label_map: Dict[int, str],
    device:    Union[str, torch.device] = "auto",
) -> List[Dict]:
    """Batch inference. sequences: (N, 40, 324)."""
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if isinstance(sequences, np.ndarray):
        sequences = torch.from_numpy(sequences.astype(np.float32))
    with torch.no_grad():
        probs = F.softmax(model(sequences.to(device)), dim=-1)
    results = []
    for p in probs:
        vals, idxs = p.topk(3)
        tl = [(label_map.get(int(i), f"class_{i}"), float(v))
              for i, v in zip(idxs.cpu().numpy(), vals.cpu().numpy())]
        results.append({"word": tl[0][0], "confidence": tl[0][1], "top3": tl})
    return results


if __name__ == "__main__":
    print("Utils API smoke test...")
    try:
        lm = get_label_map()
        print(f"  Label map: {len(lm)} classes, e.g. 0='{lm.get(0)}'")
    except FileNotFoundError as e:
        print(f"  (label_map pending — run dataset_analysis.py): {e}")
    seq = np.zeros((40, 324), dtype=np.float32)
    print(f"  Dummy sequence shape: {seq.shape}  ✓")
    print("  API: predict_sequence(np.ndarray) → dict with 'word','confidence','top3'")
