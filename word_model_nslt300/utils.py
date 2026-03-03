"""
utils.py — Phase 9: Integration-Ready API

Provides a clean, dependency-free interface for loading the model and
running predictions on pre-extracted feature sequences.

Usage example (stand-alone):
    from word_model_nslt300.utils import load_model, predict_sequence
    model, label_map = load_model()
    result = predict_sequence(sequence_np, model, label_map)
    # result = {"word": "hello", "confidence": 0.93, "top3": [...]}
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

MODULE_DIR  = Path(__file__).resolve().parent
CKPT_PATH   = MODULE_DIR / "checkpoints" / "best_model.pt"
META_DIR    = MODULE_DIR / "data" / "metadata"
WLASL_JSON  = MODULE_DIR.parent / "WLASL_v0.3.json"


def get_label_map(source: str = "auto") -> Dict[int, str]:
    """
    Returns {class_index: word_gloss} mapping.
    
    Args:
        source: "auto" | "json" | "metadata"
            "metadata" — read from data/metadata/label_map.json
            "json"     — read from WLASL_v0.3.json
            "auto"     — metadata first, fallback to json
    """
    label_map_path = META_DIR / "label_map.json"

    if source in ("auto", "metadata") and label_map_path.exists():
        with open(label_map_path, "r") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    if source in ("auto", "json") and WLASL_JSON.exists():
        with open(WLASL_JSON, "r") as f:
            wlasl = json.load(f)
        return {i: entry["gloss"] for i, entry in enumerate(wlasl)}

    raise FileNotFoundError(
        "Cannot find label map. Run prepare_splits.py or dataset_analysis.py first."
    )


def load_model(
    checkpoint_path: Union[str, Path] = None,
    device: Union[str, torch.device] = "auto",
) -> Tuple[torch.nn.Module, Dict[int, str]]:
    """
    Load the trained sign language model from a checkpoint.

    Args:
        checkpoint_path : path to .pt checkpoint (default: checkpoints/best_model.pt)
        device          : "auto" | "cpu" | "cuda" | torch.device

    Returns:
        (model, label_map) — model in eval mode, label_map dict
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

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt.get("cfg", {})

    model = SignModel(
        num_classes = cfg.get("num_classes", 300),
        seq_len     = cfg.get("seq_len",     30),
        feat_dim    = cfg.get("feat_dim",    162),
        lstm_hidden = cfg.get("lstm_hidden", 256),
        dropout1    = cfg.get("dropout1",    0.3),
        dropout2    = cfg.get("dropout2",    0.4),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    label_map = get_label_map()
    return model, label_map


def predict_sequence(
    sequence: Union[np.ndarray, torch.Tensor],
    model: torch.nn.Module = None,
    label_map: Dict[int, str] = None,
    device: Union[str, torch.device] = "auto",
    top_k: int = 3,
) -> Dict:
    """
    Run inference on a single pre-extracted feature sequence.

    Args:
        sequence  : numpy array or tensor of shape (30, 162)
        model     : loaded SignModel (or None → loads from default checkpoint)
        label_map : {int: str} mapping (or None → auto-loaded)
        device    : "auto" | "cpu" | "cuda"
        top_k     : how many top predictions to return (default 3)

    Returns:
        {
          "word"       : str,           # top-1 predicted word
          "confidence" : float,         # top-1 probability
          "top3"       : [(word, prob), ...]  # top-k predictions
        }
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if model is None or label_map is None:
        _model, _lmap = load_model(device=device)
        model     = _model     if model     is None else model
        label_map = _lmap      if label_map is None else label_map

    # ── Prepare input ─────────────────────────────────────────
    if isinstance(sequence, np.ndarray):
        sequence = torch.from_numpy(sequence.astype(np.float32))

    if sequence.ndim == 2:
        sequence = sequence.unsqueeze(0)   # (1, 30, 162)

    sequence = sequence.to(device)

    # ── Inference ─────────────────────────────────────────────
    with torch.no_grad():
        logits = model(sequence)                    # (1, num_classes)
        probs  = F.softmax(logits, dim=-1)[0]       # (num_classes,)

    top_k   = min(top_k, probs.size(0))
    vals, idxs = probs.topk(top_k)
    vals    = vals.cpu().numpy()
    idxs    = idxs.cpu().numpy()

    top_list: List[Tuple[str, float]] = [
        (label_map.get(int(i), f"class_{i}"), float(v))
        for i, v in zip(idxs, vals)
    ]

    return {
        "word":       top_list[0][0],
        "confidence": top_list[0][1],
        "top3":       top_list,
    }


def predict_batch(
    sequences: Union[np.ndarray, torch.Tensor],
    model: torch.nn.Module,
    label_map: Dict[int, str],
    device: Union[str, torch.device] = "auto",
) -> List[Dict]:
    """
    Batch inference on multiple sequences.

    Args:
        sequences : (N, 30, 162) array or tensor
        model     : loaded SignModel
        label_map : {int: str}
        device    : target device

    Returns:
        List of predict_sequence-style dicts, one per sequence.
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if isinstance(sequences, np.ndarray):
        sequences = torch.from_numpy(sequences.astype(np.float32))

    sequences = sequences.to(device)
    with torch.no_grad():
        logits = model(sequences)
        probs  = F.softmax(logits, dim=-1)

    results = []
    for i in range(probs.size(0)):
        p = probs[i]
        vals, idxs = p.topk(3)
        top_list = [
            (label_map.get(int(ix), f"class_{ix}"), float(v))
            for ix, v in zip(idxs.cpu().numpy(), vals.cpu().numpy())
        ]
        results.append({
            "word":       top_list[0][0],
            "confidence": top_list[0][1],
            "top3":       top_list,
        })
    return results


if __name__ == "__main__":
    # Smoke test
    print("Utils API smoke test...")
    try:
        lm = get_label_map()
        print(f"  Label map loaded: {len(lm)} classes")
        print(f"  Sample: class 0 = '{lm.get(0, 'N/A')}'")
    except FileNotFoundError as e:
        print(f"  (label_map not yet available — run dataset_analysis.py first): {e}")

    # Test predict_sequence with random data (no model loaded)
    dummy = np.zeros((30, 162), dtype=np.float32)
    print(f"  Dummy sequence shape: {dummy.shape}")
    print("  API structure OK → predict_sequence(sequence) returns dict with 'word', 'confidence', 'top3'")
