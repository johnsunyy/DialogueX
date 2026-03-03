"""
inference/smoothing.py

Rolling prediction buffer with majority-vote stability filter.
Only emits a letter when:
  - At least `agreement_threshold` of the last `buffer_size` predictions agree.
  - Average confidence of matching predictions >= `confidence_threshold`.
"""

from collections import deque
from typing import Optional, Tuple


class PredictionSmoother:
    """
    Stabilizes webcam predictions by requiring temporal consistency.

    Args:
        buffer_size          : Number of recent predictions to consider.
        agreement_threshold  : Min votes needed from the dominant label.
        confidence_threshold : Min avg confidence of the dominant label's predictions.
    """

    def __init__(
        self,
        buffer_size: int = 10,
        agreement_threshold: int = 7,
        confidence_threshold: float = 0.85,
    ):
        self.buffer_size          = buffer_size
        self.agreement_threshold  = agreement_threshold
        self.confidence_threshold = confidence_threshold
        self._buffer: deque = deque(maxlen=buffer_size)

    def update(self, label: str, confidence: float) -> Optional[str]:
        """
        Add a new (label, confidence) and return the stable label if criteria are met.

        Returns:
            The stable label string if criteria are met, else None.
        """
        self._buffer.append((label, confidence))

        if len(self._buffer) < self.buffer_size:
            return None  # Not enough data yet

        # Count votes and aggregate confidence per label
        vote_counts  = {}
        vote_confs   = {}
        for lbl, conf in self._buffer:
            vote_counts[lbl] = vote_counts.get(lbl, 0) + 1
            if lbl not in vote_confs:
                vote_confs[lbl] = []
            vote_confs[lbl].append(conf)

        # Find dominant label
        dominant = max(vote_counts, key=vote_counts.get)
        count    = vote_counts[dominant]
        avg_conf = sum(vote_confs[dominant]) / len(vote_confs[dominant])

        if count >= self.agreement_threshold and avg_conf >= self.confidence_threshold:
            return dominant

        return None

    def reset(self):
        """Clear the buffer."""
        self._buffer.clear()

    @property
    def current_buffer(self):
        return list(self._buffer)

    @property
    def dominant_label(self) -> Optional[Tuple[str, int, float]]:
        """Returns (dominant_label, vote_count, avg_confidence) or None if buffer empty."""
        if not self._buffer:
            return None
        vote_counts = {}
        vote_confs  = {}
        for lbl, conf in self._buffer:
            vote_counts[lbl] = vote_counts.get(lbl, 0) + 1
            vote_confs.setdefault(lbl, []).append(conf)
        dominant = max(vote_counts, key=vote_counts.get)
        return (dominant, vote_counts[dominant], sum(vote_confs[dominant]) / len(vote_confs[dominant]))
