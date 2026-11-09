"""pigeon: Refactor-friendly evaluation and planning skeleton.

This package is intentionally self-contained for evaluation and planning code.

High-level features:
- Parallel evaluation workers using threads.
- CSV-based resumable evaluation logging.
- Pluggable GPU vision service clients (detector/segmenter) via host:port.
- RNN-based low-level executor (parallel to A*) with proximity-first planner gateway.
- Video writer utility designed to avoid memory leaks during long eval runs.
"""

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
