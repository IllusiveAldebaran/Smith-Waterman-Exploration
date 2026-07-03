"""Concrete Smith-Waterman implementations.

Each module provides one variant of local sequence alignment. To add a new
implementation (e.g. Snytsar's lazy-F optimisation, a GPU kernel wrapper):

  1. Create a new module here, e.g. sw_implementations/snytsar.py
  2. Implement a scoring function with the canonical signature:
       def run(query, reference, match, mismatch, gap_open, gap_extend, lanes, rec)
           -> AlignmentResult
     The `lanes` parameter is available for SIMD-style implementations; purely
     scalar ones may ignore it.
  3. Register it in sw_wrapper.SCORING_REGISTRY.

Current implementations:
  scalar  – plain affine-gap DP (also provides traceback)
  farrar  – Farrar's striped SIMD method
"""
