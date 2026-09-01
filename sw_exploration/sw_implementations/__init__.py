"""Concrete Smith-Waterman implementations.

Each implementation lives in its own subdirectory/subpackage here (not a flat
module) so it can carry whatever extra files it needs alongside its Python
entry point -- C/HIP sources, headers, its own Makefile, etc. -- without
crowding this top-level package. To add a new implementation (e.g. Snytsar's
lazy-F optimisation, a new GPU kernel wrapper):

  1. Create sw_implementations/myimpl/__init__.py with a class
     MyImpl(Aligner) implementing:
       def run(self, pen: array.array) -> None
     which iterates self.pairs and populates self.results / self.pair_recs
     (see types.Aligner for the full contract).
  2. Add any supporting files (myimpl.c, a Makefile, etc.) alongside it in
     sw_implementations/myimpl/ -- __init__.py's own directory (`_here` in
     the cffi/cffi-like implementations) is the natural place to look them up.
  3. Register it in sw_wrapper.SCORING_REGISTRY.

Current implementations:
  scalar/       – plain affine-gap DP in pure Python (also provides traceback)
  farrar/       – Farrar's striped SIMD method, simulated in pure Python
  c_scalar/     – C-backed scalar DP via cffi (scalar.c / swag.h)
  hip_diagonal/ – HIP/GPU diagonal-striped DP (diagonal.c / diagonal.h / diagonal.hip)
"""
