"""runner.py — removed.

Alignment logic has moved into the implementation classes (ScalarImpl,
FarrarImpl, etc.). Each class owns its Recorder, iterates over all pairs in
run(), and stores results in self.results / self.pair_recs / self.h_matrices.

See sw_implementations/scalar.py and sw_implementations/farrar.py.
"""
