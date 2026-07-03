#!/usr/bin/env python3
"""Entry point – delegates to sw_exploration.cli.

The implementation has been split into the sw_exploration package:
  sw_exploration/types.py     – AlignmentResult, TracebackResult, GridCell, Recorder
  sw_exploration/fasta.py     – FASTA parsing, batch pair loading with uniform-length check
  sw_exploration/sw_dp.py     – scalar Smith-Waterman DP + traceback
  sw_exploration/sw_farrar.py – Farrar's striped method + query profile build
  sw_exploration/output.py    – CSV writing, SVG heatmap, console helpers
  sw_exploration/runner.py    – run_one_pair (shared inner loop for grid and batch)
  sw_exploration/grid.py      – grid mode: random pairs over a length sweep
  sw_exploration/batch.py     – batch mode: parallel FASTA files
  sw_exploration/cli.py       – argument parsing and single-pair mode
"""

from sw_exploration.cli import main

if __name__ == "__main__":
    main()
