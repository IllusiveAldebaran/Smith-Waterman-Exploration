#!/usr/bin/env python3
"""Entry point – delegates to sw_exploration.cli.

The implementation has been split into the sw_exploration package:
  sw_exploration/types.py               – AlignmentResult, TracebackResult, GridCell, Recorder
  sw_exploration/fasta.py               – FASTA parsing, batch pair loading with uniform-length check
  sw_exploration/sw_implementations     – implementations of Smith-Waterman DP + traceback
  sw_exploration/output.py              – CSV writing, SVG heatmap, console helpers
  sw_exploration/runner.py              – run_one_pair (shared inner loop for grid and batch)
  sw_exploration/cli.py                 – argument parsing and single-pair mode
"""

from sw_exploration.cli import main

if __name__ == "__main__":
    main()
