# CLAUDE.md — Smith-Waterman Exploration

Instrumented plain-Python exploration of Smith-Waterman local sequence alignment.
The goal is to compare algorithmic variants (currently scalar DP and Farrar's striped
method) by recording every major event — cell fills, gap updates, lazy-F passes,
profile loads — so the counts and timings can be inspected, plotted, and validated.

---

## Setup

```bash
make          # builds kseq_wrapper.so (C++17 + zlib required)
```

The `.so` is needed only for FASTA input. All other input sources work without it.

Entry point: `./smith_waterman_exploration.py` (thin shim calling `sw_exploration.cli:main`).

---

## Package layout

```
sw_exploration/
  cli.py               # argparse, build_pairs(), main loop
  types.py             # AlignmentResult, TracebackResult, Recorder
  output.py            # console helpers, build_matrix_figure, build_summary_figure, write_output
  fasta.py             # load_fasta_pairs, normalize_sequence (wraps kseq_reader)
  sw_wrapper.py        # SCORING_REGISTRY, run_scoring() dispatch
  sw_implementations/
    scalar.py          # reference affine-gap DP (smith_waterman_dp, traceback_alignment, run)
    farrar.py          # Farrar's striped method (make_query_profile, run)

kseq_reader.py         # cffi ABI binding for kseq_wrapper.so → KseqReader
kseq_wrapper.cpp       # extern "C" shim over kseqpp SeqStreamIn
kseqpp_lib/            # vendored kseqpp v4.0.0 header-only library (MIT)
```

---

## Core data flow

```
build_pairs(args)                          # cli.py — collects all input sources
    → list of (query_name, query_seq, ref_name, ref_seq)

run_one_pair(query, ref, args, validate)   # runner.py
    → (AlignmentResult, score_mismatch, counts, times, events, h_matrix, cell_events)

write_output(path, pairs_data, args)       # output.py — JSON
build_matrix_figure / build_summary_figure # output.py — matplotlib
```

All input sources (inline, FASTA, random) produce the same flat list. After
`build_pairs()` there is no distinction between "single pair" and "batch".

---

## Recorder — instrumentation

`Recorder` (`types.py`) is passed into every implementation and internal function.

```python
rec.count("farrar.lazy_f_corrections")          # increment a named counter
rec.count("farrar.lane_adds", n)                # increment by n
with rec.timed("farrar.main_striped_pass"):      # wall-clock a stage
    ...
rec.add_cell_event("farrar.lazy_f_trigger", row, col)  # sparse per-cell coord
```

- `rec.counts`      — `Counter[str]`, aggregated in `main()` across all pairs
- `rec.times`       — `Counter[str]`, wall-clock seconds per stage
- `rec.cell_events` — `dict[str, list[(row, col)]]`, H-matrix coordinates
- `rec.events`      — list of strings for `--verbose` output

`cell_events["farrar.lazy_f_trigger"]` stores (query_row, ref_col) in
**H-matrix space** (1-indexed; row 0 and col 0 are the gap-boundary row/column).

---

## Implementations

All implementations share the same call signature:

```python
def run(query, reference, match, mismatch, gap_open, gap_extend, lanes, rec) -> AlignmentResult
```

Register in `sw_wrapper.py`:

```python
SCORING_REGISTRY["myimpl"] = myimpl.run
```

`--implementation` selects from this registry at runtime. The CLI choices list
is derived from `sorted(SCORING_REGISTRY)` automatically.

### scalar (`sw_implementations/scalar.py`)

Standard affine-gap DP filling H, E, F and a pointer matrix.
`smith_waterman_dp()` returns `(AlignmentResult, h_matrix, ptr_matrix)`.
`traceback_alignment()` walks ptr back to the zero boundary.
`run()` is the registry entry point — calls `smith_waterman_dp` and discards the matrices.

### farrar (`sw_implementations/farrar.py`)

Farrar's striped SIMD method simulated in Python with plain lists.

Three recorded stages:
1. **`farrar.profile_build`** — `make_query_profile()` precomputes substitution scores
   per `(reference_symbol, segment)` into a dict of `seg_len × lanes` vectors.
2. **`farrar.main_striped_pass`** — per-reference-column H/E/F update over segments.
3. **`farrar.lazy_f_correction`** — propagates vertical-gap scores across lane
   boundaries until stable. Counts:
   - `farrar.lazy_f_lane_passes` — outer iterations (always at least 1)
   - `farrar.lazy_f_corrections` — iterations where F actually propagated (`stop=False`)
   - `farrar.lazy_f_trigger` in `cell_events` — H-matrix cells where F improved H

`seg_len = ceil(query_len / lanes)`. Query residues are indexed via
`striped_index_to_query_index(segment, lane, seg_len) = lane * seg_len + segment`.

---

## H matrix conventions

- Shape: `(query_len + 1) × (ref_len + 1)`, row 0 and col 0 are all zeros.
- `h_matrix[i][j]` = best local alignment score ending at query position i, reference position j (1-indexed).
- Lazy-F trigger coords `(row, col)` are in this same space.
- **Visualisation transposes** the matrix: `mat.T` so X-axis = query, Y-axis = reference.
  After transpose, trigger coords become `x = row, y = col`.

---

## Output

**JSON** (`results<N>.json`, auto-numbered):
```
metadata    scoring parameters
pairs[]
  query_name, reference_name, query_seq, reference_seq
  query_len, reference_len
  score, end_query, end_reference
  lazy_f_corrections      — farrar.lazy_f_corrections counter for this pair
  lazy_f_triggers         — list of [row, col] from cell_events
  score_mismatch          — 1 if scalar and chosen implementation disagree
  smith_waterman_time_s, farrar_time_s
  h_matrix                — null unless --show-matrix / --validate-scalar / --preview / --heatmap
```

**Matrix figure** (`build_matrix_figure`):
- One subplot per pair; imshow of `mat.T` with colorbar.
- Query on X axis (top), reference on Y axis (left).
- Ticks every `max(1, len // 8)` positions.
- Overlays opt-in via `--heatmap-overlay KEY` (repeatable):
  - `lazy_f` — lime scatter dots at lazy-F trigger cells
  - `match` — green tint where query[i] == ref[j]
  - `mismatch` — orange tint where query[i] != ref[j]
- Adding a new overlay: add a `_overlay_<name>(ax, p, mat)` function to
  `output.py` and register it in `OVERLAY_REGISTRY`.

**Summary figure** (`build_summary_figure`):
- Enabled with `--summary`. Does **not** require H matrices.
- Aggregates and averages metrics across all pairs.
- Left panel: mean ± std of score and lazy-F corrections.
- Right panel: mean ± std of farrar / scalar DP timing (only when non-zero).

---

## need_matrix flag

`runner.py` computes `need_matrix` to decide whether to run scalar DP (slow):

```python
need_matrix = (
    validate_scalar
    or args.show_matrix
    or (args.preview and not summary_only)
    or (args.heatmap and not summary_only)
)
```

`--summary` without `--show-matrix` skips scalar DP entirely.

---

## FASTA reader

`kseq_reader.py` wraps `kseq_wrapper.so` via cffi ABI mode (no Python build step,
just `dlopen`). `KseqReader` is a context-manager iterator yielding `(name, seq)`.
The `.so` is built from `kseq_wrapper.cpp` which wraps kseqpp's `SeqStreamIn`.
kseqpp is vendored in `kseqpp_lib/` (MIT license, v4.0.0, header-only).

---

## Adding a new implementation

1. Create `sw_exploration/sw_implementations/myimpl.py` with:
   ```python
   def run(query, reference, match, mismatch, gap_open, gap_extend, lanes, rec):
       ...
       return AlignmentResult(score, end_query, end_reference)
   ```
2. In `sw_wrapper.py`:
   ```python
   from .sw_implementations import myimpl
   SCORING_REGISTRY["myimpl"] = myimpl.run
   ```
   The CLI `--implementation` choices and help text update automatically.
