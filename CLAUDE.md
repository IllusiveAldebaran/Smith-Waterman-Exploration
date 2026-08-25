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
  types.py             # AlignmentResult, TracebackResult, Recorder, Aligner
  output.py            # console helpers, build_matrix_figure, build_summary_figure, write_output
  fasta.py             # load_fasta_pairs, normalize_sequence (wraps kseq_reader)
  sw_wrapper.py        # SCORING_REGISTRY, create_impl() dispatch
  sw_implementations/
    scalar/__init__.py       # reference affine-gap DP (smith_waterman_dp,
                              # traceback_alignment, ScalarImpl)
    farrar/__init__.py       # Farrar's striped method (make_query_profile, FarrarImpl)
    c_scalar/                # C-backed scalar DP via cffi
      __init__.py            # CScalarImpl -- compiles scalar.c at instantiation time
      scalar.c, swag.h
    c_farrar/                # C-backed Farrar's striped method via cffi, SSW-kernel-backed
      __init__.py            # CFarrarImpl -- compiles ssw_wrap.c + submodule's ssw.c
      ssw_wrap.c, ssw_wrap.h
      Complete-Striped-Smith-Waterman-Library/  # vendored SSW, git submodule
    hip_diagonal/            # HIP/GPU diagonal-striped DP
      __init__.py            # HIPDiagonalImpl -- loads libhipdiagonal.so via cffi dlopen
      diagonal.c, diagonal.h, diagonal.hip
      test_wrap.cpp, seq_utils.h  # standalone C++ dev harness (dp_test), not used by Python
      Makefile                    # builds libhipdiagonal.so and dp_test

kseq_reader.py         # cffi ABI binding for kseq_wrapper.so → KseqReader
kseq_wrapper.cpp       # extern "C" shim over kseqpp SeqStreamIn
kseqpp_lib/            # vendored kseqpp v4.0.0 header-only library (MIT)
```

Each implementation lives in its own subpackage/subdirectory (`__init__.py` plus
whatever else it needs -- C/HIP sources, headers, its own `Makefile`) rather than
a flat `.py` file, so implementations with more supporting machinery (like
`hip_diagonal`) don't crowd `sw_implementations/`. A module's own directory
(`_here = os.path.dirname(os.path.abspath(__file__))` in `c_scalar`/`hip_diagonal`)
is where it looks up its C sources / compiled `.so`.

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

Every implementation is a class subclassing `Aligner` (`types.py`):

```python
class MyImpl(Aligner):
    def __init__(self, verbose: int = 0, best_cell_only: bool = False, ...) -> None: ...
    def run(self, pen: array.array) -> None:
        # iterate self.pairs, populate self.results / self.pair_recs
        ...
```

Register the class (not a function) in `sw_wrapper.py`:

```python
from .sw_implementations import myimpl
SCORING_REGISTRY["myimpl"] = myimpl.MyImpl
```

`create_impl()` instantiates the class, passing `verbose` and `best_cell_only`
to every implementation, plus `lanes` (implementations listed in `_LANES_IMPLS`)
and `second_pass` (implementations listed in `_SECOND_PASS_IMPLS`). `--implementation`
selects from `SCORING_REGISTRY` at runtime; the CLI choices list is derived from
`sorted(SCORING_REGISTRY)` automatically.

`--implementation-options [NAME|all]` prints what an implementation supports
(`lanes`, `best_cell_only`, `second_pass`, `full_traceback`, `gpu`, plus a
free-text note) and exits without aligning anything -- e.g.
`--implementation-options hip_diagonal` to check whether it supports lanes or
a full traceback. Backed by `sw_wrapper.IMPLEMENTATION_OPTIONS`
(`lanes`/`second_pass`/`best_cell_only` are derived from the membership sets
above so they can't drift; `full_traceback`/`gpu`/notes are hand-maintained
facts in `_FULL_TRACEBACK`/`_GPU`/`_NOTES` -- update those three when adding
a new implementation).

### scalar (`sw_implementations/scalar/__init__.py`)

Standard affine-gap DP filling H, E, F and a pointer matrix.
`smith_waterman_dp()` returns `(AlignmentResult, h_matrix, ptr_matrix)`.
`traceback_alignment()` walks ptr back to the zero boundary.
`ScalarImpl.run()` always calls `smith_waterman_dp`; `best_cell_only` just
skips the subsequent `traceback_alignment` call and reuses the
`AlignmentResult` `smith_waterman_dp()` already returns. It doesn't get its
own reduced-memory DP variant -- this is pure Python, so a hand-rolled
rolling-row rewrite wouldn't be meaningfully faster, and it's not worth the
extra logic to maintain.

### farrar (`sw_implementations/farrar/__init__.py`)

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

`best_cell_only` is a no-op for `FarrarImpl` -- accepted only for uniformity
with `create_impl()`'s kwargs. `"h_matrix"`/`"farrar.lazy_f_trigger"`
`cell_events` are always recorded.

### c_scalar (`sw_implementations/c_scalar/__init__.py`)

Same affine-gap DP as `scalar`, but the fill loop is C (`scalar.c`, compiled
via cffi's `ffibuilder.compile()` at instantiation time into `_swag_ffi`).
`CScalarImpl._align_batch()` packs every pair's sequences into one contiguous
buffer, makes a single `alignBatch()` call, then walks `H_buf` in Python to
populate each pair's `"h_matrix"` `cell_events` -- that walk (not `H_buf`
itself, which is one contiguous C allocation) is what `best_cell_only` skips.

### c_farrar (`sw_implementations/c_farrar/__init__.py`)

C-backed Farrar's striped method, but unlike `farrar` (which *simulates* the
striped method with plain Python lists), `c_farrar` uses the vendored
[Complete Striped Smith-Waterman Library](https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library)
(SSW, a git submodule at `c_farrar/Complete-Striped-Smith-Waterman-Library/`)
as the actual SSE2 kernel, via `ssw_wrap.c`/`ssw_wrap.h` compiled together
with the submodule's `ssw.c` into cffi's `_ssw_ffi` extension.

SSW's `ssw_align()` internally runs up to three passes, gated by a `flag`
argument, and `best_cell_only` maps directly onto that: `best_cell_only=True`
→ `flag=0`, only the forward striped SIMD pass runs (score + end position).
`best_cell_only=False` (the default) → also runs SSW's reverse SIMD pass
(begin position) and a banded-DP pass that produces a real CIGAR, which
`CFarrarImpl._decode_cigar()` walks into a `TracebackResult` per pair. So
unlike every other implementation here, `c_farrar`'s full traceback is the
*default* path, not something extra you opt into.

SSW only has one affine gap cost applied to both insertions and deletions --
no separate `ins_open`/`ins_ext` like our `Penalties` struct has.
`CFarrarImpl._check_gap_penalties()` checks `del_open==ins_open` and
`del_ext==ins_ext` at the top of `run()`, before packing anything or calling
into C, and raises `SystemExit` if they don't match rather than silently
aligning with the wrong cost.

Pairs don't need uniform-length padding the way `c_scalar`/`hip_diagonal`
require -- each pair gets its own independent `ssw_init()` call C-side, so
`_align_batch()` just concatenates raw sequences with an explicit per-pair
length array.

### hip_diagonal (`sw_implementations/hip_diagonal/__init__.py`)

GPU implementation; `_align_batch()` packs all pairs into one `alignBatchNpar()`
HIP kernel call (see `diagonal.c`/`diagonal.hip`). H/E/F live and die on the
device as kernel scratch and are never copied back — only `best_cell`
(score, row, col) per pair survives, so `best_cell_only` is a no-op here
(accepted only so the flag can be passed uniformly across implementations)
and no `"h_matrix"` `cell_events` are ever recorded for this implementation.

`--second-pass` (`HIPDiagonalImpl.second_pass`, `hip_diagonal`-only): after the
normal GPU best-cell pass, `_second_pass_backtrace()` re-runs the scalar CPU
DP (`smith_waterman_dp` + `traceback_alignment`) on every pair selected by
`_needs_backtrace()` to produce a real `TracebackResult`, stored per-pair in
`self.tracebacks` (`None` for unselected/skipped pairs). `_needs_backtrace()`
is currently a stub that returns `True` unconditionally -- the real selection
condition (e.g. a score threshold) is not implemented yet, so every pair
qualifies. There is no dedicated GPU traceback kernel yet; this is a
CPU fallback, exact but redundant (re-fills the whole DP matrix) per selected pair.

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
  backtrace_available      — whether h_matrix (and therefore a traceback) exists for this pair;
                             always false under --best-cell-only unless --validate-scalar supplied one
  traceback                — null unless an implementation's second pass (e.g. hip_diagonal's
                             --second-pass) produced a real TracebackResult for this pair
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

1. Create `sw_exploration/sw_implementations/myimpl/__init__.py` with a class
   subclassing `Aligner` (put any C/HIP sources, headers, or a `Makefile` it
   needs alongside `__init__.py` in the same `myimpl/` directory):
   ```python
   class MyImpl(Aligner):
       def __init__(self, verbose: int = 0, best_cell_only: bool = False) -> None:
           self.verbose = verbose
           self.best_cell_only = best_cell_only
           self.rec = Recorder(verbose=verbose)
           self.results: list[AlignmentResult] = []
           self.pair_recs: list[Recorder] = []

       def run(self, pen: array.array) -> None:
           for _qname, qseq, _rname, rseq in self.pairs:
               pair_rec = Recorder(verbose=self.verbose)
               result = ...  # align qseq against rseq, returning AlignmentResult
               self.results.append(result)
               self.pair_recs.append(pair_rec)
   ```
2. In `sw_wrapper.py`:
   ```python
   from .sw_implementations import myimpl
   SCORING_REGISTRY["myimpl"] = myimpl.MyImpl
   ```
   The CLI `--implementation` choices and help text update automatically.
   `verbose` and `best_cell_only` are passed to every implementation's
   constructor; add the name to `_LANES_IMPLS` if it should also receive
   `lanes`, or to `_SECOND_PASS_IMPLS` if it should receive `second_pass`.
