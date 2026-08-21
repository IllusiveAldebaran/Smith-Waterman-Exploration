# Smith-Waterman Exploration

The Smith-Waterman family of algorithms has many variants and implementation
strategies. This repo explores how a few of them fare against each other —
in score, timing, and internal event counts — from a plain Python reference
implementation up through a HIP/GPU kernel.

Every implementation records its major algorithmic events (cell fills, gap
updates, lazy-F passes, profile loads) and stage timings through a shared
`Recorder`, so runs can be compared and matrices/heatmaps visualised.

## Requirements

- Python 3.10+
- `cffi` (all implementations except plain `scalar` are cffi-backed)
- `make` + a C++17 compiler + zlib (to build the FASTA reader)
- A C compiler on `PATH` (`c_scalar` compiles `scalar.c` via cffi the first
  time it's instantiated)
- `matplotlib` + `numpy` (only needed for `--preview` / `--heatmap` / `--summary`)
- ROCm + `hipcc` (only needed to build `hip_diagonal`'s `libhipdiagonal.so`;
  everything else works without a GPU)

Build the FASTA reader shared library once before first use:

```bash
make
```

The `.so` this builds is only needed for `--query-file`/`--reference-file`
input; every other input source works without it.

`hip_diagonal` needs its own shared library, built separately:

```bash
cd sw_exploration/sw_implementations/hip_diagonal
make            # builds libhipdiagonal.so (and dp_test, a standalone C++ dev harness)
```

`hip_diagonal.py` loads `libhipdiagonal.so` from its own directory by
default; override the path with the `SWAG_DIAGONAL_HIP_LIB` environment
variable if you build it elsewhere.

## Usage

There is one command. All input sources are additive and produce a flat list of
`(query, reference)` pairs that are aligned in sequence. A single inline
sequence is a list of one pair; a FASTA file is a list of many.

```bash
./smith_waterman_exploration.py [options]
```

### Input sources

Sources may be freely combined. All pairs from all sources are concatenated and
processed together; if nothing is specified, one built-in default pair is used.

```bash
# built-in default (ACACACTA vs AGCACACA)
./smith_waterman_exploration.py

# inline sequences
./smith_waterman_exploration.py --single-query ACACACTA --single-reference AGCACACA

# FASTA files — record i in each file is paired with record i in the other
./smith_waterman_exploration.py \
  --query-file queries.fa \
  --reference-file references.fa

# random pairs
./smith_waterman_exploration.py \
  --random-count 32 \
  --query-length 64 \
  --reference-length 64 \
  --seed 1

# combine sources — inline pair + 32 random pairs
./smith_waterman_exploration.py \
  --single-query ACGT --single-reference TGCA \
  --random-count 32

# reverse every sequence, or swap query/reference for every pair
./smith_waterman_exploration.py --random-count 8 --reverse
./smith_waterman_exploration.py --random-count 8 --transpose
```

Random sequences are generated over the alphabet `ACGT`.

## Implementations

Select with `--implementation` (default: `farrar`).

| Name           | GPU | `--lanes` | Full traceback     | Notes |
|----------------|-----|-----------|---------------------|-------|
| `scalar`       | no  | no        | yes                 | Reference affine-gap DP in pure Python. Fills H, E, F and a pointer matrix; correct, readable, and the ground truth for `--validate-scalar`. |
| `farrar`       | no  | yes       | no                  | Farrar's striped method, simulated in Python with plain lists. Query residues are rearranged into SIMD-style vector segments; three recorded stages (profile build, main striped pass, lazy-F correction). |
| `c_scalar`     | no  | no        | no                  | Same DP as `scalar`, but the fill loop is C, compiled via `cffi` the first time it's instantiated. |
| `hip_diagonal` | yes | yes       | only with `--second-pass` | HIP/GPU kernel; a whole batch of pairs is packed and aligned in one kernel launch. H/E/F live and die on the device — only each pair's best score/location ever comes back to Python. |

Run `--implementation-options NAME` (or `--implementation-options all`) to
print what a specific implementation actually supports — whether it takes
`--lanes`, whether `--best-cell-only`/`--second-pass` do anything for it, and
a short note on its internals — without aligning anything:

```bash
./smith_waterman_exploration.py --implementation-options hip_diagonal
```

```text
hip_diagonal:
  lanes           yes
  best_cell_only  yes
  second_pass     yes
  full_traceback  second_pass only
  gpu             yes
  notes: HIP/GPU diagonal-striped DP. H/E/F live and die on the device --
          best_cell_only is a no-op there (nothing is ever copied back
          regardless). --second-pass gets a real traceback via a scalar CPU
          fallback per pair selected by _needs_backtrace() (currently a stub
          that always returns True); there's no GPU traceback kernel yet.
```

## Options Reference

**Input**

```text
--single-query TEXT             inline query sequence
--single-reference TEXT         inline reference sequence   (alias: --single-target)
--query-file PATH               FASTA file of query sequences
--reference-file PATH           FASTA file of reference sequences   (alias: --target-file)
--random-count N                generate N random query/reference pairs
--query-length INT              length of each random query        default: 32
--reference-length INT          length of each random reference    default: 32
--seed INT                      RNG seed                           default: 0
--reverse                       reverse every sequence (query and reference)
--transpose                     swap query and reference for every pair
```

**Scoring**

```text
--penalties M,X,DO,DE,IO,IE   six comma-separated int8 penalties: MATCH,MISMATCH,
                                DEL_OPEN,DEL_EXT,INS_OPEN,INS_EXT.
                                MATCH/MISMATCH are pre-negated: actual score delta
                                is -MATCH on a match, -MISMATCH on a mismatch.
                                default: -2,1,3,1,3,1  (= +2 reward / -1 penalty)
--implementation NAME         scoring implementation              default: farrar
                                choices: c_scalar, farrar, hip_diagonal, scalar
--implementation-options [NAME]
                               print NAME's supported options and exit without
                                aligning anything; NAME defaults to --implementation,
                                or use "all" to list every implementation
--lanes INT                   SIMD lane count for farrar / hip_diagonal   default: 8
--best-cell-only               track only the best cell's score/location, not the
                                full H/E/F matrices, where an implementation
                                supports it (see --implementation-options); the
                                pure-Python implementations (scalar, farrar) get
                                no memory benefit from this, only c_scalar/hip_diagonal do
--second-pass                  hip_diagonal only: after the normal GPU best-cell
                                pass, compute a real backtrace (currently for every
                                pair — the real selection condition isn't implemented yet)
```

**Output**

```text
--verbose 0|1|2         0: summary only  1: stage events per pair  2: full counters
                          default: 0
--show-matrix           print the H matrix for each pair
--show-triggers          print per-pair counts for every recorded cell-event trigger
                          label (e.g. farrar.lazy_f_trigger)
--heatmap PATH           save an H-matrix figure to file (.png, .pdf, …)
--preview                open an interactive matplotlib window showing H matrices
--annotate-heatmap       overlay each cell's score value on the heatmap
--heatmap-overlay KEY    overlay drawn on the matrix figure; may be repeated
                          choices: lazy_f, match, mismatch
--summary                show a per-pair bar chart (mean score + lazy-F corrections,
                          and timing) instead of H matrices; doesn't need a matrix
--min-score INT          accepted but currently unused by any overlay or metric
--validate-scalar        also run scalar DP and flag score mismatches
--output PATH            accepted but not currently wired up -- no results file
                          is written regardless of this flag (see Output below)
--no-results             accepted, same caveat as --output
--progress               accepted; currently a no-op
```

## Visualisation

`--show-matrix` prints each pair's H matrix as a text grid to the console.

`--preview` / `--heatmap` build one matplotlib figure with one subplot per
pair (an `imshow` of that pair's H matrix, query on the X axis, reference on
the Y axis), unless `--summary` is also given, in which case a single bar
chart of mean±std score/lazy-F-corrections (and timing, if any implementation
recorded time) is built instead of per-pair matrices.

Both need an H matrix to draw from. Not every implementation records one:
`c_scalar`/`hip_diagonal` skip it under `--best-cell-only`, and
`hip_diagonal` never records one at all regardless (see the table above). If
none of the pairs being shown have a matrix, a warning is printed and nothing
is drawn; add `--validate-scalar` to get a matrix via scalar DP as a fallback.

`--heatmap-overlay` adds a marker layer on top of the matrix figure; may be
repeated:

```text
lazy_f      lime dots at H-matrix cells where Farrar's lazy-F correction fired
match       green tint where query[i] == reference[j]
mismatch    orange tint where query[i] != reference[j]
```

```bash
./smith_waterman_exploration.py \
  --single-query CGGACTACGAG --single-reference ACGTACG \
  --heatmap-overlay match --heatmap-overlay lazy_f \
  --preview
```

## Output

Each pair's result is printed to the console as it's aligned:

```text
pair 1/1: query x reference
max_score=10 at (7, 6), dp_fill_time=0.000214s
```

`--verbose 1`/`2` additionally print every recorded `Recorder` event
(counter increments, stage timings) for each pair; `--show-triggers` prints
per-pair counts for every non-`h_matrix` cell-event label recorded (e.g. how
many `farrar.lazy_f_trigger` cells fired).

Internally, every pair's result (score, H matrix when available,
score-mismatch flag against `--validate-scalar`, timing breakdown, and any
`--second-pass` traceback) is collected into a `pairs_data` list — but as of
this writing `--output`/`--no-results` aren't actually wired up to write it
to a file (`output.write_output()` exists and can serialise it to JSON, it's
just not called from `cli.main()` yet).

## Adding a new implementation

Each implementation lives in its own subpackage under
`sw_exploration/sw_implementations/`, not a flat `.py` file, so it can carry
whatever else it needs (C/HIP sources, headers, its own `Makefile`) alongside
its Python entry point.

1. Create `sw_exploration/sw_implementations/myimpl/__init__.py` with a class
   subclassing `Aligner` (`sw_exploration/types.py`):
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
2. Register it in `sw_exploration/sw_wrapper.py`:
   ```python
   from .sw_implementations import myimpl
   SCORING_REGISTRY["myimpl"] = myimpl.MyImpl
   ```
   The CLI `--implementation` choices update automatically. `verbose` and
   `best_cell_only` are passed to every implementation's constructor; add the
   name to `_LANES_IMPLS`/`_SECOND_PASS_IMPLS` if it should also receive
   `lanes`/`second_pass`. Add an entry to `_FULL_TRACEBACK`/`_GPU`/`_NOTES`
   too so `--implementation-options` describes it correctly.

See `CLAUDE.md` for the full internal data-flow and `Recorder` conventions.

## Examples

**Align two sequences, validate against scalar DP, and preview the H matrix:**

```bash
./smith_waterman_exploration.py \
  --single-query CGGACTACGAG \
  --single-reference ACGTACG \
  --validate-scalar \
  --preview
```

The preview window shows the H matrix (query on X, reference on Y).
`score_mismatch` in the (would-be) output data flags disagreement between
Farrar and scalar DP for that pair.

---

**Same pair, with match and lazy-F overlays:**

```bash
./smith_waterman_exploration.py \
  --single-query CGGACTACGAG \
  --single-reference ACGTACG \
  --heatmap-overlay match \
  --heatmap-overlay lazy_f \
  --preview
```

Green tint marks cells where query and reference characters match. Lime dots
mark cells where Farrar's lazy-F correction actually changed H.

---

**Run 10 random pairs with Farrar lane width 8, preview the H matrices:**

```bash
./smith_waterman_exploration.py \
  --random-count 10 --query-length 40 --reference-length 40 \
  --lanes 8 --preview
```

`--lanes 8` sets Farrar's stripe width. With query length 40 that gives
`seg_len = ceil(40 / 8) = 5` segments. The preview shows all 10 H matrices
as subplots.

---

**Run 10 random pairs and show a per-pair score/timing summary instead of H matrices:**

```bash
./smith_waterman_exploration.py \
  --random-count 10 --query-length 40 --reference-length 40 \
  --summary --preview
```

No H matrices are needed for `--summary`, so this is cheap even for large batches.

---

**GPU batch alignment, checking what hip_diagonal supports first:**

```bash
./smith_waterman_exploration.py --implementation-options hip_diagonal

./smith_waterman_exploration.py \
  --implementation hip_diagonal --lanes 128 \
  --random-count 100000 --query-length 96 --reference-length 256 \
  --seed 1 --best-cell-only
```

`--best-cell-only` matters here specifically because `hip_diagonal` is
cffi/GPU-backed — it skips copying anything beyond each pair's best
score/location back from the device.

---

**Align paired FASTA files, then preview with match/mismatch overlays:**

```bash
# First pass — check scores, no matrix needed:
./smith_waterman_exploration.py \
  --query-file sequences/synthetic-query.fa \
  --reference-file sequences/synthetic-reference.fa

# Then preview with overlays:
./smith_waterman_exploration.py \
  --query-file sequences/synthetic-query.fa \
  --reference-file sequences/synthetic-reference.fa \
  --heatmap-overlay match \
  --heatmap-overlay mismatch \
  --preview
```

Green tint = character match cells, orange tint = mismatch cells, overlaid on
the score heatmap.
