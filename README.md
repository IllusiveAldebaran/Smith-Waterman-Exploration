# Smith-Waterman Exploration
 
The Smith-Waterman family of algorithm has had many slight differences and implementations. For academia and curiousness we explore how some of these implemenations fare in terms of instructions, computing, score, etc.

Plain-Python instrumentation for Smith-Waterman local sequence alignment.
Implements two variants — scalar affine-gap DP and Farrar's striped method —
and records every major algorithmic event (cell fills, gap updates, lazy-F
passes, profile loads, traceback steps) along with wall-clock timings, so you
can answer questions like "how many DP cells does this fill?" or "how much time
goes into Farrar's lazy-F correction?".

## Requirements

- Python 3.10+
- `make` + a C++17 compiler + zlib (to build the FASTA reader)
- `matplotlib` (only needed for `--preview` / `--heatmap`)

Build the FASTA reader shared library once before first use:

```bash
make
```

## Usage

There is one command. All input sources are additive and produce a flat list of
`(query, reference)` pairs that are aligned in sequence. A single inline
sequence is a list of one pair; a FASTA file is a list of many.

```bash
./smith_waterman_exploration.py [options]
```

### Input sources

Sources may be freely combined. All pairs from all sources are concatenated and
processed together.

```bash
# built-in defaults (ACACACTA vs AGCACACA)
./smith_waterman_exploration.py

# inline sequences
./smith_waterman_exploration.py --single-query ACACACTA --single-reference AGCACACA

# FASTA files — record i in each file is paired with record i in the other
./smith_waterman_exploration.py \
  --query-file queries.fa \
  --reference-file references.fa

# random pairs (both counts must be equal)
./smith_waterman_exploration.py \
  --random-count 32 \
  --query-length 64 \
  --reference-length 64

# or separately
./smith_waterman_exploration.py \
  --random-query-count 32 --random-reference-count 32 \
  --query-length 64 --reference-length 64

# combine sources — inline pair + 32 random pairs
./smith_waterman_exploration.py \
  --single-query ACGT --single-reference TGCA \
  --random-count 32
```

### Output

```bash
# print the H matrix for each pair
./smith_waterman_exploration.py --show-matrix

# save a heatmap of lazy-F corrections grouped by (query_len × reference_len)
./smith_waterman_exploration.py --random-count 100 \
  --query-length 32 --reference-length 32 \
  --heatmap results.png

# open an interactive matplotlib window
./smith_waterman_exploration.py --random-count 100 --preview

# cross-check Farrar scores against scalar DP
./smith_waterman_exploration.py --random-count 32 --validate-scalar

# print progress per pair
./smith_waterman_exploration.py --random-count 1000 --progress
```

## Options Reference

**Input**

```text
--single-query TEXT           inline query sequence
--single-reference TEXT       inline reference sequence  (alias: --single-target)
--query-file PATH             FASTA file of query sequences
--reference-file PATH         FASTA file of reference sequences  (alias: --target-file)
--random-count N              generate N random query/reference pairs
--random-query-count N        generate N random queries (must equal --random-reference-count)
--random-reference-count N    generate N random references
--query-length INT            length of each random query      default: 32
--reference-length INT        length of each random reference  default: 32
--alphabet TEXT               random sequence alphabet         default: ACGT
--seed INT                    RNG seed                         default: 0
```

**Scoring**

```text
--match INT           substitution match score        default: 2
--mismatch INT        substitution mismatch score     default: -1
--gap-open INT        affine gap open penalty         default: 3
--gap-extend INT      affine gap extend penalty       default: 1
--implementation      scoring implementation          default: farrar
                        choices: farrar, scalar
--lanes INT           Farrar vector lane count        default: 8
```

**Output**

```text
--verbose 0|1|2       0: summary only  1: stage events per pair  2: full counters
                        default: 0
--show-matrix         print the scalar DP H matrix for each pair
--heatmap PATH        save metric heatmap to file (format from extension: .png, .pdf, …)
--preview             open an interactive matplotlib window showing the heatmap
--heatmap-metric KEY  metric to plot in the heatmap (default: farrar.lazy_f_corrections)
--min-score INT       score threshold; enables score_pass_rate metric
--heatmap-overlay KEY overlay drawn on the matrix figure; may be repeated
                        choices: lazy_f, match, mismatch
--csv PATH            output CSV path template (auto-numbered)  default: results.csv
--validate-scalar     also run scalar DP and flag score mismatches
--summary             show a per-pair bar chart (score + lazy-F) instead of H matrices
--progress            print progress per pair
```

## Heatmap

`--heatmap` and `--preview` build a 2-D figure grouping pairs by
`(query_len × reference_len)` and showing the average of the chosen metric per
cell. When all pairs share the same length the figure is 1×1.

`--heatmap-metric` can be any count or timing key recorded by the Recorder.
Useful choices:

```text
# count metrics
farrar.lazy_f_corrections          how many times lazy-F actually propagated
farrar.lazy_f_lane_passes          outer lazy-F iterations (includes mandatory first pass)
farrar.lazy_f_segment_iterations
farrar.main_segment_iterations
farrar.best_score_updates
score_pass_rate                    fraction of pairs meeting --min-score (requires --min-score)

# timing metrics — prefix with time:
time:farrar.profile_build
time:farrar.main_striped_pass
time:farrar.lazy_f_correction
```

When `--min-score` is set, cells with score ≥ N are highlighted in red in the
matrix figure.

## Output File

Results are written as JSON (default: `results.json`, auto-numbered to avoid
overwriting). Top-level keys:

```text
metadata    scoring parameters and implementation name
pairs       list of per-pair results
```

Each pair entry contains:

```text
query_name, reference_name, query_seq, reference_seq
query_len, reference_len
score, end_query, end_reference
lazy_f_corrections       total lazy-F outer iterations that propagated
lazy_f_triggers          list of [row, col] H-matrix cells where lazy-F changed H
score_mismatch           1 if scalar and chosen implementation disagree
smith_waterman_time_s, farrar_time_s
h_matrix                 full (m+1)×(n+1) H matrix, null if not computed
```

`h_matrix` and `lazy_f_triggers` are populated only when the scalar DP runs
(i.e. when `--show-matrix`, `--validate-scalar`, `--preview`, or `--heatmap`
is set).

## Implementations

| Name     | Description |
|----------|-------------|
| `farrar` | Farrar's striped Smith-Waterman. Query residues are rearranged into SIMD-style vector segments; substitution scores are precomputed into a query profile. Three stages: profile build, main striped pass, lazy-F correction. |
| `scalar` | Reference scalar affine-gap DP. Fills H, E, F matrices cell by cell. Correct and readable; used as the ground-truth score for `--validate-scalar`. |

Adding a new implementation: create `sw_exploration/sw_implementations/myimpl.py`
with a `run(query, reference, match, mismatch, gap_open, gap_extend, lanes, rec)`
function and register it in `sw_exploration/sw_wrapper.py`.

## Examples

**Align two sequences, validate Farrar against scalar DP, and preview the H matrix:**

```bash
./smith_waterman_exploration.py \
  --single-query CGGACTACGAG \
  --single-reference ACGTACG \
  --validate-scalar \
  --preview
```

The preview window shows the 12×8 H matrix (query on X / reference on Y).
Score mismatches between Farrar and scalar DP cause an immediate error.

---

**Same pair, with match and lazy-F overlays:**

```bash
./smith_waterman_exploration.py \
  --single-query CGGACTACGAG \
  --single-reference ACGTACG \
  --validate-scalar \
  --heatmap-overlay match \
  --heatmap-overlay lazy_f \
  --preview
```

Green tint marks cells where query and reference characters match. Lime dots
mark cells where Farrar's lazy-F correction actually changed H.

---

**Run 10 random pairs of length 40 with Farrar lane width 8, preview the H matrices:**

```bash
./smith_waterman_exploration.py \
  --random-count 10 \
  --query-length 40 \
  --reference-length 40 \
  --lanes 8 \
  --preview
```

`--lanes 8` sets Farrar's stripe width. With query length 40 that gives
`seg_len = ceil(40 / 8) = 5` segments. The preview shows all 10 H matrices
as subplots.

---

**Run 10 random pairs and show a per-pair score summary instead of H matrices:**

```bash
./smith_waterman_exploration.py \
  --random-count 10 \
  --query-length 40 \
  --reference-length 40 \
  --summary \
  --preview
```

`--summary` shows a bar chart of scores and lazy-F correction counts per pair.
No scalar DP is run, so this is faster than the full matrix view.

---

**Align paired FASTA files, check the score range, then preview with match/mismatch overlays:**

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
