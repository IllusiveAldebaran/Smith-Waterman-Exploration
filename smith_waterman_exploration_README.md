# Smith-Waterman / Farrar Exploration

`smith_waterman_exploration.py` is a plain-Python instrumentation script for
Smith-Waterman DP and Farrar's striped method. It ignores AIE/MLIR and focuses
on counts, timings, and heatmap-ready batch experiments.

## Single-Pair Mode

Run the default example:

```bash
./smith_waterman_exploration.py
```

Run explicit sequences:

```bash
./smith_waterman_exploration.py --query ACACACTA --reference AGCACACA
```

Read FASTA/plain-text files:

```bash
./smith_waterman_exploration.py --query-file query.fa --reference-file ref.fa
```

Print the scalar Smith-Waterman H matrix:

```bash
./smith_waterman_exploration.py --matrix --verbose=3
```

## Grid / Heatmap Mode

Run random sequence pairs over a `(query_len, reference_len)` grid:

```bash
./smith_waterman_exploration.py \
  --grid \
  --query-lengths 8:128:8 \
  --reference-lengths 8:128:8 \
  --pairs-per-cell 1000 \
  --grid-csv farrar_grid.csv \
  --heatmap-svg farrar_heatmap.svg
```

Validate Farrar scores against scalar DP for every generated pair:

```bash
./smith_waterman_exploration.py --grid --grid-validate-scalar
```

Show progress per grid cell:

```bash
./smith_waterman_exploration.py --grid --progress
```

## Length Specs

`--query-lengths` and `--reference-lengths` accept comma lists and inclusive
ranges:

```text
8,16,32
8:64:8
10:20
8,16,32:128:32
```

## Main Options

```text
--query TEXT
--reference TEXT / --target TEXT
--query-file PATH
--reference-file PATH / --target-file PATH
--match INT                 default: 2
--mismatch INT              default: -1
--gap-open INT              default: 3
--gap-extend INT            default: 1
--lanes INT                 default: 8
--verbose 1|2|3             default: 1
--matrix
```

Verbosity levels:

```text
--verbose=1   compact default; invocation counts and total times only
--verbose=2   debug metrics; results, stage timing events, and per-cell grid metrics
--verbose=3   full detail; all counters, per-operation events, and matrix output when --matrix is set
```

Grid options:

```text
--grid
--query-lengths SPEC        default: 8:64:8
--reference-lengths SPEC    default: 8:64:8
--pairs-per-cell INT        default: 100
--alphabet TEXT             default: ACGT
--seed INT                  default: 0
--grid-csv PATH             default template: farrar_grid.csv
--heatmap-svg PATH          default template: farrar_heatmap.svg
--heatmap-metric KEY        default: farrar.lazy_f_lane_passes
--grid-validate-scalar
--progress
```

Grid output paths are numbered automatically to avoid overwriting earlier runs.
For example, the default templates generate `farrar_grid1.csv` and
`farrar_heatmap1.svg`, then `farrar_grid2.csv` and `farrar_heatmap2.svg`, and so
on.

Heatmap SVGs place reference length on the horizontal axis and query length on
the vertical axis.

## Useful Heatmap Metrics

Count metrics:

```text
farrar.lazy_f_lane_passes
farrar.lazy_f_segment_iterations
farrar.lazy_f.max_h_f.vector_max_calls
farrar.main_segment_iterations
farrar.profile_vector_loads
farrar.best_score_updates
```

Timing metrics use the `time:` prefix:

```text
time:farrar.profile_build
time:farrar.main_striped_pass
time:farrar.lazy_f_correction
```

## CSV Columns

Grid mode writes one row per `(query_len, reference_len)` cell. Important
columns include:

```text
query_len
reference_len
pairs
avg_score
mismatches
metric
metric_total
metric_avg
profile_build_s_avg
main_striped_pass_s_avg
lazy_f_correction_s_avg
lazy_f_lane_passes_total
lazy_f_segment_iterations_total
lazy_f_max_h_f_calls_total
main_segment_iterations_total
```

## Scaling Note

Total generated pairs are:

```text
len(query_lengths) * len(reference_lengths) * pairs_per_cell
```

For a million pairs, start with a small grid first to confirm the metric, then
increase `--pairs-per-cell`.
