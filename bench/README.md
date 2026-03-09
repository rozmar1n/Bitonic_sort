# Benchmark Workflow

This directory contains scripts for reproducible comparison between:
- GPU bitonic sort (`bitonic_bench`)
- CPU `std::sort`

## Prerequisites

- Built project (`cmake -S . -B build && cmake --build build`)
- OpenCL runtime available on the machine
- Python 3
- `matplotlib` for plotting:
  - `pip install matplotlib`

## Artifacts

By default, benchmark outputs are written to:
- `artifacts/bench/raw_runs.jsonl`
- `artifacts/bench/summary.csv`
- `artifacts/bench/plots/*.png`

Note: `artifacts/bench/` is ignored by git.

## 1) Run benchmark and build summary CSV

```bash
python3 bench/run_bench.py \
  --binary build/src/apps/bitonic_bench/bitonic_bench \
  --min-exp 5 --max-exp 20 \
  --seeds 3 --warmup 3 --iters 10
```

What this does:
- runs `bitonic_bench` for sizes `2^5..2^20`
- records per-run raw metrics to JSONL
- filters warmup runs
- aggregates measured results by size
- writes `summary.csv`

## 2) Build comparison charts

```bash
python3 bench/plot_bench.py \
  --summary-csv artifacts/bench/summary.csv \
  --out-dir artifacts/bench/plots \
  --title-prefix "Bitonic vs std::sort"
```

Generated charts:
- `time_vs_n.png`
- `speedup_vs_n.png`
- `cpu_gpu_breakdown.png`
- `time_cpu_vs_gpu_bar.png` (CPU/GPU time bars for `2^10..2^20`)

## Smoke example (fast local check)

```bash
python3 bench/run_bench.py \
  --binary build/src/apps/bitonic_bench/bitonic_bench \
  --min-exp 5 --max-exp 6 \
  --seeds 1 --warmup 1 --iters 2 \
  --jsonl-out artifacts/bench/raw_runs_smoke.jsonl \
  --summary-csv artifacts/bench/summary_smoke.csv

python3 bench/plot_bench.py \
  --summary-csv artifacts/bench/summary_smoke.csv \
  --out-dir artifacts/bench/plots_smoke \
  --title-prefix "Smoke"
```
