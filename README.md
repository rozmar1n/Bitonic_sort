# Bitonic sort

OpenCL-based Bitonic Sort implementation with:
- C++ host code (`CL/opencl.hpp`)
- OpenCL kernel execution on GPU
- deterministic e2e validation
- benchmark and plotting pipeline for `std::sort` vs bitonic GPU sort

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Run CLI sorter

`bitonic_sort_cli` reads integers from `stdin` until EOF and prints sorted output.

```bash
echo "9 -1 4 4 0 -3 7 2 5 1" | ./build/src/apps/bitonic_sort_cli/bitonic_sort_cli
```

## E2E tests

```bash
python3 tests/e2e/run_e2e.py --binary build/src/apps/bitonic_sort_cli/bitonic_sort_cli
```

You can limit e2e cases by the maximum power-of-two exponent:

```bash
python3 tests/e2e/run_e2e.py \
  --binary build/src/apps/bitonic_sort_cli/bitonic_sort_cli \
  --cases-dir tests/e2e/cases \
  --max-exp 20
```

## CI (Linux + Windows)

A CI workflow has been added:
- `.github/workflows/ci.yml`

What it does:
- builds on `ubuntu-latest` and `windows-latest`,
- verifies project configurability and compilability (`cmake` configure + build),
- runs only the smoke runtime test (`opencl_smoke_test`) on both platforms,
- does not run e2e or benchmarks in CI.

## Benchmark & plots

Benchmark workflow documentation:
- [bench/README.md](bench/README.md)

Quick start:

```bash
python3 bench/run_bench.py --binary build/src/apps/bitonic_bench/bitonic_bench
python3 bench/plot_bench.py --summary-csv artifacts/bench/summary.csv --out-dir artifacts/bench/plots
```

## Results on My Machine

Benchmark environment:
- OS: `Ubuntu 24.04.4 LTS`
- CPU: `12th Gen Intel(R) Core(TM) i5-1235U`
- OpenCL platform/device: `Intel(R) OpenCL Graphics` / `Intel(R) Iris(R) Xe Graphics`

Benchmark command:

```bash
python3 bench/run_bench.py \
  --binary build/src/apps/bitonic_bench/bitonic_bench \
  --min-exp 5 --max-exp 20 \
  --seeds 3 --warmup 3 --iters 10
```

Summary table (mean timings, GPU end-to-end):

| N (2^k) | Size | CPU mean, ms | GPU e2e mean, ms | Speedup (CPU/GPU e2e) |
| --- | ---: | ---: | ---: | ---: |
| 2^10 | 1,024 | 0.065 | 3.640 | 0.018 |
| 2^12 | 4,096 | 0.480 | 5.059 | 0.095 |
| 2^14 | 16,384 | 2.191 | 5.804 | 0.378 |
| 2^16 | 65,536 | 9.781 | 9.979 | 0.980 |
| 2^18 | 262,144 | 42.985 | 25.659 | 1.675 |
| 2^20 | 1,048,576 | 188.727 | 57.765 | 3.267 |

CPU vs GPU time bar chart for sizes `2^10, 2^12, 2^14, 2^16, 2^18, 2^20`:

![CPU vs GPU time bar chart](bench/my_results/time_cpu_vs_gpu_bar.png)
