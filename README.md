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

## Benchmark & plots

Benchmark workflow documentation:
- [bench/README.md](bench/README.md)

Quick start:

```bash
python3 bench/run_bench.py --binary build/src/apps/bitonic_bench/bitonic_bench
python3 bench/plot_bench.py --summary-csv artifacts/bench/summary.csv --out-dir artifacts/bench/plots
```
