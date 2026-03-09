#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass


REQUIRED_KEYS = {
    "size",
    "warmup",
    "correct",
    "cpu_sort_ns",
    "cpu_bitonic_ns",
    "gpu_end_to_end_ns",
    "gpu_kernel_ns",
    "gpu_h2d_ns",
    "gpu_d2h_ns",
}


@dataclass
class AggregatedMetrics:
    size: int
    samples: int
    cpu_std_mean_ns: float
    cpu_std_median_ns: float
    cpu_bitonic_mean_ns: float
    cpu_bitonic_median_ns: float
    gpu_e2e_mean_ns: float
    gpu_e2e_median_ns: float
    gpu_kernel_mean_ns: float
    gpu_kernel_median_ns: float
    gpu_h2d_mean_ns: float
    gpu_d2h_mean_ns: float
    speedup_std_e2e: float
    speedup_std_kernel: float
    speedup_bitonic_e2e: float
    speedup_bitonic_kernel: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run bitonic_bench and aggregate raw JSONL benchmark runs into summary CSV."
        )
    )
    parser.add_argument(
        "--binary",
        default="build/src/apps/bitonic_bench/bitonic_bench",
        help="Path to bitonic_bench binary.",
    )
    parser.add_argument("--kernel", default="kernels/bitonic_sort.cl")
    parser.add_argument("--min-exp", type=int, default=5)
    parser.add_argument("--max-exp", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--value-min", type=int, default=-1_000_000)
    parser.add_argument("--value-max", type=int, default=1_000_000)
    parser.add_argument("--jsonl-out", default="artifacts/bench/raw_runs.jsonl")
    parser.add_argument("--summary-csv", default="artifacts/bench/summary.csv")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Pass --no-verify to bitonic_bench (default benchmark run verifies results).",
    )
    return parser.parse_args()


def run_benchmark_binary(args: argparse.Namespace) -> None:
    cmd = [
        str(pathlib.Path(args.binary)),
        "--kernel",
        args.kernel,
        "--min-exp",
        str(args.min_exp),
        "--max-exp",
        str(args.max_exp),
        "--seeds",
        str(args.seeds),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--value-min",
        str(args.value_min),
        "--value-max",
        str(args.value_max),
        "--jsonl-out",
        args.jsonl_out,
    ]
    if args.no_verify:
        cmd.append("--no-verify")
    else:
        cmd.append("--verify")

    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, file=sys.stderr, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        raise RuntimeError(f"Benchmark binary failed with exit code {completed.returncode}")

    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")


def load_jsonl_records(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        raise RuntimeError(f"JSONL file not found: {path}")

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at {path}:{line_no}: {exc.msg}"
                ) from exc

            missing = REQUIRED_KEYS.difference(obj.keys())
            if missing:
                raise RuntimeError(
                    f"Record at {path}:{line_no} missing keys: {sorted(missing)}"
                )

            records.append(obj)

    if not records:
        raise RuntimeError(f"No records found in JSONL file: {path}")
    return records


def aggregate_records(records: list[dict]) -> list[AggregatedMetrics]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        if rec["warmup"]:
            continue
        if not rec["correct"]:
            continue
        size = int(rec["size"])
        grouped[size].append(rec)

    if not grouped:
        raise RuntimeError("No valid measured records left after warmup/correct filtering")

    rows: list[AggregatedMetrics] = []
    for size in sorted(grouped):
        bucket = grouped[size]

        cpu_std = [float(r["cpu_sort_ns"]) for r in bucket]
        cpu_bitonic = [float(r["cpu_bitonic_ns"]) for r in bucket]
        gpu_e2e = [float(r["gpu_end_to_end_ns"]) for r in bucket]
        gpu_kernel = [float(r["gpu_kernel_ns"]) for r in bucket]
        gpu_h2d = [float(r["gpu_h2d_ns"]) for r in bucket]
        gpu_d2h = [float(r["gpu_d2h_ns"]) for r in bucket]

        cpu_std_mean = statistics.fmean(cpu_std)
        cpu_bitonic_mean = statistics.fmean(cpu_bitonic)
        gpu_e2e_mean = statistics.fmean(gpu_e2e)
        gpu_kernel_mean = statistics.fmean(gpu_kernel)

        if gpu_e2e_mean <= 0.0 or gpu_kernel_mean <= 0.0:
            raise RuntimeError(f"Non-positive GPU mean time for size={size}")

        rows.append(
            AggregatedMetrics(
                size=size,
                samples=len(bucket),
                cpu_std_mean_ns=cpu_std_mean,
                cpu_std_median_ns=float(statistics.median(cpu_std)),
                cpu_bitonic_mean_ns=cpu_bitonic_mean,
                cpu_bitonic_median_ns=float(statistics.median(cpu_bitonic)),
                gpu_e2e_mean_ns=gpu_e2e_mean,
                gpu_e2e_median_ns=float(statistics.median(gpu_e2e)),
                gpu_kernel_mean_ns=gpu_kernel_mean,
                gpu_kernel_median_ns=float(statistics.median(gpu_kernel)),
                gpu_h2d_mean_ns=statistics.fmean(gpu_h2d),
                gpu_d2h_mean_ns=statistics.fmean(gpu_d2h),
                speedup_std_e2e=cpu_std_mean / gpu_e2e_mean,
                speedup_std_kernel=cpu_std_mean / gpu_kernel_mean,
                speedup_bitonic_e2e=cpu_bitonic_mean / gpu_e2e_mean,
                speedup_bitonic_kernel=cpu_bitonic_mean / gpu_kernel_mean,
            )
        )

    return rows


def write_summary_csv(path: pathlib.Path, rows: list[AggregatedMetrics]) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(
            [
                "size",
                "samples",
                "cpu_mean_ns",
                "cpu_median_ns",
                "cpu_std_mean_ns",
                "cpu_std_median_ns",
                "cpu_bitonic_mean_ns",
                "cpu_bitonic_median_ns",
                "gpu_e2e_mean_ns",
                "gpu_e2e_median_ns",
                "gpu_kernel_mean_ns",
                "gpu_kernel_median_ns",
                "gpu_h2d_mean_ns",
                "gpu_d2h_mean_ns",
                "speedup_e2e",
                "speedup_kernel",
                "speedup_std_e2e",
                "speedup_std_kernel",
                "speedup_bitonic_e2e",
                "speedup_bitonic_kernel",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.size,
                    row.samples,
                    f"{row.cpu_std_mean_ns:.3f}",
                    f"{row.cpu_std_median_ns:.3f}",
                    f"{row.cpu_std_mean_ns:.3f}",
                    f"{row.cpu_std_median_ns:.3f}",
                    f"{row.cpu_bitonic_mean_ns:.3f}",
                    f"{row.cpu_bitonic_median_ns:.3f}",
                    f"{row.gpu_e2e_mean_ns:.3f}",
                    f"{row.gpu_e2e_median_ns:.3f}",
                    f"{row.gpu_kernel_mean_ns:.3f}",
                    f"{row.gpu_kernel_median_ns:.3f}",
                    f"{row.gpu_h2d_mean_ns:.3f}",
                    f"{row.gpu_d2h_mean_ns:.3f}",
                    f"{row.speedup_std_e2e:.6f}",
                    f"{row.speedup_std_kernel:.6f}",
                    f"{row.speedup_std_e2e:.6f}",
                    f"{row.speedup_std_kernel:.6f}",
                    f"{row.speedup_bitonic_e2e:.6f}",
                    f"{row.speedup_bitonic_kernel:.6f}",
                ]
            )


def main() -> int:
    args = parse_args()

    jsonl_path = pathlib.Path(args.jsonl_out)
    summary_path = pathlib.Path(args.summary_csv)

    run_benchmark_binary(args)
    records = load_jsonl_records(jsonl_path)
    rows = aggregate_records(records)
    write_summary_csv(summary_path, rows)

    print(f"Raw runs: {jsonl_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Aggregated sizes: {len(rows)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
