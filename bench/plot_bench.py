#!/usr/bin/env python3
import argparse
import csv
import math
import pathlib
import sys


REQUIRED_COLUMNS = {
    "size",
    "cpu_mean_ns",
    "gpu_e2e_mean_ns",
    "gpu_kernel_mean_ns",
    "gpu_h2d_mean_ns",
    "gpu_d2h_mean_ns",
    "speedup_e2e",
    "speedup_kernel",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build benchmark comparison plots from summary.csv."
    )
    parser.add_argument(
        "--summary-csv",
        default="artifacts/bench/summary.csv",
        help="Path to benchmark summary CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/bench/plots",
        help="Directory for output PNG charts.",
    )
    parser.add_argument(
        "--title-prefix",
        default="Bitonic vs std::sort",
        help="Prefix added to chart titles.",
    )
    return parser.parse_args()


def load_summary(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        raise RuntimeError(f"Summary CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS.difference(fieldnames)
        if missing:
            raise RuntimeError(f"Summary CSV missing columns: {sorted(missing)}")

        rows: list[dict] = []
        for row in reader:
            rows.append(
                {
                    "size": int(row["size"]),
                    "cpu_mean_ns": float(row["cpu_mean_ns"]),
                    "gpu_e2e_mean_ns": float(row["gpu_e2e_mean_ns"]),
                    "gpu_kernel_mean_ns": float(row["gpu_kernel_mean_ns"]),
                    "gpu_h2d_mean_ns": float(row["gpu_h2d_mean_ns"]),
                    "gpu_d2h_mean_ns": float(row["gpu_d2h_mean_ns"]),
                    "speedup_e2e": float(row["speedup_e2e"]),
                    "speedup_kernel": float(row["speedup_kernel"]),
                }
            )

    if not rows:
        raise RuntimeError(f"Summary CSV is empty: {path}")

    rows.sort(key=lambda r: r["size"])
    return rows


def ns_to_ms(values_ns: list[float]) -> list[float]:
    return [value / 1_000_000.0 for value in values_ns]


def size_labels(sizes: list[int]) -> list[str]:
    labels: list[str] = []
    for size in sizes:
        if size > 0 and (size & (size - 1)) == 0:
            labels.append(f"2^{int(math.log2(size))}")
        else:
            labels.append(str(size))
    return labels


def make_time_plot(rows: list[dict], out_path: pathlib.Path, title_prefix: str) -> None:
    import matplotlib.pyplot as plt

    sizes = [row["size"] for row in rows]
    cpu_ms = ns_to_ms([row["cpu_mean_ns"] for row in rows])
    gpu_e2e_ms = ns_to_ms([row["gpu_e2e_mean_ns"] for row in rows])
    gpu_kernel_ms = ns_to_ms([row["gpu_kernel_mean_ns"] for row in rows])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sizes, cpu_ms, marker="o", label="CPU std::sort (mean)")
    ax.plot(sizes, gpu_e2e_ms, marker="o", label="GPU end-to-end (mean)")
    ax.plot(sizes, gpu_kernel_ms, marker="o", label="GPU kernel (mean)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Input size (N)")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"{title_prefix}: time_vs_n")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_speedup_plot(
    rows: list[dict], out_path: pathlib.Path, title_prefix: str
) -> None:
    import matplotlib.pyplot as plt

    sizes = [row["size"] for row in rows]
    speedup_e2e = [row["speedup_e2e"] for row in rows]
    speedup_kernel = [row["speedup_kernel"] for row in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sizes, speedup_e2e, marker="o", label="CPU / GPU end-to-end")
    ax.plot(sizes, speedup_kernel, marker="o", label="CPU / GPU kernel")
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.0, label="x1 baseline")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Input size (N)")
    ax.set_ylabel("Speedup")
    ax.set_title(f"{title_prefix}: speedup_vs_n")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_breakdown_plot(
    rows: list[dict], out_path: pathlib.Path, title_prefix: str
) -> None:
    import matplotlib.pyplot as plt

    sizes = [row["size"] for row in rows]
    labels = size_labels(sizes)
    cpu_ms = ns_to_ms([row["cpu_mean_ns"] for row in rows])
    gpu_e2e_ms = ns_to_ms([row["gpu_e2e_mean_ns"] for row in rows])
    gpu_kernel_ms = ns_to_ms([row["gpu_kernel_mean_ns"] for row in rows])
    gpu_copy_ms = ns_to_ms(
        [row["gpu_h2d_mean_ns"] + row["gpu_d2h_mean_ns"] for row in rows]
    )

    x = list(range(len(rows)))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([value - 1.5 * width for value in x], cpu_ms, width, label="CPU std::sort")
    ax.bar([value - 0.5 * width for value in x], gpu_e2e_ms, width, label="GPU end-to-end")
    ax.bar([value + 0.5 * width for value in x], gpu_kernel_ms, width, label="GPU kernel")
    ax.bar([value + 1.5 * width for value in x], gpu_copy_ms, width, label="GPU H2D+D2H")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_xlabel("Input size")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"{title_prefix}: cpu_gpu_breakdown")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    summary_path = pathlib.Path(args.summary_csv)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc

    rows = load_summary(summary_path)

    time_path = out_dir / "time_vs_n.png"
    speedup_path = out_dir / "speedup_vs_n.png"
    breakdown_path = out_dir / "cpu_gpu_breakdown.png"

    make_time_plot(rows, time_path, args.title_prefix)
    make_speedup_plot(rows, speedup_path, args.title_prefix)
    make_breakdown_plot(rows, breakdown_path, args.title_prefix)

    print(f"Summary CSV: {summary_path}")
    print(f"Generated: {time_path}")
    print(f"Generated: {speedup_path}")
    print(f"Generated: {breakdown_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
