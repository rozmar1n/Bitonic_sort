#!/usr/bin/env python3
import argparse
import pathlib
import random
import sys


def write_values(path: pathlib.Path, values: list[int]) -> None:
    if values:
        path.write_text(" ".join(str(value) for value in values) + "\n", encoding="utf-8")
    else:
        path.write_text("\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic large e2e cases for bitonic_sort_cli."
    )
    parser.add_argument("--output-dir", default="tests/e2e/cases")
    parser.add_argument("--min-exp", type=int, default=5, help="Minimum exponent for 2^k size.")
    parser.add_argument("--max-exp", type=int, default=20, help="Maximum exponent for 2^k size.")
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed for reproducibility.")
    parser.add_argument("--value-min", type=int, default=-1_000_000)
    parser.add_argument("--value-max", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.min_exp < 0 or args.max_exp < 0:
        print("min-exp and max-exp must be non-negative", file=sys.stderr)
        return 1
    if args.min_exp > args.max_exp:
        print("min-exp must be <= max-exp", file=sys.stderr)
        return 1
    if args.value_min > args.value_max:
        print("value-min must be <= value-max", file=sys.stderr)
        return 1
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    for exponent in range(args.min_exp, args.max_exp + 1):
        size = 1 << exponent
        data = [rng.randint(args.value_min, args.value_max) for _ in range(size)]
        expected = sorted(data)

        base_name = f"large_case_pow_{exponent}"
        dat_path = output_dir / f"{base_name}.dat"
        ans_path = output_dir / f"{base_name}.ans"

        write_values(dat_path, data)
        write_values(ans_path, expected)

        print(f"{base_name}: size={size}, files=({dat_path}, {ans_path})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
