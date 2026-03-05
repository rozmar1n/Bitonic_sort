#!/usr/bin/env python3
import argparse
import pathlib
import subprocess
import sys


def parse_int_tokens(raw: str, label: str) -> list[int]:
    tokens = raw.split()
    values: list[int] = []
    for token in tokens:
        try:
            values.append(int(token))
        except ValueError as exc:
            raise RuntimeError(f"{label}: non-integer token '{token}'") from exc
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        default="build/src/apps/bitonic_sort_cli/bitonic_sort_cli",
    )
    parser.add_argument("--cases-dir", default="tests/e2e/cases")
    args = parser.parse_args()

    binary = pathlib.Path(args.binary)
    cases_dir = pathlib.Path(args.cases_dir)

    case_files = sorted(cases_dir.rglob("*.dat"))
    if not case_files:
        print(f"No .dat files found in {cases_dir}", file=sys.stderr)
        return 1

    failures = 0
    for dat_path in case_files:
        ans_path = dat_path.with_suffix(".ans")
        if not ans_path.exists():
            rel_dat = dat_path.relative_to(cases_dir)
            print(f"{rel_dat}: FAIL (missing {ans_path.name})")
            failures += 1
            continue

        input_text = dat_path.read_text(encoding="utf-8")
        expected_text = ans_path.read_text(encoding="utf-8")

        completed = subprocess.run(
            [str(binary)],
            input=input_text,
            text=True,
            capture_output=True,
        )

        if completed.returncode != 0:
            stderr = completed.stderr.strip() or "<empty stderr>"
            rel_dat = dat_path.relative_to(cases_dir)
            print(f"{rel_dat}: FAIL (exit={completed.returncode}) {stderr}")
            failures += 1
            continue

        try:
            actual = parse_int_tokens(completed.stdout, f"{dat_path.name} stdout")
            expected = parse_int_tokens(expected_text, f"{ans_path.name} expected")
        except RuntimeError as error:
            rel_dat = dat_path.relative_to(cases_dir)
            print(f"{rel_dat}: FAIL ({error})")
            failures += 1
            continue

        if actual != expected:
            rel_dat = dat_path.relative_to(cases_dir)
            print(f"{rel_dat}: FAIL (output mismatch)")
            failures += 1
            continue

        rel_dat = dat_path.relative_to(cases_dir)
        print(f"{rel_dat}: OK")

    return 1 if failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
