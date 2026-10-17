
#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

METRICS = [
    "simTicks",
    "simSeconds",
    "hostSeconds",
    "simInsts",
    "system.cpu.numCycles",
    "system.cpu.ipc",
    "system.cpu.cpi",
]

NUMBER = re.compile(
    r"^\s*(\S+)\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?:\s|$)"
)


def read_stats(path):
    """Read statistics, retaining the last value for each metric."""
    values = {}

    for line in path.read_text(errors="replace").splitlines():
        match = NUMBER.match(line)

        if match:
            key, value = match.groups()
            try:
                values[key] = float(value)
            except ValueError:
                pass

    return values


def get_metric(stats, key):
    """Return a metric or None if gem5 did not report it."""
    if key in stats:
        return stats[key]

    # Handle common alternative CPU-statistic names.
    alternatives = {
        "system.cpu.numCycles": [
            "system.cpu.numCycles",
            "system.cpu.num_cycles",
        ],
        "system.cpu.ipc": [
            "system.cpu.ipc",
        ],
        "system.cpu.cpi": [
            "system.cpu.cpi",
        ],
    }

    for candidate in alternatives.get(key, []):
        if candidate in stats:
            return stats[candidate]

    return None


def fmt(value):
    if value is None:
        return "N/A"
    return f"{value:.8g}"


def speedup(baseline, comparison):
    """Baseline cycles / comparison cycles."""
    if baseline is None or comparison is None or comparison <= 0:
        return None
    return baseline / comparison


def main():
    parser = argparse.ArgumentParser(
        description="Summarize the four independent gem5 NMS benchmark runs."
    )
    parser.add_argument(
        "--root", type=Path, default=ROOT,
        help="project root (defaults to the directory containing this script's parent)",
    )
    args = parser.parse_args()
    project_root = args.root.resolve()
    run_names = (
        "NMS_scalar_odd_even_gem5",
        "NMS_scalar_bitonic_gem5",
        "NMS_rvv_odd_even_gem5",
        "NMS_rvv_bitonic_gem5",
    )
    rows = []
    results = {}

    for target in run_names:
        name = target.removeprefix("NMS_").removesuffix("_gem5")
        directory = project_root / "build" / "benchmark" / f"{target}-m5out"
        stats_file = directory / "stats.txt"

        if not stats_file.is_file():
            print(f"[MISSING] {name}: {stats_file}")
            continue

        stats = read_stats(stats_file)
        metrics = {
            metric: get_metric(stats, metric)
            for metric in METRICS
        }

        results[name] = metrics

        row = {"run": name, "stats_file": str(stats_file)}
        row.update(metrics)
        rows.append(row)

        # Create a summary beside this run's stats.txt.
        summary_file = directory / "gem5_summary.txt"

        with summary_file.open("w", encoding="utf-8") as out:
            out.write(f"Run: {name}\n")
            out.write(f"Stats file: {stats_file}\n\n")

            for metric, value in metrics.items():
                out.write(f"{metric}: {fmt(value)}\n")

        print(f"[OK] {name} -> {summary_file}")

    if not rows:
        print("\nNo stats.txt files found. Check the directory paths.")
        return

    # Combined metrics CSV.
    summary_csv = project_root / "gem5_summary.csv"
    fields = ["run", "stats_file"] + METRICS

    with summary_csv.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Performance comparisons.
    comparisons = [
        (
            "Bitonic vs scalar odd-even",
            "scalar_odd_even",
            "scalar_bitonic",
        ),
        (
            "Bitonic vs RVV odd-even",
            "rvv_odd_even",
            "rvv_bitonic",
        ),
        (
            "RVV vs scalar odd-even",
            "scalar_odd_even",
            "rvv_odd_even",
        ),
        (
            "RVV vs scalar bitonic",
            "scalar_bitonic",
            "rvv_bitonic",
        ),
    ]

    comparison_csv = project_root / "gem5_comparison.csv"

    with comparison_csv.open(
        "w", newline="", encoding="utf-8"
    ) as out:
        fields = [
            "comparison",
            "baseline_cycles",
            "comparison_cycles",
            "speedup",
        ]
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()

        for label, baseline_name, comparison_name in comparisons:
            baseline = results.get(baseline_name, {}).get(
                "system.cpu.numCycles"
            )
            candidate = results.get(comparison_name, {}).get(
                "system.cpu.numCycles"
            )

            writer.writerow({
                "comparison": label,
                "baseline_cycles": baseline,
                "comparison_cycles": candidate,
                "speedup": speedup(baseline, candidate),
            })

    print(f"\nCombined metrics: {summary_csv}")
    print(f"Speedup comparisons: {comparison_csv}")
    print("\nSpeedup > 1 means the comparison implementation")
    print("used fewer cycles than the baseline.")
    print("Speedup < 1 means the comparison implementation")
    print("used more cycles than the baseline.")


if __name__ == "__main__":
    main()
