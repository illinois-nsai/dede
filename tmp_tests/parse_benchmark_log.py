import argparse
import csv
import re
from typing import Optional, TypedDict

test_re = re.compile(r"Testing (\w+) n=(\d+),? num_cpus=(\d+)")
executed_re = re.compile(r"Executed (\S+) in ([\d.]+)s")
iter_re = re.compile(r"iter(\d+):")

BenchmarkKey = tuple[str, int, int]


class BenchmarkData(TypedDict):
    timings: dict[str, float]
    num_iterations: int


def parse_log(log_path: str) -> dict[BenchmarkKey, BenchmarkData]:
    results: dict[BenchmarkKey, BenchmarkData] = {}
    current_key: Optional[BenchmarkKey] = None
    current_data: Optional[BenchmarkData] = None

    with open(log_path) as f:
        for line in f:
            m = test_re.search(line)
            if m:
                if current_key is not None:
                    assert current_data is not None
                    results[current_key] = current_data
                test_type, n, num_cpus = m.group(1), int(m.group(2)), int(m.group(3))
                current_key = (test_type, n, num_cpus)
                current_data = {"timings": {}, "num_iterations": 0}
                continue

            if current_key is None:
                continue
            assert current_data is not None

            m = executed_re.search(line)
            if m:
                current_data["timings"][m.group(1)] = float(m.group(2))
                continue

            m = iter_re.search(line)
            if m:
                current_data["num_iterations"] = max(
                    current_data["num_iterations"], int(m.group(1))
                )

    if current_key is not None:
        assert current_data is not None
        results[current_key] = current_data

    return results


def write_csv(results: dict[BenchmarkKey, BenchmarkData], out_path: str) -> None:
    timing_keys = sorted({k for data in results.values() for k in data["timings"]})
    fieldnames = ["test_type", "n", "num_cpus", "num_iterations"] + timing_keys

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (test_type, n, num_cpus), data in results.items():
            row: dict[str, object] = {
                "test_type": test_type,
                "n": n,
                "num_cpus": num_cpus,
                "num_iterations": data["num_iterations"],
                **data["timings"],
            }
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse dede benchmark log into CSV.")
    parser.add_argument("log_path", help="Path to the benchmark log file")
    parser.add_argument("--out", default=None, help="Output CSV path (default: <log_path>.csv)")
    args = parser.parse_args()

    out_path = args.out or args.log_path.removesuffix(".txt") + ".csv"
    results = parse_log(args.log_path)
    write_csv(results, out_path)
    print(f"Saved {len(results)} entries to {out_path}")


if __name__ == "__main__":
    main()
