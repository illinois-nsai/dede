import argparse
import csv
import re
from typing import Optional, TypedDict

test_re = re.compile(r"Testing (\w+) (.+)")
kwarg_re = re.compile(r"(\w+)=(\S+?)(?:,|$)")
executed_re = re.compile(r"Executed (\S+) in ([\d.]+)s")
iter_re = re.compile(r"iter(\d+):")
dede_solve_time_re = re.compile(r"DeDe Solve Time:\s+([\d.]+)")
dede_iterations_re = re.compile(r"DeDe Iterations:\s+(\d+)")
result_re = re.compile(r"Result\s+(\S+)")

BenchmarkKey = tuple[str, tuple[tuple[str, str], ...]]


class BenchmarkData(TypedDict):
    timings: dict[str, float]
    num_iterations: int
    dede_solve_time: Optional[float]
    dede_iterations: Optional[int]
    result: Optional[str]


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
                test_type = m.group(1)
                kwargs = tuple(sorted(
                    (k, v) for k, v in kwarg_re.findall(m.group(2))
                ))
                current_key = (test_type, kwargs)
                current_data = {"timings": {}, "num_iterations": 0, "dede_solve_time": None, "dede_iterations": None, "result": None}
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
                continue

            m = dede_solve_time_re.search(line)
            if m:
                current_data["dede_solve_time"] = float(m.group(1))
                continue

            m = dede_iterations_re.search(line)
            if m:
                current_data["dede_iterations"] = int(m.group(1))
                continue

            m = result_re.search(line)
            if m:
                current_data["result"] = m.group(1)

    if current_key is not None:
        assert current_data is not None
        results[current_key] = current_data

    return results


def write_csv(results: dict[BenchmarkKey, BenchmarkData], out_path: str) -> None:
    kwarg_keys = sorted({k for _, kwargs in results for k, _ in kwargs})
    timing_keys = sorted({k for data in results.values() for k in data["timings"]})
    fieldnames = ["test_type"] + kwarg_keys + ["num_iterations", "dede_solve_time", "dede_iterations"] + timing_keys + ["result"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (test_type, kwargs), data in results.items():
            row: dict[str, object] = {
                "test_type": test_type,
                **dict(kwargs),
                "num_iterations": data["num_iterations"],
                "dede_solve_time": data["dede_solve_time"],
                "dede_iterations": data["dede_iterations"],
                **data["timings"],
                "result": data["result"],
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
