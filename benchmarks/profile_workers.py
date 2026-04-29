"""
Line-profile Ray workers and driver without modifying the core API.

Strategy:
  1. Subclass SubproblemsWrap, auto-starting the worker profiler in __init__ so
     the actor initialization cost is captured from the start.
  2. Monkey-patch dede.problem to use the subclass before any actors are created.
  3. Enable the driver-side profiler on all Problem methods.
  4. Single prob.solve() call — no warm-up phase.
  5. Collect and print driver stats + per-actor stats.
"""

import inspect
import io
import sys

import numpy as np
import ray

import dede as dd

sys.setrecursionlimit(10000)

# ---------------------------------------------------------------------------
# Subclass with profiling support — profiler starts in __init__
# ---------------------------------------------------------------------------
import dede.problem as _problem_module  # noqa: E402
import dede.subproblem as _subproblem_module  # noqa: E402
import dede.subproblems_wrap as _subproblems_wrap_module  # noqa: E402
from dede.subproblems_wrap import SubproblemsWrap  # noqa: E402


def _add_all_methods(lp, cls):
    """Add every plain function defined directly on cls to the LineProfiler."""
    for obj in vars(cls).values():
        if inspect.isfunction(obj):
            lp.add_function(obj)


class ProfiledSubproblemsWrap(SubproblemsWrap):
    def __init__(self, *args, **kwargs):
        from line_profiler import LineProfiler

        self._lp = LineProfiler()
        _add_all_methods(self._lp, _subproblem_module.Subproblem)
        _add_all_methods(self._lp, _subproblems_wrap_module.SubproblemsWrap)
        self._lp.enable_by_count()

        super().__init__(*args, **kwargs)

    def get_line_profile_stats(self) -> str:
        self._lp.disable_by_count()
        stream = io.StringIO()
        self._lp.print_stats(stream=stream, output_unit=1e-3, stripzeros=True)
        return stream.getvalue()


# Patch before any dd.Problem.solve() call creates actors
_problem_module.SubproblemsWrap = ProfiledSubproblemsWrap

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _add_all_problem_methods(lp, cls):
    """Add all Problem methods and classmethods, unwrapping @time_all_methods where present."""
    for obj in vars(cls).values():
        if isinstance(obj, classmethod):
            # classmethod descriptors are not callable, so time_all_methods skips them —
            # access the raw function directly via __func__
            lp.add_function(obj.__func__)
        elif callable(obj):
            unwrapped = inspect.unwrap(obj)
            if inspect.isfunction(unwrapped):
                lp.add_function(unwrapped)


def run_with_line_profile(prob: dd.Problem, **solve_kwargs):
    """
    Profile a single prob.solve() call end-to-end (driver + all workers).
    Returns (driver_stats, [per_worker_stats]).
    """
    from line_profiler import LineProfiler

    from dede.problem import Problem

    # Driver-side profiler: all methods on Problem (unwrapped past @time_all_methods)
    driver_lp = LineProfiler()
    _add_all_problem_methods(driver_lp, Problem)
    driver_lp.enable_by_count()

    prob.solve(**solve_kwargs)

    driver_lp.disable_by_count()

    driver_stream = io.StringIO()
    driver_lp.print_stats(stream=driver_stream, output_unit=1e-3, stripzeros=True)

    worker_stats = ray.get([a.get_line_profile_stats.remote() for a in prob._subprob_cache.probs])

    return driver_stream.getvalue(), worker_stats


def print_stats(driver_stats: str, worker_stats: list[str]) -> None:
    print("\n" + "=" * 60)
    print("Driver (dede.Problem)")
    print("=" * 60)
    print(driver_stats)
    for i, stats in enumerate(worker_stats):
        print(f"\n{'=' * 60}")
        print(f"Worker {i} (Subproblem + SubproblemsWrap + cp.Problem.solve)")
        print("=" * 60)
        print(stats)


# ---------------------------------------------------------------------------
# Benchmark problems
# ---------------------------------------------------------------------------


def profile_sum(n, num_cpus, num_iter=20):
    N, M = n, n
    x = dd.Variable((N, M), nonneg=True)
    prob = dd.Problem(
        dd.Maximize(dd.sum(x)),
        [x[i, :].sum() >= i for i in range(N)],
        [x[:, j].sum() <= j for j in range(M)],
    )
    driver_stats, worker_stats = run_with_line_profile(
        prob, ray_address="auto", solver=dd.GUROBI, num_cpus=num_cpus, num_iter=num_iter
    )
    print(f"\n[profile_sum n={n}, num_cpus={num_cpus}, num_iter={num_iter}]")
    print_stats(driver_stats, worker_stats)


def profile_weighted(n, num_cpus, num_iter=20):
    N, M = n, n
    x = dd.Variable((N, M), nonneg=True)
    w = 9 * np.random.uniform(0, 1, (N, M)) + 1
    bn = 9 * np.random.uniform(0, 1, (N,)) + 1
    bm = 9 * np.random.uniform(0, 1, (M,)) + 1
    prob = dd.Problem(
        dd.Minimize(dd.sum(dd.multiply(x, w))),
        [x[i, :].sum() >= bn[i] for i in range(N)],
        [x[:, j].sum() >= bm[j] for j in range(M)],
    )
    driver_stats, worker_stats = run_with_line_profile(
        prob, ray_address="auto", solver=dd.GUROBI, num_cpus=num_cpus, num_iter=num_iter
    )
    print(f"\n[profile_weighted n={n}, num_cpus={num_cpus}, num_iter={num_iter}]")
    print_stats(driver_stats, worker_stats)


if __name__ == "__main__":
    profile_sum(n=400, num_cpus=8, num_iter=20)
