"""
Run every test module and print one compact summary.

The individual test files print a [PASS] line per check, which is useful
while developing but runs to 46 lines -- too long for a single screenshot
and too noisy to read at a glance. This runs them all and reports totals.

    python run_all_tests.py          # summary only
    python run_all_tests.py -v       # also echo each [PASS] line

Model-dependent tests are excluded on purpose: smoke_test_nli_backend.py
downloads and loads a transformer, which belongs in its own run.
"""

import subprocess
import sys
import time

MODULES = [
    ("test_belief_store.py", "Belief store: weighting, state, RIF floor, batched query"),
    ("test_reconsolidation.py", "Reconsolidation: contest/reinforce branches, RIF, skips"),
    ("test_nli_backend.py", "NLI backend: label maps, thresholds, abstention, batching"),
    ("test_ept_loop.py", "End-to-end loop: claim extraction, flag accumulation"),
]

WIDTH = 78


def main():
    verbose = "-v" in sys.argv

    print("=" * WIDTH)
    print("EpstLoop test suite".center(WIDTH))
    print("=" * WIDTH)

    results = []
    total_checks = 0
    failed = 0
    t_start = time.perf_counter()

    for filename, description in MODULES:
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, filename], capture_output=True, text=True
        )
        elapsed = time.perf_counter() - t0

        checks = proc.stdout.count("[PASS]")
        ok = proc.returncode == 0
        total_checks += checks
        failed += not ok

        status = f"PASS  {checks:>2} checks" if ok else "FAIL"
        print(f"\n  {filename:<28} {status:<16} {elapsed:>6.2f}s")
        print(f"    {description}")

        if verbose and ok:
            for line in proc.stdout.splitlines():
                if line.startswith("[PASS]"):
                    print(f"      {line}")

        if not ok:
            print("    ---- output ----")
            for line in (proc.stdout + proc.stderr).strip().splitlines()[-15:]:
                print(f"    {line}")

        results.append((filename, checks, ok))

    total_time = time.perf_counter() - t_start

    print()
    print("=" * WIDTH)
    if failed:
        print(f"  {failed} MODULE(S) FAILED  —  {total_checks} checks ran in {total_time:.2f}s")
        print("=" * WIDTH)
        sys.exit(1)

    print(f"  ALL {total_checks} CHECKS PASSED across {len(MODULES)} modules in {total_time:.2f}s")
    print("=" * WIDTH)
    print("\n  Not included here (needs the transformer model):")
    print("    smoke_test_nli_backend.py   — live NLI verdicts + end-to-end run")
    print("    smoke_test_belief_gate.py   — belief gate on MPS")
    print("    smoke_test_fok_gate.py      — FOK gate triggering and gradients")


if __name__ == "__main__":
    main()
