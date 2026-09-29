import time
import numpy as np
from belief_store import BeliefStore, SourceTier, ContentType

rng = np.random.default_rng(0)
D = 512  # dict_size, matches belief_gate.py default


def build_store(n_entries):
    store = BeliefStore()
    for i in range(n_entries):
        key = rng.normal(size=D)
        store.create(key, content_kv=None, weight=rng.uniform(0.3, 1.0),
                     source_tier=rng.choice(list(SourceTier)),
                     content_type=rng.choice(list(ContentType)), t_now=0)
    return store


def bench_old_loop(store, queries, t_now):
    start = time.perf_counter()
    for q in queries:
        store.query(q, t_now=t_now)
    return time.perf_counter() - start


def bench_new_batch(store, queries, t_now):
    start = time.perf_counter()
    store.query_batch(queries, t_now=t_now)
    return time.perf_counter() - start


def run(n_entries, n_tokens):
    store_old = build_store(n_entries)
    store_new = build_store(n_entries)  # separate instance, same seed pattern, fair comparison
    queries = rng.normal(size=(n_tokens, D))

    t_old = bench_old_loop(store_old, queries, t_now=1)
    t_new = bench_new_batch(store_new, queries, t_now=1)

    speedup = t_old / t_new if t_new > 0 else float("inf")
    print(f"entries={n_entries:>6}  tokens={n_tokens:>5}  "
          f"old_loop={t_old:.4f}s  batched={t_new:.4f}s  speedup={speedup:.1f}x")


if __name__ == "__main__":
    print("Benchmarking old per-token loop vs new query_batch()\n")
    run(n_entries=100, n_tokens=256)     # roughly toy-scale (seq=256)
    run(n_entries=500, n_tokens=2048)    # closer to real seq length
    run(n_entries=2000, n_tokens=2048)   # larger belief store, real seq length