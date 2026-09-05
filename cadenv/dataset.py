"""Build a task set at scale.

The artifact is a manifest, not a pile of solids. Every row carries the seed that
produced it, so the geometry is reproducible byte-for-byte from ~2 MB of JSONL
rather than shipped as ~500 MB of STEP. Regenerating is also the strongest
integrity check there is: if a row's recorded invariants do not match what its
seed rebuilds, something is wrong.

Deduplication is by rigid-motion-invariant signature, so two programs that
happen to describe the same solid collapse to one task.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter

from .generate import build, diversity, sample_program, signature

TIERS = [("easy", 0, 4), ("medium", 4, 8), ("hard", 8, 99)]


def tier_for(complexity: float) -> str:
    for name, lo, hi in TIERS:
        if lo <= complexity < hi:
            return name
    return TIERS[-1][0]


def build_dataset(n_target: int, out: str, start_seed: int = 0,
                  max_attempts: int | None = None, progress_every: int = 1000):
    """Generate n_target unique valid tasks, writing a manifest as we go."""
    from cadverify.invariants import invariants

    max_attempts = max_attempts or n_target * 4
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    seen = set()
    kept = 0
    fails = Counter()
    dupes = 0
    tiers = Counter()
    t0 = time.time()

    with open(out, "w") as fh:
        for seed in range(start_seed, start_seed + max_attempts):
            if kept >= n_target:
                break
            prog = sample_program(seed)
            solid, err = build(prog)
            if solid is None:
                fails[err.split(":")[0]] += 1
                continue
            try:
                sig = signature(solid.wrapped)
                inv = invariants(solid.wrapped)
            except Exception as e:
                fails[f"measure:{type(e).__name__}"] += 1
                continue
            if sig in seen:
                dupes += 1
                continue
            seen.add(sig)

            tier = tier_for(prog.complexity)
            tiers[tier] += 1
            fh.write(json.dumps({
                "task_id": f"cadenv-{seed:07d}",
                "seed": seed,
                "prompt": prog.prompt(),
                "base": prog.base,
                "ops": [o.kind for o in prog.ops],
                "n_ops": len(prog.ops),
                "complexity": prog.complexity,
                "tier": tier,
                "volume_mm3": round(inv.volume, 4),
                "area_mm2": round(inv.area, 4),
                "n_faces": inv.n_faces,
                "n_edges": inv.n_edges,
            }) + "\n")
            kept += 1
            if kept % progress_every == 0:
                print(f"  {kept}/{n_target}  seed={seed}  "
                      f"{time.time() - t0:.0f}s", flush=True)

    attempts = seed - start_seed + 1
    return {"kept": kept, "attempts": attempts, "yield": kept / max(attempts, 1),
            "duplicates_rejected": dupes, "build_failures": dict(fails.most_common(8)),
            "tiers": dict(tiers), "seconds": round(time.time() - t0, 1)}


def load_manifest(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def rebuild(row):
    """Reconstruct a task's solid from its seed alone."""
    solid, err = build(sample_program(row["seed"]))
    if solid is None:
        raise ValueError(f"{row['task_id']} no longer builds: {err}")
    return solid.wrapped


def verify_manifest(path, sample=200, seed=0):
    """Rebuild a random sample and confirm the recorded invariants still hold.

    This is the integrity check that makes shipping seeds instead of geometry
    safe. If it fails, the generator changed and the manifest is stale.
    """
    import random

    from cadverify.invariants import invariants

    rows = load_manifest(path)
    rng = random.Random(seed)
    picked = rng.sample(rows, min(sample, len(rows)))
    bad = []
    for r in picked:
        try:
            inv = invariants(rebuild(r))
        except Exception as e:
            bad.append((r["task_id"], f"rebuild failed: {e}"))
            continue
        if abs(inv.volume - r["volume_mm3"]) > 1e-3:
            bad.append((r["task_id"], f"volume {inv.volume} vs {r['volume_mm3']}"))
        elif inv.n_faces != r["n_faces"]:
            bad.append((r["task_id"], f"faces {inv.n_faces} vs {r['n_faces']}"))
    return {"checked": len(picked), "mismatches": len(bad), "detail": bad[:10]}
