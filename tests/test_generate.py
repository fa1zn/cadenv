"""Generator: validity, diversity, reproducibility, and one specific crash."""

import pytest

from cadenv.dataset import build_dataset, rebuild, tier_for, verify_manifest
from cadenv.generate import build, diversity, sample_program, signature


def test_seed_11562_does_not_segfault():
    """A through-hole landing in an L-section's removed corner splits the part
    in two. OpenCascade then *segfaults* — it does not raise — when a fillet is
    applied to the disconnected result, so no try/except catches it. This
    crashed a 10,000-sample build at 9,219. The fix is a connectivity check
    after every operation, not once at the end.
    """
    solid, err = build(sample_program(11562))
    assert solid is None
    assert "split-into-2" in err


def test_connectivity_is_checked_after_every_operation():
    """Guard the fix directly: a program whose first cut splits the part must
    fail on that cut, before any later operation runs."""
    p = sample_program(11562)
    assert p.ops[1].kind == "fillet-edges", "fixture drifted; pick another seed"
    _, err = build(p)
    assert err.startswith("through-hole"), (
        f"failed at {err} — the split should be caught before the fillet"
    )


def test_generation_is_deterministic():
    for seed in (0, 137, 4242):
        a, _ = build(sample_program(seed))
        b, _ = build(sample_program(seed))
        assert signature(a.wrapped) == signature(b.wrapped)


def test_yield_is_reasonable():
    ok = sum(1 for s in range(150) if build(sample_program(s))[0] is not None)
    assert ok / 150 > 0.6, f"yield dropped to {ok/150:.0%}"


def test_batch_is_diverse():
    """Ten thousand samples from a template is n=1. Measure it."""
    shapes = []
    for s in range(300):
        solid, _ = build(sample_program(s))
        if solid is not None:
            shapes.append(solid.wrapped)
    d = diversity(shapes)
    assert d["duplicate_rate"] < 0.02
    assert d["near_identical_pairs_pct"] < 1.0
    assert d["pairwise_median"] > 1.5


def test_tiers_span_the_range():
    tiers = {tier_for(sample_program(s).complexity) for s in range(200)}
    assert tiers == {"easy", "medium", "hard"}


def test_manifest_rebuilds_from_seed(tmp_path):
    out = str(tmp_path / "m.jsonl")
    stats = build_dataset(60, out, max_attempts=400, progress_every=10**9)
    assert stats["kept"] == 60
    v = verify_manifest(out, sample=25)
    assert v["mismatches"] == 0, v["detail"]
