"""The gate has to catch known-bad rewards and clear the sound one.

A gate that only ever runs against the reward you believe in proves nothing.
Two deliberately broken rewards ship with this package so these tests have
something to fail on.
"""

import pytest

from cadenv.gate import run_gate
from cadenv.policies import ATTACKS, MUST_PASS
from cadenv.reward import REWARDS
from cadenv.task import generate_task, procedural_taskset

FAST = ["volume-only", "axis-aligned-box"]


@pytest.fixture(scope="module")
def tasks():
    return procedural_taskset(8)


@pytest.fixture(scope="module")
def small_tasks():
    return procedural_taskset(4)


def test_generated_task_is_a_valid_single_solid():
    from cadverify.invariants import invariants

    for seed in range(6):
        inv = invariants(generate_task(seed).reference)
        assert inv.n_solids == 1
        assert inv.volume > 0
        assert inv.n_faces >= 6


def test_prompt_and_reference_cannot_drift():
    """The plate dimensions in the prompt must match the solid that was built."""
    from cadverify.invariants import invariants
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    for seed in range(6):
        t = generate_task(seed)
        b = Bnd_Box()
        BRepBndLib.Add_s(t.reference, b, True)
        x0, y0, z0, x1, y1, z1 = b.Get()
        got = sorted([x1 - x0, y1 - y0, z1 - z0])
        want = sorted([t.meta["x"], t.meta["y"], t.meta["z"]])
        for a, c in zip(got, want):
            assert a == pytest.approx(c, abs=1e-6)


@pytest.mark.parametrize("name", FAST)
def test_gate_catches_broken_rewards(name, tasks):
    rep = run_gate(REWARDS[name], tasks)
    assert not rep.passed(), f"{name} is known-exploitable but the gate cleared it"
    assert rep.worst_attack.pass_rate > 0.5


def test_volume_only_falls_to_any_matching_primitive(tasks):
    rep = run_gate(REWARDS["volume-only"], tasks)
    scored = {s.policy: s.pass_rate for s in rep.attacks}
    assert scored["volume-cube"] == 1.0
    assert scored["volume-sphere"] == 1.0, "a box-specific fix would not be a fix"


def test_axis_aligned_box_fails_in_both_directions(tasks):
    """It admits a hollow block and rejects a correctly-posed correct answer."""
    rep = run_gate(REWARDS["axis-aligned-box"], tasks)
    attacks = {s.policy: s.pass_rate for s in rep.attacks}
    legit = {s.policy: s.pass_rate for s in rep.must_pass}
    assert attacks["pocketed-block"] == 1.0
    assert legit["rotated-reference"] == 0.0
    assert legit["exact-reference"] == 1.0, "it does at least accept the unrotated answer"


@pytest.mark.slow
def test_pose_invariant_reward_clears_the_gate(small_tasks):
    rep = run_gate(REWARDS["pose-invariant"], small_tasks)
    assert rep.worst_attack.pass_rate == 0.0, (
        f"{rep.worst_attack.policy} scored against the sound reward"
    )
    assert rep.worst_legitimate.pass_rate == 1.0, (
        f"{rep.worst_legitimate.policy} was rejected by the sound reward"
    )
    assert rep.passed()


def test_every_attack_declares_what_it_knows():
    for p in ATTACKS + MUST_PASS:
        assert p.knows, f"{p.name} does not declare its information access"


def test_reward_components_are_reported_not_just_a_verdict(small_tasks):
    """A reward that emits only pass/fail cannot be tuned or debugged."""
    t = small_tasks[0]
    for name in FAST:
        res = REWARDS[name](t, t.reference)
        assert res.components, f"{name} returned no components"
        assert all(isinstance(v, (int, float, bool)) for v in res.components.values())
