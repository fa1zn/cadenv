"""Rewards for the CAD environment, including the ones that don't work.

Three implementations ship deliberately. Two of them are exploitable, and they
are here so the admission gate has something to catch — a gate that has never
caught anything is not evidence of a safe reward.

Every reward returns components as well as a scalar. A reward that emits only a
number cannot be tuned, audited, or debugged when it starts paying out for the
wrong thing.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from cadverify.align import align
from cadverify.invariants import invariants
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

TOLERANCE_PCT = 3.0
SHAPE_THRESHOLD_PCT = 0.5   # % of bbox diagonal


def _dims(shape):
    b = Bnd_Box()
    BRepBndLib.Add_s(shape, b, True)
    x0, y0, z0, x1, y1, z1 = b.Get()
    return (x1 - x0, y1 - y0, z1 - z0)


def _pct(a, b):
    return abs(a - b) / abs(b) * 100 if b else float("inf")


@dataclass
class RewardResult:
    reward: float                       # 1.0 pass, 0.0 fail
    passed: bool
    components: dict = field(default_factory=dict)
    failed_gate: str | None = None


class Reward:
    name = "abstract"
    exploitable = None                  # documented expectation, asserted by the gate

    def __call__(self, task, submission) -> RewardResult:
        raise NotImplementedError


class VolumeOnlyReward(Reward):
    """Volume within tolerance. Pose-invariant, and one scalar — so infinitely
    many shapes satisfy it. Kept as the simplest exploitable baseline."""

    name = "volume-only"
    exploitable = True

    def __call__(self, task, submission):
        ref, sub = invariants(task.reference), invariants(submission)
        err = _pct(sub.volume, ref.volume)
        ok = err <= TOLERANCE_PCT
        return RewardResult(float(ok), ok, {"volume_err_pct": err},
                            None if ok else "volume")


class AxisAlignedBoxReward(Reward):
    """Volume plus bounding box compared as an ordered (x, y, z) triple.

    This is the shape of reward that shipped with the corpus this environment
    was built from. An axis-aligned box is not rotation-invariant, so it rejects
    correct parts in the wrong pose — on that corpus it failed the *ground truth*
    on 114 of 150 samples — while still admitting any solid whose box happens to
    match. Kept so the gate can demonstrate both failure modes at once.
    """

    name = "axis-aligned-box"
    exploitable = True

    def __call__(self, task, submission):
        ref, sub = invariants(task.reference), invariants(submission)
        rb, sb = _dims(task.reference), _dims(submission)
        vol = _pct(sub.volume, ref.volume)
        box = max(_pct(sb[i], rb[i]) for i in range(3))
        ok = vol <= TOLERANCE_PCT and box <= TOLERANCE_PCT
        return RewardResult(
            float(ok), ok,
            {"volume_err_pct": vol, "bbox_err_pct": box,
             "bbox_err_pct_best_permutation":
                 min(max(_pct(sb[p[i]], rb[i]) for i in range(3))
                     for p in itertools.permutations(range(3)))},
            None if ok else ("volume" if vol > TOLERANCE_PCT else "bbox"))


class PoseInvariantReward(Reward):
    """Invariants for admission, aligned shape distance for rejection.

    Rule one: never compare a frame-dependent quantity without aligning first.
    Rule two: invariants alone are necessary and not sufficient, because a single
    scalar is shared by many shapes. Both gates must pass.
    """

    name = "pose-invariant"
    exploitable = False

    def __call__(self, task, submission):
        ref, sub = invariants(task.reference), invariants(submission)
        vol = _pct(sub.volume, ref.volume)
        area = _pct(sub.area, ref.area)
        res = align(task.reference, submission)
        shape = res["chamfer_pct_diag"]

        comps = {"volume_err_pct": vol, "area_err_pct": area,
                 "shape_dist_pct_diag": shape, "degenerate": res["degenerate"]}
        if vol > TOLERANCE_PCT:
            return RewardResult(0.0, False, comps, "volume")
        if area > TOLERANCE_PCT:
            return RewardResult(0.0, False, comps, "area")
        if shape > SHAPE_THRESHOLD_PCT:
            return RewardResult(0.0, False, comps, "shape")
        return RewardResult(1.0, True, comps, None)


REWARDS = {r.name: r for r in (VolumeOnlyReward(), AxisAlignedBoxReward(),
                               PoseInvariantReward())}
