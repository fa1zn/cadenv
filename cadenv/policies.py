"""Policies that do not solve the task.

Each one produces a solid without understanding the spec. They exist to be run
against a reward: if any of them scores, the reward is measuring something other
than the task.

Two of them are given the answer key's summary statistics. That is deliberate.
In a real loop the policy never sees the key, but the *reward* is computed from
it, so gradient descent is free to find exactly these solutions. What these
measure is what the reward permits — an upper bound on how badly it can be gamed,
not a prediction about any particular model.
"""

from __future__ import annotations

import math

from build123d import Box, Cylinder, Pos, Sphere
from cadverify.invariants import invariants
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib


def _dims(shape):
    b = Bnd_Box()
    BRepBndLib.Add_s(shape, b, True)
    x0, y0, z0, x1, y1, z1 = b.Get()
    return (x1 - x0, y1 - y0, z1 - z0)


class Policy:
    name = "abstract"
    knows = "nothing"

    def __call__(self, task):
        raise NotImplementedError


class ConstantCube(Policy):
    """Ignores everything. The true null policy — a fixed 50mm cube."""

    name = "constant-cube"
    knows = "nothing"

    def __call__(self, task):
        return Box(50, 50, 50).wrapped


class VolumeCube(Policy):
    """A cube of the reference volume. Defeats any volume-only reward."""

    name = "volume-cube"
    knows = "target volume"

    def __call__(self, task):
        s = invariants(task.reference).volume ** (1 / 3)
        return Box(s, s, s).wrapped


class VolumeSphere(Policy):
    """Same trick, different primitive — a reward that special-cases boxes is
    not fixed, only narrowed."""

    name = "volume-sphere"
    knows = "target volume"

    def __call__(self, task):
        r = (3 * invariants(task.reference).volume / (4 * math.pi)) ** (1 / 3)
        return Sphere(r).wrapped


class PocketedBlock(Policy):
    """A block matching the reference bounding box, pocketed to match its volume.

    Satisfies a volume gate and an axis-aligned bounding-box gate simultaneously
    while being geometrically unrelated to the part. This is the strongest
    exploit the suite carries.
    """

    name = "pocketed-block"
    knows = "target volume + bounding box"

    def __call__(self, task):
        x, y, z = _dims(task.reference)
        target = invariants(task.reference).volume
        outer = x * y * z
        block = Box(x, y, z)
        if target >= outer:
            return block.wrapped
        # remove a centred box scaled by s in every axis so the outer box survives
        s = ((outer - target) / outer) ** (1 / 3)
        return (block - Box(x * s, y * s, z * s)).wrapped


class ScaledReference(Policy):
    """The right shape at the wrong size. Not an exploit — a control that any
    working reward must reject, and a check that the reward is not simply
    accepting everything."""

    name = "scaled-reference-1.1"
    knows = "the answer, scaled wrong"

    def __call__(self, task):
        from OCP.gp import gp_Pnt, gp_Trsf

        from cadverify.invariants import transformed

        t = gp_Trsf()
        t.SetScale(gp_Pnt(0, 0, 0), 1.1)
        return transformed(task.reference, t)


class ExactReference(Policy):
    """The reference itself. A reward that rejects this is broken in the other
    direction, which is the failure the corpus this came from actually had."""

    name = "exact-reference"
    knows = "the answer"

    def __call__(self, task):
        return task.reference


class RotatedReference(Policy):
    """The reference under a rigid rotation — the same part, differently posed.
    Any reward that rejects this is comparing a frame-dependent quantity."""

    name = "rotated-reference"
    knows = "the answer, rotated"

    def __call__(self, task):
        from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf

        from cadverify.invariants import transformed

        t = gp_Trsf()
        t.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0.37, 0.61, 0.70)), 1.05)
        return transformed(task.reference, t)


# Policies that must score 0 against a sound reward.
ATTACKS = [ConstantCube(), VolumeCube(), VolumeSphere(), PocketedBlock(),
           ScaledReference()]

# Policies that must score 1 against a sound reward.
MUST_PASS = [ExactReference(), RotatedReference()]

ALL = ATTACKS + MUST_PASS
