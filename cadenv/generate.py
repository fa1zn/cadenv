"""A compositional CAD task generator.

Not a template with random numbers. A part is a *program* — a sequence of
operations sampled from a grammar — so the space of reachable shapes grows with
program length rather than being fixed by the template's parameters.

Three things this has to get right, in order of how badly they bite:

  diversity   Ten thousand samples from a template is one task repeated. Every
              batch is measured for near-duplicates and pairwise shape spread,
              and the numbers ship with the data.

  difficulty  Labelled difficulty is an assertion until something measures it.
              Programs carry a complexity score from their own structure, and
              `calibrate` replaces that with observed solve rates when a
              reference solver is available. The corpus this came from labelled
              everything "easy" and was never checked.

  validity    Random programs produce invalid solids. Failures are counted and
              reported as a yield, not silently dropped — a generator with a 40%
              yield is making different parts than one with 95%.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

from build123d import (Axis, Box, Cylinder, Location, Plane, Pos, Rot, Solid,
                       chamfer, fillet)

# ---------------------------------------------------------------- primitives

BASES = ("box", "cylinder", "prism", "L-section", "T-section")
CUTS = ("through-hole", "blind-hole", "slot", "pocket", "corner-notch", "step")
ADDS = ("boss", "rib", "pad")
MODS = ("fillet-edges", "chamfer-edges")
PATTERNS = ("linear", "circular", "mirror")

# how much each operation contributes to program complexity
WEIGHT = {"through-hole": 1, "blind-hole": 1.5, "slot": 2, "pocket": 1.5,
          "corner-notch": 1.5, "step": 2, "boss": 1.5, "rib": 2.5, "pad": 1.5,
          "fillet-edges": 3, "chamfer-edges": 2.5,
          "linear": 2, "circular": 3, "mirror": 2}


@dataclass
class Op:
    kind: str
    params: dict = field(default_factory=dict)

    def describe(self) -> str:
        p = self.params
        k = self.kind
        if k == "through-hole":
            return (f"a {p['d']:g} mm through-hole at ({p['u']:g}, {p['v']:g}) mm "
                    f"from the part centre")
        if k == "blind-hole":
            return (f"a {p['d']:g} mm hole {p['depth']:g} mm deep at "
                    f"({p['u']:g}, {p['v']:g}) mm from the part centre")
        if k == "slot":
            return (f"a {p['w']:g} mm wide slot {p['length']:g} mm long through the part, "
                    f"centred at ({p['u']:g}, {p['v']:g}) mm and rotated {p['angle']:g}°")
        if k == "pocket":
            return (f"a {p['w']:g} x {p['h']:g} mm pocket {p['depth']:g} mm deep at "
                    f"({p['u']:g}, {p['v']:g}) mm from the part centre")
        if k == "corner-notch":
            return f"a {p['w']:g} x {p['h']:g} mm notch removed from one corner"
        if k == "step":
            return f"a {p['depth']:g} mm step cut along one full edge, {p['w']:g} mm wide"
        if k == "boss":
            return (f"a {p['d']:g} mm diameter boss standing {p['h']:g} mm proud at "
                    f"({p['u']:g}, {p['v']:g}) mm")
        if k == "rib":
            return f"a {p['t']:g} mm thick rib {p['h']:g} mm tall across the part"
        if k == "pad":
            return f"a {p['w']:g} x {p['h']:g} mm pad raised {p['t']:g} mm on the top face"
        if k == "fillet-edges":
            return f"all vertical edges filleted to {p['r']:g} mm radius"
        if k == "chamfer-edges":
            return f"the top face edges chamfered {p['r']:g} mm"
        if k == "linear":
            return f"the previous feature repeated {p['n']}x at {p['pitch']:g} mm pitch"
        if k == "circular":
            return (f"the previous feature repeated {p['n']}x on a "
                    f"{p['pcd']:g} mm bolt circle")
        if k == "mirror":
            return "the previous feature mirrored about the part centreline"
        return k


@dataclass
class Program:
    seed: int
    base: str
    dims: dict
    ops: list

    @property
    def complexity(self) -> float:
        return round(sum(WEIGHT.get(o.kind, 1) for o in self.ops), 1)

    def prompt(self) -> str:
        d = self.dims
        if self.base == "box":
            head = f"A rectangular block {d['x']:g} x {d['y']:g} x {d['z']:g} mm."
        elif self.base == "cylinder":
            head = f"A cylinder {d['d']:g} mm diameter and {d['z']:g} mm tall."
        elif self.base == "prism":
            head = (f"A regular {d['sides']}-sided prism {d['across']:g} mm across flats "
                    f"and {d['z']:g} mm tall.")
        elif self.base == "L-section":
            head = (f"An L-shaped bracket {d['x']:g} x {d['y']:g} mm overall, "
                    f"{d['z']:g} mm thick, with {d['leg']:g} mm legs.")
        else:
            head = (f"A T-shaped part {d['x']:g} x {d['y']:g} mm overall, "
                    f"{d['z']:g} mm thick, with a {d['leg']:g} mm web.")
        body = " ".join(o.describe().capitalize() + "." for o in self.ops)
        return (head + " " + body +
                " Model it in millimetres. Origin and orientation are yours to choose.")


# ------------------------------------------------------------------ sampling

def _sample_base(rng):
    base = rng.choice(BASES)
    z = rng.choice([6, 8, 10, 12, 15, 20, 25, 30])
    if base == "box":
        return base, {"x": rng.choice([40, 50, 60, 80, 100, 120, 150]),
                      "y": rng.choice([30, 40, 50, 60, 80, 100]), "z": z}
    if base == "cylinder":
        return base, {"d": rng.choice([30, 40, 50, 60, 80, 100]), "z": z}
    if base == "prism":
        return base, {"sides": rng.choice([5, 6, 8]),
                      "across": rng.choice([40, 50, 60, 80]), "z": z}
    x = rng.choice([60, 80, 100, 120])
    y = rng.choice([50, 60, 80, 100])
    return base, {"x": x, "y": y, "z": z, "leg": rng.choice([15, 20, 25, 30])}


def _sample_op(rng, dims, extent):
    kind = rng.choices(
        CUTS + ADDS + MODS + PATTERNS,
        weights=[6, 4, 4, 5, 3, 3] + [3, 2, 3] + [3, 3] + [2, 2, 2])[0]
    half = extent / 2
    u = round(rng.uniform(-half * 0.6, half * 0.6), 1)
    v = round(rng.uniform(-half * 0.6, half * 0.6), 1)
    z = dims.get("z", 10)
    if kind == "through-hole":
        return Op(kind, {"d": rng.choice([4, 5, 6, 8, 10, 12, 16]), "u": u, "v": v})
    if kind == "blind-hole":
        return Op(kind, {"d": rng.choice([5, 6, 8, 10, 12]), "u": u, "v": v,
                         "depth": round(rng.uniform(z * 0.3, z * 0.7), 1)})
    if kind == "slot":
        return Op(kind, {"w": rng.choice([5, 6, 8, 10]), "u": u, "v": v,
                         "length": round(rng.uniform(extent * 0.2, extent * 0.5), 1),
                         "angle": rng.choice([0, 30, 45, 60, 90])})
    if kind == "pocket":
        return Op(kind, {"w": round(rng.uniform(extent * .15, extent * .4), 1),
                         "h": round(rng.uniform(extent * .15, extent * .35), 1),
                         "depth": round(rng.uniform(z * .2, z * .6), 1), "u": u, "v": v})
    if kind == "corner-notch":
        return Op(kind, {"w": round(rng.uniform(extent * .1, extent * .25), 1),
                         "h": round(rng.uniform(extent * .1, extent * .25), 1)})
    if kind == "step":
        return Op(kind, {"w": round(rng.uniform(extent * .1, extent * .3), 1),
                         "depth": round(rng.uniform(z * .2, z * .5), 1)})
    if kind == "boss":
        return Op(kind, {"d": rng.choice([10, 12, 16, 20]),
                         "h": round(rng.uniform(z * .3, z * .8), 1), "u": u, "v": v})
    if kind == "rib":
        return Op(kind, {"t": rng.choice([4, 5, 6, 8]),
                         "h": round(rng.uniform(z * .5, z * 1.2), 1)})
    if kind == "pad":
        return Op(kind, {"w": round(rng.uniform(extent * .2, extent * .4), 1),
                         "h": round(rng.uniform(extent * .15, extent * .3), 1),
                         "t": round(rng.uniform(z * .2, z * .5), 1)})
    if kind == "fillet-edges":
        return Op(kind, {"r": rng.choice([2, 3, 4, 5])})
    if kind == "chamfer-edges":
        return Op(kind, {"r": rng.choice([1, 1.5, 2, 3])})
    if kind == "linear":
        return Op(kind, {"n": rng.choice([2, 3, 4]),
                         "pitch": round(rng.uniform(extent * .12, extent * .25), 1)})
    if kind == "circular":
        return Op(kind, {"n": rng.choice([3, 4, 6, 8]),
                         "pcd": round(extent * rng.uniform(.4, .7), 1)})
    return Op("mirror", {})


def sample_program(seed: int, n_ops=None) -> Program:
    rng = random.Random(seed)
    base, dims = _sample_base(rng)
    extent = dims.get("x", dims.get("d", dims.get("across", 60)))
    k = n_ops if n_ops is not None else rng.randint(1, 6)
    ops = []
    for i in range(k):
        op = _sample_op(rng, dims, extent)
        # a pattern needs something to repeat
        if op.kind in PATTERNS and not any(o.kind in CUTS + ADDS for o in ops):
            op = Op("through-hole", {"d": rng.choice([6, 8, 10]),
                                     "u": round(rng.uniform(-extent / 4, extent / 4), 1),
                                     "v": 0.0})
        ops.append(op)
    return Program(seed=seed, base=base, dims=dims, ops=ops)


# ------------------------------------------------------------------ building

def _base_solid(p: Program):
    d = p.dims
    if p.base == "box":
        return Box(d["x"], d["y"], d["z"])
    if p.base == "cylinder":
        return Cylinder(d["d"] / 2, d["z"])
    if p.base == "prism":
        from build123d import RegularPolygon, extrude
        return extrude(RegularPolygon(d["across"] / 2, d["sides"]), amount=d["z"])
    if p.base == "L-section":
        x, y, z, leg = d["x"], d["y"], d["z"], d["leg"]
        return Box(x, y, z) - Pos((x - leg) / 2, (y - leg) / 2, 0) * Box(x, y, z)
    x, y, z, leg = d["x"], d["y"], d["z"], d["leg"]
    return (Box(x, leg, z) + Pos(0, 0, 0) * Box(leg, y, z))


def _apply(solid, op: Op, dims, extent, prev):
    """Apply one operation.

    Returns (solid, feature) where feature is (shape, additive) — a following
    pattern operation repeats it with the same polarity it was first applied
    with. An earlier version always subtracted, which silently turned patterned
    bosses into patterned holes.
    """
    k, p = op.kind, op.params
    z = dims.get("z", 10)

    if k == "through-hole":
        f = Pos(p["u"], p["v"], 0) * Cylinder(p["d"] / 2, z * 3)
        return solid - f, (f, False)
    if k == "blind-hole":
        f = Pos(p["u"], p["v"], z / 2 - p["depth"] / 2 + 0.001) * Cylinder(p["d"] / 2, p["depth"])
        return solid - f, (f, False)
    if k == "slot":
        f = (Pos(p["u"], p["v"], 0) * Rot(0, 0, p["angle"])
             * Box(p["length"], p["w"], z * 3))
        return solid - f, (f, False)
    if k == "pocket":
        f = Pos(p["u"], p["v"], z / 2 - p["depth"] / 2 + 0.001) * Box(p["w"], p["h"], p["depth"])
        return solid - f, (f, False)
    if k == "corner-notch":
        f = Pos(extent / 2 - p["w"] / 2, extent / 2 - p["h"] / 2, 0) * Box(p["w"], p["h"], z * 3)
        return solid - f, (f, False)
    if k == "step":
        f = Pos(extent / 2 - p["w"] / 2, 0, z / 2 - p["depth"] / 2 + 0.001) * \
            Box(p["w"], extent * 3, p["depth"])
        return solid - f, (f, False)
    if k == "boss":
        f = Pos(p["u"], p["v"], z / 2 + p["h"] / 2 - 0.001) * Cylinder(p["d"] / 2, p["h"])
        return solid + f, (f, True)
    if k == "rib":
        f = Pos(0, 0, z / 2 + p["h"] / 2 - 0.001) * Box(extent * 0.9, p["t"], p["h"])
        return solid + f, (f, True)
    if k == "pad":
        f = Pos(0, 0, z / 2 + p["t"] / 2 - 0.001) * Box(p["w"], p["h"], p["t"])
        return solid + f, (f, True)
    if k == "fillet-edges":
        edges = solid.edges().filter_by(Axis.Z)
        if not edges:
            return solid, prev
        return fillet(edges, radius=p["r"]), prev
    if k == "chamfer-edges":
        edges = solid.edges().group_by(Axis.Z)[-1]
        if not edges:
            return solid, prev
        return chamfer(edges, length=p["r"]), prev
    if k in PATTERNS and prev is not None:
        feat, additive = prev
        out = solid
        if k == "linear":
            offsets = [Pos(p["pitch"] * i, 0, 0) for i in range(1, p["n"])]
        elif k == "circular":
            offsets = [Pos(p["pcd"] / 2 * math.cos(2 * math.pi * i / p["n"]),
                           p["pcd"] / 2 * math.sin(2 * math.pi * i / p["n"]), 0)
                       for i in range(1, p["n"])]
        else:
            offsets = [Rot(0, 0, 180)]
        for off in offsets:
            out = out + off * feat if additive else out - off * feat
        return out, prev
    return solid, prev


def build(p: Program):
    """Execute a program. Returns (solid, None) or (None, reason)."""
    try:
        solid = _base_solid(p)
    except Exception as e:
        return None, f"base:{type(e).__name__}"
    extent = p.dims.get("x", p.dims.get("d", p.dims.get("across", 60)))
    prev = None
    for op in p.ops:
        try:
            solid, prev = _apply(solid, op, p.dims, extent, prev)
        except Exception as e:
            return None, f"{op.kind}:{type(e).__name__}"
        if solid is None or solid.volume <= 0:
            return None, f"{op.kind}:empty"
        # Connectivity must be checked after EVERY operation, not once at the
        # end. A cut can split the part in two, and OpenCascade segfaults —
        # not raises — when a subsequent fillet is applied to a disconnected
        # shape. Seed 11562 crashed the 10k build this way.
        try:
            n = len(solid.solids())
        except Exception as e:
            return None, f"{op.kind}:topology:{type(e).__name__}"
        if n != 1:
            return None, f"{op.kind}:split-into-{n}"
    return solid, None


# ----------------------------------------------------------------- diversity

def signature(shape, ndigits=4) -> tuple:
    """A rigid-motion-invariant fingerprint, for near-duplicate detection."""
    from cadverify.invariants import invariants
    inv = invariants(shape)
    scale = inv.volume ** (1 / 3) or 1.0
    return (round(inv.area / scale ** 2, ndigits),
            tuple(round(m / inv.volume / scale ** 2, ndigits) for m in inv.moments),
            inv.n_faces, inv.n_edges)


def embed(shape):
    """Scale-normalised invariant vector, used for pairwise spread."""
    import numpy as np

    from cadverify.invariants import invariants
    inv = invariants(shape)
    s = inv.volume ** (1 / 3) or 1.0
    m = [x / inv.volume / s ** 2 for x in inv.moments]
    return np.array([inv.area / s ** 2, *m, inv.n_faces / 20, inv.n_edges / 40])


def diversity(shapes) -> dict:
    """Measure it, do not assume it. Ten thousand near-identical parts is n=1."""
    import numpy as np

    sigs = [signature(s) for s in shapes]
    uniq = len(set(sigs))
    X = np.array([embed(s) for s in shapes])
    X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=(min(4000, len(X) * 4), 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    d = np.linalg.norm(X[idx[:, 0]] - X[idx[:, 1]], axis=1)
    return {"n": len(shapes),
            "unique_signatures": uniq,
            "duplicate_rate": 1 - uniq / len(shapes),
            "pairwise_median": float(np.median(d)),
            "pairwise_p05": float(np.percentile(d, 5)),
            "near_identical_pairs_pct": float((d < 0.1).mean() * 100)}
