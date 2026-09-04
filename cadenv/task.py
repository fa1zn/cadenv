"""Tasks: a natural-language spec in, a solid out, with known ground truth.

Two task sources. The procedural generator is the default because it is fully
self-contained — the part is constructed first, so the ground truth is exact by
construction rather than by annotation, and nothing has to be downloaded. The
corpus adapter reads a delivered dataset in the layout cadverify expects.
"""

from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass, field

from build123d import Axis, Box, Cylinder, Location, Mode, Part, Pos


@dataclass
class Task:
    """One environment task."""

    task_id: str
    prompt: str
    reference: object                 # TopoDS_Shape — the ground-truth solid
    units: str = "mm"
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# procedural
# --------------------------------------------------------------------------

def _hole_phrase(h):
    return (f"a {h['d']:g} mm diameter hole through the {h['face']} face, "
            f"centred {h['u']:g} mm and {h['v']:g} mm from the corner")


def generate_task(seed: int) -> Task:
    """A plate with through-holes and an optional pocket.

    The solid is built first and the prompt is written from the same numbers, so
    the spec and the reference cannot drift apart.
    """
    rng = random.Random(seed)
    x = rng.choice([40, 50, 60, 80, 100, 120])
    y = rng.choice([30, 40, 50, 60, 80])
    z = rng.choice([8, 10, 12, 15, 20, 25])

    solid = Box(x, y, z)
    holes = []
    for _ in range(rng.randint(1, 3)):
        d = rng.choice([5, 6, 8, 10, 12])
        u = round(rng.uniform(d, x - d), 1)
        v = round(rng.uniform(d, y - d), 1)
        holes.append({"d": d, "u": u, "v": v, "face": "top"})
        solid -= Pos(u - x / 2, v - y / 2, 0) * Cylinder(d / 2, z * 2)

    pocket = None
    if rng.random() < 0.5:
        pw = round(rng.uniform(x * 0.25, x * 0.5), 1)
        ph = round(rng.uniform(y * 0.25, y * 0.5), 1)
        pd = round(rng.uniform(z * 0.2, z * 0.5), 1)
        pocket = {"w": pw, "h": ph, "depth": pd}
        solid -= Pos(0, 0, z / 2 - pd / 2) * Box(pw, ph, pd)

    parts = [f"A rectangular plate {x} x {y} x {z} mm."]
    parts += [_hole_phrase(h) + "." for h in holes]
    if pocket:
        parts.append(f"A {pocket['w']:g} x {pocket['h']:g} mm rectangular pocket "
                     f"{pocket['depth']:g} mm deep, centred on the top face.")
    parts.append("Model it in millimetres. The origin and orientation are yours to choose.")

    return Task(
        task_id=f"proc-{seed:05d}",
        prompt=" ".join(parts),
        reference=solid.wrapped,
        meta={"source": "procedural", "seed": seed, "x": x, "y": y, "z": z,
              "n_holes": len(holes), "pocket": pocket is not None},
    )


def procedural_taskset(n: int = 50, start: int = 0) -> list[Task]:
    return [generate_task(s) for s in range(start, start + n)]


# --------------------------------------------------------------------------
# delivered corpus
# --------------------------------------------------------------------------

def corpus_taskset(root: str, limit: int | None = None) -> list[Task]:
    """Read a delivered corpus in the layout cadverify's tests expect."""
    from cadverify.invariants import load_step

    tasks = []
    for path in sorted(glob.glob(os.path.join(root, "samples", "*", "*")))[:limit]:
        meta = json.load(open(os.path.join(path, "sample.json")))
        tasks.append(Task(
            task_id=meta["id"],
            prompt=meta.get("prompt", ""),
            reference=load_step(os.path.join(path, "ground_truth", "reference.step")),
            units=meta.get("units", "mm"),
            meta={"source": "corpus", "path": path, "title": meta.get("title"),
                  "answer_key": meta.get("answerKey", {})},
        ))
    return tasks
