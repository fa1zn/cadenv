"""Miles connector: cadenv as a custom reward function.

Miles' plug-point for a reward is one function:

    async def custom_rm(args, sample: Sample) -> float

wired with ``--custom-rm-path cadenv.miles_adapter.cad_reward``. Everything else
about the training loop is Miles' problem.

Why cadenv fits this interface well: tasks are seed-addressable. The reward needs
a single integer to reconstruct exact ground truth, so nothing has to be mounted,
downloaded, or kept in sync between the trainer and the grader. ``sample.label``
carries the seed; the reference solid is rebuilt on the spot.

Generated code runs in a subprocess. A model writing CAD code will eventually
write code that hangs, allocates without bound, or segfaults OpenCascade — that
last one is not hypothetical, it killed a 10,000-sample generation run here — and
none of those are catchable in-process.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile

from .generate import build, sample_program
from .reward import REWARDS

CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)
EXEC_TIMEOUT_S = 120

_RUNNER = "\n".join([
    "import sys",
    "from build123d import *",
    "ns = {}",
    "exec(open(sys.argv[1]).read(), ns)",
    "r = ns.get('result')",
    "if r is None: raise SystemExit('no result variable')",
    "shape = r.wrapped if hasattr(r, 'wrapped') else r",
    "from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs",
    "w = STEPControl_Writer(); w.Transfer(shape, STEPControl_AsIs); w.Write(sys.argv[2])",
])


def extract_code(response: str) -> str:
    m = CODE_RE.search(response or "")
    return m.group(1) if m else (response or "")


def seed_of(sample) -> int | None:
    """Miles carries ground truth in `label`; `metadata` is the fallback."""
    for value in (getattr(sample, "label", None),
                  (getattr(sample, "metadata", None) or {}).get("seed"),
                  (getattr(sample, "metadata", None) or {}).get("task_id")):
        if value is None:
            continue
        if isinstance(value, int):
            return value
        s = str(value)
        if s.isdigit():
            return int(s)
        m = re.search(r"(\d+)", s)
        if m:
            return int(m.group(1))
    return None


def run_to_shape(code: str, python: str | None = None):
    """Execute generated code out-of-process and return a solid, or None."""
    python = python or sys.executable
    src = tempfile.mktemp(suffix=".py")
    runner = tempfile.mktemp(suffix=".py")
    step = tempfile.mktemp(suffix=".step")
    open(src, "w").write(code)
    open(runner, "w").write(_RUNNER)
    try:
        proc = subprocess.run([python, runner, src, step],
                              capture_output=True, text=True, timeout=EXEC_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception as e:
        return None, f"spawn:{type(e).__name__}"
    if proc.returncode != 0 or not os.path.exists(step):
        # returncode < 0 is a signal — OpenCascade segfaults on some inputs
        tag = "segfault" if proc.returncode and proc.returncode < 0 else "exec"
        return None, f"{tag}:{(proc.stderr or '')[-160:]}"
    from cadverify.invariants import load_step
    try:
        return load_step(step), None
    except Exception as e:
        return None, f"load:{type(e).__name__}"
    finally:
        for p in (src, runner):
            try:
                os.unlink(p)
            except OSError:
                pass


def score_sync(sample, reward_name: str = "pose-invariant") -> float:
    """The whole grading path, synchronous. Returns 1.0 or 0.0."""
    seed = seed_of(sample)
    if seed is None:
        return 0.0
    ref, err = build(sample_program(seed))
    if ref is None:
        return 0.0                      # a task that no longer builds scores nothing
    shape, err = run_to_shape(extract_code(getattr(sample, "response", "")))
    if shape is None:
        _record(sample, {"failed_gate": "execution", "exec_error": err})
        return 0.0

    from .task import Task
    task = Task(task_id=f"cadenv-{seed:07d}", prompt="", reference=ref.wrapped)
    res = REWARDS[reward_name](task, shape)
    _record(sample, {"failed_gate": res.failed_gate, **res.components})
    return float(res.reward)


def _record(sample, detail: dict):
    """Miles logs `metadata`; put the components there so a zero is debuggable."""
    md = getattr(sample, "metadata", None)
    if isinstance(md, dict):
        md["cadenv"] = detail


async def cad_reward(args, sample) -> float:
    """Miles plug-point.  --custom-rm-path cadenv.miles_adapter.cad_reward"""
    return await asyncio.to_thread(score_sync, sample)


async def batched_cad_reward(args, samples) -> list[float]:
    """Batched plug-point.  add --group-rm"""
    return list(await asyncio.gather(
        *(asyncio.to_thread(score_sync, s) for s in samples)))


# --------------------------------------------------------------- data export

PROMPT_SUFFIX = ("\n\nReply with ONLY one ```python code block defining a module-level "
                 "variable `result` holding the final build123d Part or Solid.")


def export_for_miles(manifest_path: str, out_path: str, tier: str | None = None):
    """Write a Miles-shaped JSONL: prompt + label, where label is the seed."""
    from .dataset import load_manifest

    rows = load_manifest(manifest_path)
    if tier:
        rows = [r for r in rows if r["tier"] == tier]
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps({"prompt": r["prompt"] + PROMPT_SUFFIX,
                                 "label": str(r["seed"]),
                                 "metadata": {"tier": r["tier"],
                                              "complexity": r["complexity"]}}) + "\n")
    return len(rows)
