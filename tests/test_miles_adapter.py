"""The Miles connector contract.

Miles' reward plug-point is `async def custom_rm(args, sample) -> float`, wired
via `--custom-rm-path`. These tests hold the adapter to that signature and to
the behaviours a training loop depends on: a wrong answer scores 0 rather than
crashing the run, and a zero is debuggable afterwards.
"""

import asyncio
import inspect
from pathlib import Path
from dataclasses import dataclass, field

import pytest

from cadenv.miles_adapter import (batched_cad_reward, cad_reward,
                                  export_for_miles, extract_code, seed_of,
                                  score_sync)


@dataclass
class Sample:
    """Mirrors the fields of miles.utils.types.Sample that the adapter reads."""
    prompt: str = ""
    response: str = ""
    label: str | None = None
    reward: float | None = None
    metadata: dict = field(default_factory=dict)


SEED = 3
# The repo root is interpolated here, at collection time. The string below is
# generated code executed in a subprocess, so a Path(__file__) inside it would
# resolve to the temp file, not to this repo.
_REPO = str(Path(__file__).resolve().parents[1])
CORRECT = f"""```python
import sys
sys.path.insert(0, {_REPO!r})
from cadenv.generate import sample_program, build
result, _ = build(sample_program({SEED}))
```"""
WRONG = "```python\nfrom build123d import *\nresult = Box(50, 50, 50)\n```"


def test_signature_matches_the_miles_plug_point():
    sig = inspect.signature(cad_reward)
    assert list(sig.parameters) == ["args", "sample"]
    assert inspect.iscoroutinefunction(cad_reward)
    assert inspect.iscoroutinefunction(batched_cad_reward)


def test_extract_code_handles_fenced_and_bare():
    assert extract_code("```python\nx=1\n```").strip() == "x=1"
    assert extract_code("```\nx=2\n```").strip() == "x=2"
    assert extract_code("x=3").strip() == "x=3"


@pytest.mark.parametrize("label,expected", [
    ("42", 42), (42, 42), ("cadenv-0000042", 42), (None, None),
])
def test_seed_extraction(label, expected):
    assert seed_of(Sample(label=label)) == expected


def test_metadata_seed_is_a_fallback():
    assert seed_of(Sample(label=None, metadata={"seed": 7})) == 7


@pytest.mark.slow
def test_correct_answer_scores_one():
    assert score_sync(Sample(response=CORRECT, label=str(SEED))) == 1.0


@pytest.mark.slow
def test_wrong_shape_scores_zero_and_says_why():
    s = Sample(response=WRONG, label=str(SEED))
    assert score_sync(s) == 0.0
    assert s.metadata["cadenv"]["failed_gate"] is not None


@pytest.mark.parametrize("response", [
    "```python\nresult = undefined_name + 1\n```",   # raises
    "```python\nx = 1\n```",                          # no `result`
    "I would model this as a plate.",                 # no code at all
    "",                                               # empty
])
@pytest.mark.slow
def test_bad_responses_score_zero_without_raising(response):
    """A training loop must not die because a model wrote nonsense."""
    s = Sample(response=response, label=str(SEED))
    assert score_sync(s) == 0.0
    assert s.metadata["cadenv"]["failed_gate"] == "execution"


def test_missing_label_scores_zero():
    assert score_sync(Sample(response=WRONG, label=None)) == 0.0


@pytest.mark.slow
def test_batched_matches_individual():
    batch = [Sample(response=CORRECT, label=str(SEED)),
             Sample(response=WRONG, label=str(SEED))]
    assert asyncio.run(batched_cad_reward(None, batch)) == [1.0, 0.0]


def test_export_shape(tmp_path):
    from cadenv.dataset import build_dataset
    man = str(tmp_path / "m.jsonl")
    build_dataset(40, man, max_attempts=300, progress_every=10**9)
    out = str(tmp_path / "miles.jsonl")
    n = export_for_miles(man, out)
    assert n == 40
    import json
    row = json.loads(open(out).readline())
    assert set(row) == {"prompt", "label", "metadata"}
    assert row["label"].isdigit(), "label must be the seed the reward rebuilds from"
    assert "```python" in row["prompt"]
