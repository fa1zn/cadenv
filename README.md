# cadenv

A CAD generation environment whose reward is gated against its own exploits.

The environment is ordinary: a natural-language part spec goes in, a solid comes
out, and a reward says whether it is right. The part that isn't ordinary is that
the reward has a CI check pointed at it. Attack policies that never solve the task
run against the reward on every commit, and if one of them scores, the build fails.

```bash
python -m cadenv.cli --all        # gate every reward, exit 1 if any is exploitable
```

## Why

This came out of auditing a delivered CAD corpus whose shipped reward was wrong in
both directions. It rejected the *ground truth* on 114 of 150 samples, because it
compared an axis-aligned bounding box across parts in different poses. And it
accepted a twenty-four-line policy that never read the drawing, because volume is a
single scalar and many shapes share it.

Neither failure was found by looking at the reward. Both were found by running
things at it that shouldn't have scored.

## What the gate does

```
reward: axis-aligned-box   tasks: 12

  attack policy           knows                             scores
  pocketed-block          target volume + bounding box      100.0%  <-- EXPLOIT
  constant-cube           nothing                             0.0%
  volume-cube             target volume                       0.0%
  volume-sphere           target volume                       0.0%
  scaled-reference-1.1    the answer, scaled wrong            0.0%

  must pass               knows                             scores
  exact-reference         the answer                        100.0%
  rotated-reference       the answer, rotated                 0.0%  <-- REJECTS A CORRECT ANSWER

  GATE: FAIL
```

Both directions are checked. A reward that admits a hollow box is broken; so is one
that rejects a correct part because someone rotated it. The corpus this came from
did both at once, which is what the `axis-aligned-box` reward reproduces.

| reward | worst attack | rejects a correct answer | gate |
|---|---|---|---|
| `volume-only` | 100% (three different primitives) | no | **FAIL** |
| `axis-aligned-box` | 100% (pocketed block) | yes (rotated) | **FAIL** |
| `pose-invariant` | 0% | no | **PASS** |

Two of those three rewards are deliberately broken and ship anyway. A gate that has
never caught anything is not evidence of a safe reward, so the test suite runs
against known-bad rewards and asserts they fail.

## The attack suite

| policy | knows | defeats |
|---|---|---|
| `constant-cube` | nothing | a reward with no reference at all |
| `volume-cube` | target volume | any volume-only gate |
| `volume-sphere` | target volume | a volume gate that was "fixed" by special-casing boxes |
| `pocketed-block` | target volume + bounding box | volume and axis-aligned-bbox gates simultaneously |
| `scaled-reference-1.1` | the answer, scaled wrong | control — must be rejected |
| `exact-reference` | the answer | control — must be accepted |
| `rotated-reference` | the answer, rotated | any frame-dependent comparison |

Three of these are handed the answer key's summary statistics, which is deliberate.
In a real loop the policy never sees the key, but the *reward is computed from it*,
so gradient descent is free to find exactly these solutions. What the suite measures
is what the reward permits — an upper bound on how badly it can be gamed, not a
prediction about any particular model.

## Tasks

Generated, not downloaded. A part is a *program* — a sequence of operations sampled
from a grammar over five base shapes, six cut types, three additive features, fillets
and chamfers, and three pattern types — so the reachable space grows with program
length rather than being fixed by a template's parameters. The prompt is written from
the same numbers that built the solid, so ground truth is exact by construction and
the spec cannot drift from the part.

```python
from cadenv.task import procedural_taskset, corpus_taskset
tasks = procedural_taskset(50)                 # self-contained, no download
tasks = corpus_taskset("~/path/to/corpus")     # a delivered dataset instead
```

### A 10,000-task set

```bash
python -c "from cadenv.dataset import build_dataset; \
           print(build_dataset(10000, 'data/tasks-10k.jsonl'))"
```

| | |
|---|---|
| kept / attempted | 10,000 / 12,596 — **79.4% yield** |
| build time | 303 s |
| duplicates rejected | 82, by rigid-motion-invariant signature |
| tiers | easy 3,420 · medium 3,887 · **hard 2,693** |
| manifest | 5.7 MB |

**The artifact is a manifest, not geometry.** Every row carries the seed that
produced it, so 10,000 solids reproduce byte-for-byte from 5.7 MB of JSONL instead of
shipping ~500 MB of STEP. That also gives the strongest integrity check available: if
a row's recorded invariants don't match what its seed rebuilds, something is wrong.
300 sampled rows rebuild with **0 mismatches**.

**Diversity is measured, not assumed**, because ten thousand samples from a template
is one task repeated:

| | 558 parts | 1,200 parts |
|---|---|---|
| unique signatures | 557 | 1,200 |
| duplicate rate | 0.18% | 0.0% |
| near-identical pairs | 0.0% | 0.0% |
| pairwise median distance | 2.826 | 2.826 |

The pairwise median is unchanged between the two, so the generator is not collapsing
as it scales.

**The difficulty tiers are honest about what they are.** `complexity` is a weighted
sum over the program's operations — a structural proxy. That is still an assertion,
which is exactly the criticism this project levelled at a corpus that labelled
everything "easy" and never checked. Calibrating the tiers against observed solve
rates needs a reference solver and has not been done. What can be said is that the
spread exists at all: 27% hard, against 0% in the corpus this replaces.

## The sound reward

`pose-invariant` is built on [cadverify](https://github.com/fa1zn/cadverify) and
follows two rules.

**Never compare a frame-dependent quantity without aligning first.** Bounding box as
an ordered triple, centre-of-mass coordinates, vertex positions — all meaningless
until both shapes are in the same frame. Sorting the three box dimensions does *not*
rescue it; only 90° axis-aligned rotations permute a box, and a general rotation
changes its size.

**Invariants alone are necessary and not sufficient.** Volume, surface area and
inertia eigenvalues survive any rigid motion, which makes them safe to compare and
useless on their own — a single scalar is shared by infinitely many shapes. Admission
needs invariants; rejection needs alignment plus a pointwise distance.

## Install and test

```bash
pip install -e ".[dev]"
pytest                    # 16 tests
pytest -m "not slow"      # skips the pose-invariant gate (~0.6s per alignment)
```

## Limitations

- **The generator makes machined-plate-like parts.** Five bases, cuts, bosses, ribs,
  fillets and patterns — genuinely varied within that family, and still nothing like
  a casting, a sheet-metal part, or an assembly.
- **Difficulty tiers are a structural proxy**, not a measured solve rate. Calibrating
  them needs a reference solver.
- **79.4% of sampled programs build.** The 20% that fail are mostly fillet and chamfer
  edge-selection failures, so the shipped distribution is biased away from the parts
  where those operations are hardest to apply.
- **The attack suite is the exploits I thought of.** A gate is a lower bound on
  reward soundness — passing means no *known* attack scores, never that none exists.
  New attacks belong in `policies.py`, not in a comment.
- **`pose-invariant` is slow.** About 0.6s per alignment, so a 12-task gate is ~50s.
  Fine for CI, too slow to sit inside a training loop without caching.
- **A 0.5% shape threshold is a judgement call**, inherited from cadverify, not a
  derived value.
- **No agent loop.** This is the task, the reward, and the gate. Plugging it into an
  RL harness is left to the harness.
