"""Run the admission gate from the command line."""

import argparse

from .gate import format_report, run_gate
from .reward import REWARDS
from .task import corpus_taskset, procedural_taskset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reward", default="pose-invariant", choices=list(REWARDS))
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--corpus", help="score against a delivered corpus instead")
    ap.add_argument("--all", action="store_true", help="gate every reward in turn")
    a = ap.parse_args()

    tasks = (corpus_taskset(a.corpus, limit=a.tasks) if a.corpus
             else procedural_taskset(a.tasks))
    names = list(REWARDS) if a.all else [a.reward]
    failed = False
    for n in names:
        rep = run_gate(REWARDS[n], tasks)
        print(format_report(rep))
        print("=" * 70)
        failed |= not rep.passed()
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
