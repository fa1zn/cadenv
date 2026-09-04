"""The admission gate: a CI check on the reward, not on the model.

Run every attack policy against a reward. If one scores, the reward pays out for
work that isn't the task, and the gate fails. Run every must-pass policy too — a
reward that rejects the correct answer fails just as hard, and that direction is
the one the corpus this came from actually got wrong.

The point of wiring this into CI is that a reward change which opens an exploit
breaks the build, rather than being discovered months later in a trained policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .policies import ATTACKS, MUST_PASS


@dataclass
class PolicyScore:
    policy: str
    knows: str
    pass_rate: float
    n: int
    detail: dict = field(default_factory=dict)


@dataclass
class GateReport:
    reward: str
    attacks: list = field(default_factory=list)
    must_pass: list = field(default_factory=list)
    n_tasks: int = 0

    @property
    def worst_attack(self) -> PolicyScore | None:
        return max(self.attacks, key=lambda s: s.pass_rate, default=None)

    @property
    def worst_legitimate(self) -> PolicyScore | None:
        return min(self.must_pass, key=lambda s: s.pass_rate, default=None)

    def passed(self, max_attack=0.0, min_legitimate=1.0) -> bool:
        a = self.worst_attack
        l = self.worst_legitimate
        return ((a is None or a.pass_rate <= max_attack)
                and (l is None or l.pass_rate >= min_legitimate))


def _score(policy, reward, tasks) -> PolicyScore:
    hits = 0
    failed_gates = {}
    for t in tasks:
        try:
            res = reward(t, policy(t))
        except Exception as e:                       # a policy that crashes scores 0
            failed_gates[f"error:{type(e).__name__}"] = \
                failed_gates.get(f"error:{type(e).__name__}", 0) + 1
            continue
        if res.passed:
            hits += 1
        else:
            failed_gates[res.failed_gate or "?"] = failed_gates.get(res.failed_gate or "?", 0) + 1
    return PolicyScore(policy.name, policy.knows, hits / len(tasks), len(tasks), failed_gates)


def run_gate(reward, tasks, attacks=None, must_pass=None) -> GateReport:
    rep = GateReport(reward=reward.name, n_tasks=len(tasks))
    for p in (attacks if attacks is not None else ATTACKS):
        rep.attacks.append(_score(p, reward, tasks))
    for p in (must_pass if must_pass is not None else MUST_PASS):
        rep.must_pass.append(_score(p, reward, tasks))
    return rep


def format_report(rep: GateReport) -> str:
    out = [f"reward: {rep.reward}   tasks: {rep.n_tasks}", ""]
    out.append(f"  {'attack policy':<24}{'knows':<32}{'scores':>8}")
    for s in sorted(rep.attacks, key=lambda s: -s.pass_rate):
        flag = "  <-- EXPLOIT" if s.pass_rate > 0 else ""
        out.append(f"  {s.policy:<24}{s.knows:<32}{s.pass_rate*100:>7.1f}%{flag}")
    out.append("")
    out.append(f"  {'must pass':<24}{'knows':<32}{'scores':>8}")
    for s in rep.must_pass:
        flag = "  <-- REJECTS A CORRECT ANSWER" if s.pass_rate < 1.0 else ""
        out.append(f"  {s.policy:<24}{s.knows:<32}{s.pass_rate*100:>7.1f}%{flag}")
    out.append("")
    out.append(f"  GATE: {'PASS' if rep.passed() else 'FAIL'}")
    return "\n".join(out)
