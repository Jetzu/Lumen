#!/usr/bin/env python3
"""Analyse the CSV logs of the Lumen vs Fiji user study (docs/usability_study/README.md).

Pure Python, standard library only. Reads a folder with

  participants.csv   participant,group,order,fiji_years,scripting,os,screen,input_device,assistive_tech,notes
  task_log.csv       participant,tool,task,completion,time_s,err_wrong_command,err_wrong_window,
                     err_destructive,err_cancelled,first_lookup,actions,notes
  sus_responses.csv  participant,tool,q1,...,q10,open_stop,open_missed

and prints a Markdown report with, per tool: SUS scores (mean, SD, 95 % CI, per participant, paired
difference), per-task completion / assist / abandon rates, time (median, mean, IQR) and errors,
paired Wilcoxon signed-rank tests (exact for n <= 15, normal approximation above) with the
matched-pairs rank-biserial correlation as effect size, the first-lookup distribution, and the
pass criteria of README.md section 6.

Usage:
  python analyze_results.py <folder> [--out report.md] [--tool-a fiji --tool-b lumen]
  python analyze_results.py --selftest
  python analyze_results.py --make-example <folder>     # write a synthetic data set (10 participants)

Values of `completion`: complete | assist | abandoned.  Values of `tool`: fiji | lumen (case-insensitive).
`time_s` in seconds; error columns are counts; empty cells are allowed and skipped.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

TASKS = list(range(1, 11))
TASK_NAMES = {
    1: "Open 3-channel image, adjust channels", 2: "Threshold + Analyze Particles", 3: "Measure a drawn rectangle",
    4: "Z-projection", 5: "Composite, LUT per channel", 6: "Filter with preview, undo", 7: "Track cells, export",
    8: "Find a command (CLAHE)", 9: "Batch a folder from the recording", 10: "Compare two images side by side",
}
ERROR_COLS = ["err_wrong_command", "err_wrong_window", "err_destructive", "err_cancelled"]
COMPLETIONS = ("complete", "assist", "abandoned")

# Two-sided 95 % critical values of Student's t for df = 1..30; beyond that a conservative 2.0 / 1.98 / 1.96.
T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
       2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


# ------------------------------------------------------------------ statistics helpers
def t_crit(df: int) -> float:
    if df < 1:
        return float("nan")
    if df <= 30:
        return T95[df - 1]
    if df <= 60:
        return 2.0
    if df <= 120:
        return 1.98
    return 1.96


def mean_ci(values: list[float]) -> dict[str, float]:
    """Mean, SD and 95 % confidence interval (t-distribution) of a sample; NaN where undefined."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "lo": float("nan"), "hi": float("nan")}
    m = statistics.fmean(values)
    if n == 1:
        return {"n": 1, "mean": m, "sd": float("nan"), "lo": float("nan"), "hi": float("nan")}
    sd = statistics.stdev(values)
    half = t_crit(n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "lo": m - half, "hi": m + half}


def average_ranks(values: list[float]) -> list[float]:
    """Ranks 1..n with ties averaged."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def normal_sf(z: float) -> float:
    """Upper tail of the standard normal distribution."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def wilcoxon_signed_rank(x: list[float], y: list[float], exact_max_n: int = 15) -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank test for paired samples x, y (differences y - x).

    Zero differences are dropped (Wilcoxon's convention). Exact permutation p for n <= exact_max_n
    (enumerates the 2^n sign assignments on the observed ranks, so ties are handled correctly);
    normal approximation with tie and continuity correction above. Effect size: matched-pairs
    rank-biserial correlation r = (W+ - W-) / (W+ + W-), positive when y > x.
    """
    diffs = [b - a for a, b in zip(x, y) if b - a != 0]
    n = len(diffs)
    if n == 0:
        return {"n": 0, "w_plus": 0.0, "w_minus": 0.0, "p": float("nan"), "r": float("nan"), "method": "no non-zero differences"}
    ranks = average_ranks([abs(d) for d in diffs])
    w_plus = sum(r for r, d in zip(ranks, diffs) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, diffs) if d < 0)
    total = w_plus + w_minus
    r_rb = (w_plus - w_minus) / total
    if n <= exact_max_n:
        w_obs = min(w_plus, w_minus)
        count = 0
        for signs in itertools.product((0, 1), repeat=n):
            w = sum(r for r, s in zip(ranks, signs) if s)
            if w <= w_obs + 1e-9:
                count += 1
        p = min(1.0, 2 * count / 2 ** n)
        method = "exact"
    else:
        mu = n * (n + 1) / 4
        tie_groups = Counter(ranks).values()
        tie_term = sum(t ** 3 - t for t in tie_groups) / 48
        sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - tie_term)
        w = min(w_plus, w_minus)
        z = (w - mu + 0.5) / sigma if sigma > 0 else 0.0
        p = min(1.0, 2 * (1 - normal_sf(z)) if z > 0 else 2 * normal_sf(-z)) if sigma > 0 else 1.0
        p = min(1.0, 2 * normal_sf(abs(w - mu) - 0.5) / 1) if False else p  # keep the corrected form above
        method = "normal approximation"
    return {"n": n, "w_plus": w_plus, "w_minus": w_minus, "p": p, "r": r_rb, "method": method}


def sus_score(answers: list[int | None]) -> tuple[float, int]:
    """SUS score 0-100 from ten answers (1-5); missing answers imputed as 3. Returns (score, n_imputed)."""
    if len(answers) != 10:
        raise ValueError("SUS needs exactly ten answers")
    total = 0
    imputed = 0
    for i, a in enumerate(answers, 1):
        if a is None:
            a = 3
            imputed += 1
        if not 1 <= a <= 5:
            raise ValueError(f"SUS answer {a!r} for item {i} is not in 1..5")
        total += (a - 1) if i % 2 == 1 else (5 - a)
    return total * 2.5, imputed


# ------------------------------------------------------------------ data loading
def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def _int(s: str) -> int | None:
    s = s.strip()
    if not s:
        return None
    return int(float(s))


def _float(s: str) -> float | None:
    s = s.strip()
    return float(s) if s else None


class Study:
    def __init__(self, folder: Path, tool_a: str = "fiji", tool_b: str = "lumen") -> None:
        self.folder = folder
        self.tool_a, self.tool_b = tool_a.lower(), tool_b.lower()
        self.participants = _read(folder / "participants.csv") if (folder / "participants.csv").exists() else []
        self.task_log = [r for r in _read(folder / "task_log.csv") if r.get("participant")]
        self.sus = [r for r in _read(folder / "sus_responses.csv") if r.get("participant")] if (folder / "sus_responses.csv").exists() else []
        self.problems: list[str] = []
        for r in self.task_log:
            r["tool"] = r.get("tool", "").lower()
            r["completion"] = r.get("completion", "").lower()
            if r["completion"] not in COMPLETIONS:
                self.problems.append(f"task_log: participant {r['participant']} tool {r['tool']} task {r.get('task')}: completion {r['completion']!r} is not one of {COMPLETIONS}")
            if r["tool"] not in (self.tool_a, self.tool_b):
                self.problems.append(f"task_log: participant {r['participant']}: tool {r['tool']!r} is neither {self.tool_a!r} nor {self.tool_b!r}")
        for r in self.sus:
            r["tool"] = r.get("tool", "").lower()

    # --- accessors
    def tools(self) -> list[str]:
        return [self.tool_a, self.tool_b]

    def rows(self, tool: str, task: int | None = None) -> list[dict[str, str]]:
        return [r for r in self.task_log if r["tool"] == tool and (task is None or _int(r.get("task", "")) == task)]

    def participant_ids(self) -> list[str]:
        ids = {r["participant"] for r in self.task_log} | {r["participant"] for r in self.sus}
        return sorted(ids)

    def sus_scores(self, tool: str) -> dict[str, tuple[float, int]]:
        out = {}
        for r in self.sus:
            if r["tool"] != tool:
                continue
            answers = [_int(r.get(f"q{i}", "")) for i in range(1, 11)]
            out[r["participant"]] = sus_score(answers)
        return out

    def paired(self, metric: Any, task: int | None = None) -> tuple[list[float], list[float], list[str]]:
        """Per-participant values of metric(row) for tool A and tool B, for participants with both."""
        a: dict[str, float] = {}
        b: dict[str, float] = {}
        for tool, store in ((self.tool_a, a), (self.tool_b, b)):
            per: dict[str, list[float]] = defaultdict(list)
            for r in self.rows(tool, task):
                v = metric(r)
                if v is not None:
                    per[r["participant"]].append(v)
            for pid, vals in per.items():
                store[pid] = sum(vals)
        ids = sorted(set(a) & set(b))
        return [a[i] for i in ids], [b[i] for i in ids], ids


# ------------------------------------------------------------------ report
def fmt(v: float, nd: int = 1) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def pct(num: int, den: int) -> str:
    return "n/a" if den == 0 else f"{100 * num / den:.0f} % ({num}/{den})"


def report(st: Study) -> str:
    A, B = st.tool_a, st.tool_b
    L: list[str] = []
    L.append(f"# Lumen vs Fiji user study: results\n")
    L.append(f"Data folder: `{st.folder}`. Participants with task rows: {len({r['participant'] for r in st.task_log})}; "
             f"task rows: {len(st.task_log)}; SUS rows: {len(st.sus)}.\n")
    if st.problems:
        L.append("**Data problems** (fix these first):\n")
        L += [f"- {p}" for p in st.problems[:30]]
        L.append("")

    # --- participants
    if st.participants:
        L.append("## Participants\n")
        L.append("| ID | group | order | Fiji years | scripting | OS | screen | input | assistive tech |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for p in st.participants:
            L.append(f"| {p.get('participant','')} | {p.get('group','')} | {p.get('order','')} | {p.get('fiji_years','')} | {p.get('scripting','')} | {p.get('os','')} | {p.get('screen','')} | {p.get('input_device','')} | {p.get('assistive_tech','')} |")
        L.append("")

    # --- SUS
    L.append("## System Usability Scale\n")
    sus_a, sus_b = st.sus_scores(A), st.sus_scores(B)
    ci_a, ci_b = mean_ci([s for s, _ in sus_a.values()]), mean_ci([s for s, _ in sus_b.values()])
    L.append(f"| Tool | n | mean | SD | 95 % CI | imputed answers |")
    L.append("|---|---|---|---|---|---|")
    for tool, sc, ci in ((A, sus_a, ci_a), (B, sus_b, ci_b)):
        L.append(f"| {tool} | {ci['n']} | {fmt(ci['mean'])} | {fmt(ci['sd'])} | {fmt(ci['lo'])} – {fmt(ci['hi'])} | {sum(i for _, i in sc.values())} |")
    both = sorted(set(sus_a) & set(sus_b))
    if both:
        diffs = [sus_b[p][0] - sus_a[p][0] for p in both]
        cid = mean_ci(diffs)
        w = wilcoxon_signed_rank([sus_a[p][0] for p in both], [sus_b[p][0] for p in both])
        L.append(f"\nPaired difference {B} − {A} (n = {len(both)}): mean {fmt(cid['mean'])}, 95 % CI {fmt(cid['lo'])} – {fmt(cid['hi'])}; "
                 f"Wilcoxon W+ = {fmt(w['w_plus'])}, W− = {fmt(w['w_minus'])}, p = {fmt(w['p'], 3)} ({w['method']}), rank-biserial r = {fmt(w['r'], 2)}.")
        L.append(f"\nReference: the published average SUS across studies is 68 (Sauro). Per participant ({A} / {B}): " +
                 ", ".join(f"{p} {sus_a[p][0]:.0f}/{sus_b[p][0]:.0f}" for p in both) + ".")
    L.append("")

    # --- completion per task
    L.append("## Task completion\n")
    L.append(f"| Task | {A}: complete | assist | abandoned | {B}: complete | assist | abandoned |")
    L.append("|---|---|---|---|---|---|---|")
    completion: dict[str, dict[int, Counter[str]]] = {A: {}, B: {}}
    for t in TASKS:
        cells = []
        for tool in (A, B):
            rows = st.rows(tool, t)
            c = Counter(r["completion"] for r in rows)
            completion[tool][t] = c
            cells += [pct(c["complete"], len(rows)), pct(c["assist"], len(rows)), pct(c["abandoned"], len(rows))]
        L.append(f"| {t} {TASK_NAMES[t]} | " + " | ".join(cells) + " |")
    L.append("")

    # --- time per task
    L.append("## Time on task (seconds)\n")
    L.append(f"| Task | {A}: n | median | mean | IQR | {B}: n | median | mean | IQR | Wilcoxon p | r |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in TASKS:
        cells = []
        for tool in (A, B):
            ts = [v for v in (_float(r.get("time_s", "")) for r in st.rows(tool, t)) if v is not None]
            if ts:
                q = statistics.quantiles(ts, n=4) if len(ts) >= 2 else [ts[0], ts[0], ts[0]]
                cells += [str(len(ts)), fmt(statistics.median(ts), 0), fmt(statistics.fmean(ts), 0), f"{fmt(q[0], 0)}–{fmt(q[2], 0)}"]
            else:
                cells += ["0", "n/a", "n/a", "n/a"]
        xa, xb, _ = st.paired(lambda r: _float(r.get("time_s", "")), t)
        w = wilcoxon_signed_rank(xa, xb) if len(xa) >= 2 else {"p": float("nan"), "r": float("nan")}
        L.append(f"| {t} | " + " | ".join(cells) + f" | {fmt(w['p'], 3)} | {fmt(w['r'], 2)} |")
    xa, xb, ids = st.paired(lambda r: _float(r.get("time_s", "")))
    if len(xa) >= 2:
        w = wilcoxon_signed_rank(xa, xb)
        L.append(f"\nTotal time over all tasks, paired (n = {len(ids)}): {A} mean {fmt(statistics.fmean(xa), 0)} s, {B} mean {fmt(statistics.fmean(xb), 0)} s; "
                 f"Wilcoxon p = {fmt(w['p'], 3)} ({w['method']}), r = {fmt(w['r'], 2)} (negative = {B} faster).")
    L.append("")

    # --- errors
    L.append("## Errors (mean per participant and task)\n")
    L.append(f"| Task | {A}: wrong command | wrong window | destructive | cancelled | total | {B}: wrong command | wrong window | destructive | cancelled | total | Wilcoxon p (total) | r |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    def err_total(r: dict[str, str]) -> float | None:
        vals = [_int(r.get(c, "")) for c in ERROR_COLS]
        if all(v is None for v in vals):
            return None
        return float(sum(v or 0 for v in vals))

    for t in TASKS:
        cells = []
        for tool in (A, B):
            rows = st.rows(tool, t)
            n = len(rows)
            tot = 0.0
            for c in ERROR_COLS:
                vals = [v for v in (_int(r.get(c, "")) for r in rows) if v is not None]
                m = sum(vals) / n if n else float("nan")
                tot += sum(vals) / n if n else 0
                cells.append(fmt(m, 2))
            cells.append(fmt(tot if n else float("nan"), 2))
        xa, xb, _ = st.paired(err_total, t)
        w = wilcoxon_signed_rank(xa, xb) if len(xa) >= 2 else {"p": float("nan"), "r": float("nan")}
        L.append(f"| {t} | " + " | ".join(cells) + f" | {fmt(w['p'], 3)} | {fmt(w['r'], 2)} |")
    L.append("")

    # --- first lookup
    L.append("## Where participants looked first\n")
    L.append(f"| Tool | " + " | ".join(["menu", "search", "memory", "help", "asked", "other/blank"]) + " |")
    L.append("|---|---|---|---|---|---|---|")
    for tool in (A, B):
        c = Counter((r.get("first_lookup", "") or "").lower() for r in st.rows(tool))
        known = ["menu", "search", "memory", "help", "asked"]
        other = sum(v for k, v in c.items() if k not in known)
        L.append(f"| {tool} | " + " | ".join(str(c[k]) for k in known) + f" | {other} |")
    L.append("")

    # --- actions (optional)
    acts = {tool: [v for v in (_float(r.get("actions", "")) for r in st.rows(tool)) if v is not None] for tool in (A, B)}
    if any(acts.values()):
        L.append("## Actions (clicks + keys + text entries, where counted)\n")
        for tool in (A, B):
            if acts[tool]:
                per_task = {t: [v for v in (_float(r.get("actions", "")) for r in st.rows(tool, t)) if v is not None] for t in TASKS}
                L.append(f"- {tool}: " + ", ".join(f"task {t} mean {fmt(statistics.fmean(v))}" for t, v in per_task.items() if v))
        L.append("")

    # --- pass criteria
    L.append("## Pass criteria for Lumen (README.md section 6)\n")
    lumen = B if B == "lumen" else A
    fiji = A if lumen == B else B
    crit1 = all((completion[lumen][t]["complete"] / sum(completion[lumen][t].values())) >= 0.9 for t in TASKS if t not in (7, 9) and sum(completion[lumen][t].values()))
    crit1_ok = crit1 and all(sum(completion[lumen][t].values()) for t in TASKS if t not in (7, 9))
    crit2_ok = all(sum(completion[lumen][t].values()) and (completion[lumen][t]["complete"] / sum(completion[lumen][t].values())) >= 0.75 for t in (7, 9))
    sl, sf = st.sus_scores(lumen), st.sus_scores(fiji)
    crit3_ok = bool(sl) and bool(sf) and statistics.fmean(s for s, _ in sl.values()) >= statistics.fmean(s for s, _ in sf.values())
    majority_abandon = [t for t in TASKS if sum(completion[lumen][t].values()) and completion[lumen][t]["abandoned"] / sum(completion[lumen][t].values()) > 0.5]
    L.append(f"1. Completion without assist ≥ 90 % on tasks 1-6, 8-10: **{'met' if crit1_ok else 'not met'}**")
    L.append(f"2. Completion without assist ≥ 75 % on tasks 7 and 9: **{'met' if crit2_ok else 'not met'}**")
    L.append(f"3. Mean SUS for {lumen} not lower than for {fiji}: **{'met' if crit3_ok else 'not met'}** ({fmt(statistics.fmean(s for s, _ in sl.values())) if sl else 'n/a'} vs {fmt(statistics.fmean(s for s, _ in sf.values())) if sf else 'n/a'})")
    L.append(f"4. No severity-4 finding: judged from the qualitative coding; tasks abandoned by a majority in {lumen}: {majority_abandon or 'none'}")
    L.append("")
    L.append("Interpretation notes: with 8-12 participants the study detects large differences only; report the "
             "effect sizes and intervals above together with the qualitative findings, not the p-values alone.")
    return "\n".join(L)


# ------------------------------------------------------------------ synthetic data and self-test
def make_example(folder: Path, n: int = 10, seed: int = 7) -> None:
    """Write a clearly synthetic data set (participant IDs X01…) for trying the script."""
    rng = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "participants.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant", "group", "order", "fiji_years", "scripting", "os", "screen", "input_device", "assistive_tech", "notes"])
        for i in range(1, n + 1):
            w.writerow([f"X{i:02d}", "fiji-weekly", "fiji-first" if i % 2 else "lumen-first", rng.randint(1, 12),
                        rng.choice(["none", "macros", "python"]), rng.choice(["macOS", "Windows", "Linux"]), "13-inch 1440x900",
                        rng.choice(["mouse", "trackpad"]), "no", "SYNTHETIC EXAMPLE DATA"])
    base_time = {1: 60, 2: 90, 3: 30, 4: 40, 5: 70, 6: 60, 7: 240, 8: 40, 9: 200, 10: 40}
    with (folder / "task_log.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant", "tool", "task", "completion", "time_s", *ERROR_COLS, "first_lookup", "actions", "notes"])
        for i in range(1, n + 1):
            for tool, factor in (("fiji", 1.0), ("lumen", 0.8)):
                for t in TASKS:
                    tm = max(10, rng.gauss(base_time[t] * factor, base_time[t] * 0.25))
                    r = rng.random()
                    comp = "complete" if r < (0.85 if t in (7, 9) else 0.93) else ("assist" if r < 0.97 else "abandoned")
                    if comp == "abandoned":
                        tm = 300
                    errs = [rng.choice([0, 0, 0, 1]), rng.choice([0, 0, 1]) if tool == "fiji" else rng.choice([0, 0, 0, 1]), rng.choice([0, 0, 0, 0, 1]), rng.choice([0, 0, 1])]
                    look = rng.choice(["menu", "menu", "search", "memory"]) if tool == "fiji" else rng.choice(["menu", "search", "search", "memory"])
                    w.writerow([f"X{i:02d}", tool, t, comp, f"{tm:.0f}", *errs, look, "", "synthetic"])
    with (folder / "sus_responses.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant", "tool", *[f"q{i}" for i in range(1, 11)], "open_stop", "open_missed"])
        for i in range(1, n + 1):
            for tool, bias in (("fiji", 0.0), ("lumen", 0.4)):
                answers = []
                for q in range(1, 11):
                    good = rng.gauss(3.6 + bias, 0.8)
                    a = round(good) if q % 2 == 1 else round(6 - good)
                    answers.append(min(5, max(1, a)))
                w.writerow([f"X{i:02d}", tool, *answers, "synthetic answer", "synthetic answer"])
    print(f"wrote synthetic example data for {n} participants to {folder}")


def selftest() -> int:
    failures: list[str] = []

    def check(cond: bool, what: str) -> None:
        if not cond:
            failures.append(what)

    # SUS scoring
    check(sus_score([5, 1, 5, 1, 5, 1, 5, 1, 5, 1])[0] == 100.0, "SUS best case = 100")
    check(sus_score([1, 5, 1, 5, 1, 5, 1, 5, 1, 5])[0] == 0.0, "SUS worst case = 0")
    check(sus_score([3] * 10)[0] == 50.0, "SUS all-neutral = 50")
    s, imp = sus_score([4, 2, 4, None, 4, 2, 4, 2, 4, 2])
    check(s == 72.5 and imp == 1, f"SUS with one imputed answer = 72.5 (got {s}, imputed {imp})")
    try:
        sus_score([6] + [3] * 9)
        check(False, "SUS rejects an answer of 6")
    except ValueError:
        pass
    # Wilcoxon: hand-computed example (differences 15,-7,5,20,0,-9,17,-12,5,-10 -> W+ = 27, W- = 18, n = 9)
    x = [125, 115, 130, 140, 140, 115, 140, 125, 140, 135]
    y = [110, 122, 125, 120, 140, 124, 123, 137, 135, 145]
    w = wilcoxon_signed_rank(y, x)  # differences x - y
    check(w["n"] == 9 and abs(w["w_plus"] - 27) < 1e-9 and abs(w["w_minus"] - 18) < 1e-9, f"Wilcoxon ranks on the textbook example (got {w})")
    check(0 < w["p"] <= 1 and w["method"] == "exact", "Wilcoxon exact p in (0, 1]")
    check(abs(w["r"] - (27 - 18) / 45) < 1e-9, "rank-biserial r")
    # symmetric case: identical pairs -> no test
    w0 = wilcoxon_signed_rank([1, 2, 3], [1, 2, 3])
    check(w0["n"] == 0 and math.isnan(w0["p"]), "all-zero differences handled")
    # one-sided extreme: all differences positive, n = 8 -> exact p = 2 / 256
    w1 = wilcoxon_signed_rank([0] * 8, [1, 2, 3, 4, 5, 6, 7, 8])
    check(abs(w1["p"] - 2 / 256) < 1e-12 and w1["r"] == 1.0, f"Wilcoxon extreme case p = 2/256 (got {w1['p']})")
    # exact vs normal approximation agree on a moderate sample
    rng = random.Random(3)
    a = [rng.gauss(0, 1) for _ in range(15)]
    b = [v + 0.6 + rng.gauss(0, 1) for v in a]
    pe = wilcoxon_signed_rank(a, b)["p"]
    pn = wilcoxon_signed_rank(a, b, exact_max_n=0)["p"]
    check(abs(pe - pn) < 0.06, f"exact and normal p agree within 0.06 (exact {pe:.3f}, normal {pn:.3f})")
    # average ranks with ties
    check(average_ranks([5, 5, 7, 9]) == [1.5, 1.5, 3, 4], "average ranks")
    # confidence interval
    ci = mean_ci([70, 75, 80, 85, 90])
    check(abs(ci["mean"] - 80) < 1e-9 and abs((ci["hi"] - ci["lo"]) / 2 - 2.776 * statistics.stdev([70, 75, 80, 85, 90]) / math.sqrt(5)) < 1e-9, "95 % CI uses t(4) = 2.776")
    # end-to-end on synthetic data in a temporary folder
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        make_example(Path(tmp), n=10)
        st = Study(Path(tmp))
        check(not st.problems, f"synthetic data parses without problems: {st.problems[:3]}")
        text = report(st)
        check("## System Usability Scale" in text and "## Pass criteria" in text, "report has all sections")
        check(len(st.sus_scores("lumen")) == 10 and len(st.sus_scores("fiji")) == 10, "ten SUS scores per tool")
    for f in failures:
        print("FAIL:", f)
    print("selftest:", "OK" if not failures else f"{len(failures)} failure(s)")
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", nargs="?", help="folder with participants.csv, task_log.csv, sus_responses.csv")
    ap.add_argument("--out", help="write the Markdown report to this file instead of stdout")
    ap.add_argument("--tool-a", default="fiji")
    ap.add_argument("--tool-b", default="lumen")
    ap.add_argument("--selftest", action="store_true", help="check the scoring and test code on known values and synthetic data")
    ap.add_argument("--make-example", metavar="FOLDER", help="write a synthetic example data set to FOLDER and exit")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.make_example:
        make_example(Path(args.make_example))
        return 0
    if not args.folder:
        ap.error("give a data folder, --selftest or --make-example")
    folder = Path(args.folder)
    if not (folder / "task_log.csv").exists():
        print(f"{folder / 'task_log.csv'} not found", file=sys.stderr)
        return 2
    text = report(Study(folder, args.tool_a, args.tool_b))
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
