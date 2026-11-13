# Lumen vs Fiji user-study kit

Everything needed to run the small comparative study proposed in
[docs/USABILITY.md §5.2](../USABILITY.md): a within-subject, counterbalanced comparison of Lumen
and Fiji on the ten everyday tasks of USABILITY.md §2.1, with 8-12 Fiji users (plus an optional
learnability group of 4 non-users). Nothing in this folder has been used with participants yet;
the pass criteria and the analysis were fixed before any data exist, which is the point.

| File | What it is |
|---|---|
| `README.md` | this protocol: design, participants, session plan, measures, analysis, pass criteria |
| `session_script.md` | what the moderator says and does, minute by minute |
| `participant_briefing.md` | one page handed to (or read to) the participant before consent |
| `consent_form.md` | generic consent text (adapt to your institution's template and ethics process) |
| `task_sheets.md` | the ten task cards with materials, end state and success criteria, one card per tool |
| `sus_questionnaire.md` | the System Usability Scale, ten items, with the scoring rule |
| `templates/participants.csv` | one row per participant: demographics and tool order |
| `templates/task_log.csv` | one row per participant × tool × task: completion, time, errors, first lookup |
| `templates/sus_responses.csv` | one row per participant × tool: the ten SUS answers (1-5) |
| `analyze_results.py` | pure-Python analysis of the three CSVs (SUS, completion, time, errors, paired tests, pass criteria) |
| `example_data/` | a **synthetic** data set (participants X01-X10, written by `analyze_results.py --make-example`) so the report can be tried before any session; never mix it with real data |

## 1. Design

- **Within-subject.** Every participant does all ten tasks in both tools.
- **Counterbalanced.** Participants with odd IDs start with Fiji, even IDs with Lumen (fill the
  `order` column of `participants.csv` before the session). The task order is the same in both
  tools (task 9 must follow task 6 so that a recording exists).
- **One 75-minute session** per participant, think-aloud, screen and audio recorded, moderator
  present, one observer logging (two people are better than one: the moderator keeps the session
  moving, the observer keeps the log).
- **Materials identical in both tools**: the sample set of task_sheets.md §0 (a 3-channel
  fluorescence image, an 8-bit blob image, a 2-channel z-stack, a nuclei time-lapse, a folder of
  ten small TIFFs). Fiji 1.54 as installed with default settings and an empty `IJ_Prefs.txt`;
  Lumen served from this repository with an empty `~/.lumen` (`python -m lumen`), in Chrome or
  Firefox at the laptop's native resolution. Reset both between participants.
- **No hints** about menu locations, the palette or shortcuts beyond the two-minute tour in
  `session_script.md`. The moderator answers "what does this do?" with "what do you think?" and
  gives an assist only after the five-minute abandon rule below.

## 2. Participants

- 8-12 researchers who use Fiji at least weekly (primary group). Optionally 4 who have never used
  Fiji (learnability group; analysed separately, never pooled).
- Record in `participants.csv`: ID, group, tool order, years of Fiji use, scripting experience
  (none / macros / Python or other), operating system, screen size and resolution, input device
  (mouse / trackpad), whether they use assistive technology.
- With these numbers the study detects large differences only. It is a usability study, not an
  A/B test: the qualitative notes are as important as the numbers.

## 3. Session plan (75 min)

| Minutes | Step |
|---|---|
| 0-5 | briefing (participant_briefing.md), consent (consent_form.md), demographics |
| 5-7 | two-minute tour of tool A (session_script.md §2: window layout, where commands live, how to undo) |
| 7-35 | tasks 1-10 in tool A, one card at a time; log each in `task_log.csv` |
| 35-38 | SUS for tool A; two open questions |
| 38-40 | two-minute tour of tool B |
| 40-68 | tasks 1-10 in tool B |
| 68-71 | SUS for tool B; two open questions |
| 71-75 | debrief: "which would you use tomorrow for your own data, and why?"; thanks |

## 4. Measures (what goes into the log)

Per participant × tool × task, one row in `task_log.csv`:

- `completion`: `complete` (end state reached without help), `assist` (reached after a moderator
  hint), `abandoned` (participant gave up, or 5 minutes elapsed; the moderator then shows the
  solution so later tasks are not blocked).
- `time_s`: seconds from the moment the card is read aloud to the end state being reached (or the
  abandon). Stop the clock during a moderator intervention; note the intervention.
- Errors, counted while they happen (one per occurrence): `err_wrong_command` (a command that is
  not on the path to the end state was run), `err_wrong_window` (looked for the result in the
  wrong window, panel or tab), `err_destructive` (an action that changed pixels or deleted data
  and had to be undone or redone), `err_cancelled` (a dialog or sheet opened and cancelled).
- `first_lookup`: where the participant looked first for the command - `menu`, `search` (Fiji's
  Command Finder / search bar, Lumen's palette), `memory` (shortcut or known path), `help`,
  `asked`.
- `actions`: clicks + keys + text entries per the counting rules of USABILITY.md §1.2, from the
  screen recording (Fiji) or the browser-side event log (Lumen); fill in after the session. Leave
  empty if not counted.
- `notes`: free text, including timestamps of anything that surprised the participant.

Per participant × tool, one row in `sus_responses.csv` (q1-q10, 1 = strongly disagree … 5 =
strongly agree) plus the two open answers.

## 5. Analysis (`analyze_results.py`)

```
python analyze_results.py templates/            # reads the three CSVs in that folder
python analyze_results.py data/study-2026-12/ --out report.md
python analyze_results.py --selftest            # checks the scoring code on synthetic data
```

The script needs only the standard library. It reports, per tool:

- SUS: mean, SD, 95 % confidence interval (t-distribution), per-participant scores, and the paired
  difference Lumen - Fiji with its interval. The published average across studies is 68
  ([Sauro, measuringu.com/sus](https://measuringu.com/sus/)); SUS works with small samples but the
  estimate is imprecise, which is why the interval is reported.
- Per task: completion rate without assist, assist rate, abandon rate; median and mean time with
  the interquartile range; mean errors per category.
- Paired comparisons per task and overall: Wilcoxon signed-rank test on time and on total errors
  (exact p for n ≤ 15, normal approximation above), with the matched-pairs rank-biserial
  correlation as the effect size. Report the effect sizes and intervals, not only p-values.
- First-lookup distribution per tool.
- The pass criteria below, each marked met / not met.

Qualitative coding is done by hand: code each think-aloud note into Nielsen's heuristics (the
scheme of USABILITY.md §1.1) and rate severity by frequency across participants; every
severity-3 finding becomes an issue with the task card and the recording timestamp attached.

Pre-register the task list, the measures and this analysis before recruiting (a dated copy of this
folder in the repository history is enough).

## 6. Pass criteria for Lumen (stated in advance, from USABILITY.md §5.2)

1. Completion without assist ≥ 90 % on tasks 1-6 and 8-10.
2. Completion without assist ≥ 75 % on tasks 7 and 9.
3. Mean SUS for Lumen not lower than mean SUS for Fiji for the same participants.
4. No severity-4 finding in the qualitative coding (any task abandoned by a majority of
   participants counts as severity 4).

`analyze_results.py` evaluates 1-3 from the CSVs; 4 is a judgement recorded in the report.

## 7. Known weak points to watch

From USABILITY.md §3.10: closing a modified image without confirmation (M1); Analyze Particles on
an unthresholded image (M2); ⌘W closing the browser tab (M3); looking for Brightness/Contrast
under Image ▸ Adjust (M9); discovering ⌘K; the server-side file browser; Apply vs Run; the number
of inspector sections; expecting display changes to be recorded or undone; wanting a detection
preview before tracking. Participants who rely on the keyboard or a screen reader will hit the
accessibility problems M4-M8 at once; either fix those before the study or recruit sighted mouse
users only and say so in the report.

## 8. Ethics and data

Recordings and logs are pseudonymised by participant ID; the key stays with the moderator and is
destroyed at the end of the study. Participants may stop at any time without giving a reason and
may ask for their data to be deleted up to the point of publication. Adapt `consent_form.md` to
the local ethics process; a usability study of software with volunteer colleagues is usually
exempt from full review, but check.
