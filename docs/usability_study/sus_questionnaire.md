# System Usability Scale (SUS)

*One copy per participant and per tool. Fill in the tool name before handing it over. The ten
items are Brooke's original (1996) with "system" replaced by "program"; keep the alternating
positive / negative wording, it is part of the instrument.*

Participant number: ________  Program: ☐ Fiji/ImageJ  ☐ Lumen  Date: __________

Please tick one box per statement, thinking about the program you have just used. Answer quickly;
if you cannot decide, tick the middle.

| | Statement | Strongly disagree 1 | 2 | 3 | 4 | Strongly agree 5 |
|---|---|---|---|---|---|---|
| 1 | I think that I would like to use this program frequently. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 2 | I found the program unnecessarily complex. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 3 | I thought the program was easy to use. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 4 | I think that I would need the support of a technical person to be able to use this program. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 5 | I found the various functions in this program were well integrated. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 6 | I thought there was too much inconsistency in this program. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 7 | I would imagine that most people would learn to use this program very quickly. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 8 | I found the program very cumbersome to use. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 9 | I felt very confident using the program. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 10 | I needed to learn a lot of things before I could get going with this program. | ☐ | ☐ | ☐ | ☐ | ☐ |

Two open questions (the moderator writes the answers down verbatim):

1. What would stop you from using this program for your own data?

2. What did you miss?

---

## Scoring (for the analyst, not the participant)

For odd items (1, 3, 5, 7, 9) the item score is *answer − 1*; for even items (2, 4, 6, 8, 10) it
is *5 − answer*. Sum the ten item scores (0-40) and multiply by 2.5 to get the SUS score (0-100).
`analyze_results.py` does this from `templates/sus_responses.csv`. The average across published
studies is 68; above ~80 is usually read as "good", below ~50 as "poor"
([Sauro, measuringu.com/sus](https://measuringu.com/sus/)). With 8-12 participants the 95 %
confidence interval is wide (often ±10 points): report it.

A missing answer is scored as the neutral 3 for that item (Brooke's recommendation); note the
number of imputed answers.
