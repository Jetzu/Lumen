# Moderator script

*What to say and do, in order. Italics are instructions to the moderator; quoted text is read
aloud. Keep the two tours equal: name the same four things in each tool, in the same order, in
about two minutes, and do not demonstrate a task.*

## 1. Before the participant arrives (10 min)

- Study laptop: `study/` folder on the Desktop with the five materials of `task_sheets.md` §0 and
  an empty `study/out/`.
- Fiji: fresh `IJ_Prefs.txt`, no images open, Recorder closed (you open it in the tour).
- Lumen: server started (`python -m lumen`), browser window at native resolution, empty
  `~/.lumen`, no images open, Script & Recorder ▸ Clear.
- Screen and audio recording armed; clock or stopwatch; the printed cards in order; the log sheet
  (`templates/task_log.csv` or a printed copy of its columns); two SUS sheets with the tool names
  filled in.
- Look up the participant's tool order in `participants.csv` (odd IDs Fiji first, even IDs Lumen
  first).

## 2. Welcome, briefing, consent (5 min)

*Hand over or read `participant_briefing.md`. Answer questions. Hand over `consent_form.md`; wait
for the signature. Fill the demographics row of `participants.csv`.*

"Before we start: how many years have you used Fiji? Do you write macros or scripts? Which
operating system and screen do you normally work on? Mouse or trackpad? Do you use any assistive
technology, like a screen reader or magnifier?"

## 3. Tour of tool A (2 min) - start the recording first

**If tool A is Fiji:**

"This is Fiji. *(1)* The small window with the toolbar is the main window; every image, dialog and
table opens in its own window. *(2)* Commands are in the menus here - File, Edit, Image, Process,
Analyze, Plugins - and there is a search field in the main window. *(3)* Edit ▸ Undo undoes the
last image operation. *(4)* I am opening the macro Recorder now *(Plugins ▸ Macros ▸ Record)* and
leaving it open in the corner; it writes down the commands you use. That is all - the rest is up
to you."

**If tool A is Lumen:**

"This is Lumen. *(1)* Everything is in this one window: images are tabs at the top, tools are on
the left, and the panel on the right holds things like display settings, results and history.
*(2)* Commands are in the menus here - File, Edit, Image, Process, Analyze, Plugins - and there is
a search field in the top bar. *(3)* Edit ▸ Undo undoes the last image operation. *(4)* The
recorder in the right panel, under *Script & Recorder*, writes down the commands you use; it is
always on. That is all - the rest is up to you."

## 4. Tasks 1-10 in tool A (28 min)

*For each card:* "Task N." *Read the Goal aloud. Start the clock when you finish reading. Log
`first_lookup` as soon as the participant acts. Count errors as they happen. Stop the clock at the
end state (check it against the card) or at the abandon.*

Prompts, used sparingly:

- Silence longer than ~20 s: "What are you thinking?"
- "What does this do?": "What do you think it does?"
- "Is this right?": "What would you check?"
- At 4 minutes: "One more minute on this one, then we move on."
- At 5 minutes: "Let's leave this one. Here is one way to do it: …" *(show the shortest path,
  log `abandoned`, then restore the state the next task needs)*.
- If an assist is given before five minutes because the participant explicitly asks for help
  twice: log `assist` and note what was said.

*Between tasks, say nothing about how it went. If the participant asks "was that right?", say
"we'll talk about it at the end".*

## 5. SUS and open questions for tool A (3 min)

"Please fill in this sheet about the program you just used. Answer quickly; if you cannot decide,
tick the middle." *Then:* "Two questions. What would stop you from using this program for your
own data?" … "What did you miss?" *Write both answers verbatim into `sus_responses.csv`.*

## 6. Tour of tool B (2 min), tasks 1-10 in tool B (28 min), SUS for tool B (3 min)

*Repeat sections 3-5 with the other tool. Reset nothing in between except closing the first tool.*

## 7. Debrief (4 min)

"Last question: if you had to analyse your own data tomorrow, which of the two would you use, and
why?" *Note the answer.* "Is there anything you expected to be asked and were not?"

"Thank you. The recording is stored under your participant number; if you want it deleted, e-mail
me the number."

*Stop the recording. Save the log. Reset both tools and `study/out/` for the next participant.*

## 8. After the session (15 min)

- Transfer the paper log and SUS answers into the CSVs; double-check the participant number and
  tool order.
- From the screen recording, count `actions` for each task (optional; the counting rules are in
  USABILITY.md §1.2) and add the timestamps of notable moments to `notes`.
- Run `python analyze_results.py <folder>` to see that the rows parse; fix typos in `completion`
  and `first_lookup` values while the session is fresh.
