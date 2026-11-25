# Public synthetic eval

The project's own retrieval numbers come from one person's private history, so nobody else can check them. This folder is a small **invented** dataset that anyone can rerun through the real code. Every project, message and question in it is made up.

```bash
python3 examples/eval/run_eval.py
```

This needs the project's requirements installed. On first use it downloads the 80 MB embedding model.

From a clean state, the command:

1. Copies the hook, the MCP server and `scripts/` into a temp folder with its own `config.yaml`. Your real config, database and `logs/` are never opened.
2. Creates the real schema from the DDL in `scripts/remember-setup.py`. None of the setup steps run.
3. Ingests `sessions/*.jsonl` with the real `ingest_jsonl.py`. The keyword index is filled by the real trigger, and every message of 50+ characters is embedded by the real model, the same way live sessions are.
4. Runs `scripts/remember_eval.py` on `golden_set.yaml` and `scripts/remember_baseline.py` on the same data.
5. Writes the results, stamped with the git commit and date, to `results/<date>-<commit>/`.

## Results

### On this synthetic set (rerun it yourself)

The one command that reproduces them is the one at the top of this page: `python3 examples/eval/run_eval.py`. The numbers below are from `examples/eval/results/2026-11-24-5505026/eval.json` (the plain-language version is `eval.txt` in the same folder), run on 2026-11-24. That folder's `run.json` says the working tree had uncommitted changes and the retrieval code did not match commit `5505026`, so treat it as a result for the code of that day and rerun to confirm on your checkout. There are 20 questions with an answer and 6 without.

| way of searching | right memory in top 5 (of 20) | ranked first (of 20) | junk on no-answer questions (of 6) | questions that got a machine message back (of 26) |
|---|---|---|---|---|
| keyword only | 13 | 11 | 1 | 2 |
| keyword, weak-match gate off | 13 | 11 | 3 | 2 |
| meaning only | 16 | 13 | 0 | 0 |
| `on_miss` (the default) | 16 | 13 | 1 | 2 |
| `always` | 16 | 13 | 1 | 2 |
| lookup tool (`search_transcripts`) | 13 | 10 | 3 | 0 |

- **The gap is all in the paraphrase questions.** Keyword search finds 0 of the 7 and meaning search finds 3 of 7. All methods find 8 of 8 of the questions that use the answer's own words (`eval.txt`, "By question type").
- **The 2 machine messages are one planted edge case.** The data holds one machine message that starts with a newline, on purpose (id 12 in the build I checked). The hook's check misses it, and it came back for q05 and q20. The lookup tool strips the newline first, so it never returned one. See the table in `GO_DEEPER.md`.
- **Project favoring changed nothing in how many were found.** In `compare_project-bias.txt` the number found is the same with and without it, for every mode. It changed which memories came back for 1 of the 26 questions (q01), and only their order.
- **One question moves a score by about 5 points**, so read the differences between rows, not the exact percentages.

### On the author's own history (words only)

The questions and the database are private, so no question text, memory text or folder is shown and nothing here can be rerun by you. Each result sits in a gitignored folder under `eval/results/` on the author's machine, and each one used the same frozen copy of the database, taken on 2026-11-23, with 38 questions that have an answer and 8 that do not.

| result folder | run date | commit | which path it measures | what it found |
|---|---|---|---|---|
| `2026-11-23-7e8fefa` | 2026-11-23 | `7e8fefa`, retrieval code matched the commit | **The no-project path.** Every question is asked with no project favored (this is how the self-test always asks). The older figure comes from here. | Default mode (`on_miss`): 32 of 38. `always`: 33. Keyword only: 28. |
| `2026-11-24-f5082bd` | 2026-11-24 | `f5082bd`, retrieval code matched the commit | **The path the hook used then for a folder with no mapping:** favoring the catch-all label `oth`. This is the cap study. | Today's cap of 5: 29 of 38. |
| `2026-11-24-2d1805e` | 2026-11-24 | `run.json` records `5505026` with uncommitted changes, so the code did not match that commit | **Both paths, asked side by side** (`--compare project-bias`). | Default mode: 29 of 38 with `oth` favored, 32 of 38 without. Three questions (q06, q30, q34) were found only without it, and none only with it. Keyword only: 26 against 28. `always`: 31 against 33. |

How to read those:

- **Older 32 of 38** was the no-project path. **29 of 38** was what the hook got when it favored `oth` for a folder with no mapping.
- **Unmapped folders now use no project**, so they now match the no-project path (32 of 38 on this database). That change shipped in commit `5505026`.
- **Mapped project folders are measured only on the synthetic data above.** On the author's own history every question was asked with no project or with `oth`, never with a real mapped project, so how much a mapped folder's head start helps or hurts on real history is not measured.
- **The 3-question difference is a small sample.** Three questions were found only without the favoring and none only with it. A sign test on 3 against 0 gives p = 0.25, so luck cannot be ruled out. It is a direction, not a proof.

## The data

There are three made-up projects, matched by `cwd`: `tdp` (a tide-pool survey app), `lgl` (bakery inventory) and `qst` (a static blog generator). They hold 22 sessions and 54 messages, dated September 2025 to May 2026.

The golden set has 20 questions whose answer is in the data and 6 whose answer isn't. Each answerable question lists the session that answers it and is tagged with what it tests:

| tag | what it checks | example |
|---|---|---|
| `exact_words` (8) | the question uses the answer's own words | "how many daily pg_dump copies of Postgres do we keep?" |
| `paraphrase` (7) | the question shares **no** search word with the answer, so only meaning search can find it | "how would a customer find out a pastry could be dangerous for someone with food sensitivities?" (the answer is about allergen labels) |
| `recency` (3) | an older and a newer session both answer; only the newer one counts. In q16 and q17 the *older* one matches more of the question's words (7 of 7 vs 4 of 7, and 5 of 5 vs 2 of 5), so only recency weighting can put the newer one first. q18 is the control: both match 3 of 3. | "what is the photo cap per survey…" |
| `machine_generated` (2) | machine messages (a `<task-notification>`, a subagent `<agent-message>`, command output) match the question's words better than the human answer does | "how does the tidal-chart screen get its tide predictions?" |

Seven machine-generated messages are in the data, and none of them should ever come back. The report counts, for every method, how many questions got one back. One of them starts with a newline on purpose, as an edge case.

The golden format is the one documented at the top of `scripts/remember_eval.py`, plus these optional keys:

- `as_of`: fixes "now" for keyword recency, so results don't drift as the calendar moves.
- `tags`: the behavior a question tests (the table above).
- `superseded`: the older answer in a recency pair. It never counts as a hit.
- `project`: the project the replay searches from.

## What's in a results folder

| file | contents |
|---|---|
| `run.json` | commit, date, whether the retrieval code matches that commit, Python and SQLite versions, database facts, the config values used |
| `eval.json` / `eval.txt` | `remember_eval.py` result, every method plus `mcp` (the `search_transcripts` tool), as JSON and as the plain-language report |
| `replay_questions.jsonl` | for each of the 26 questions, searched from its own project as of `as_of`: what the hook's keyword path would inject (result ids in order, plus a sha256 of the exact text) and the `on_miss` gate's decision |
| `replay_prompts.jsonl` | the same for the user prompts stored in the sessions, each as of its own timestamp |
| `replay.json` | `remember_baseline.py` summaries for both replays |
| `compare_*.json` / `compare_*.txt` | paired comparisons from `scripts/remember_compare.py`, each asking the same questions two ways: `project-bias` (project favored or not), `mcp-machine-filter` (the lookup tool before and after it hid machine messages) and `mcp-gate` (a weak-match gate for the lookup tool, simulated and not shipped). Ids and counts only |

The two `replay_*.jsonl` files are the "before" snapshot for any refactor of the search. The same data and code must give the same ids and the same hashes.

## Limits: where this set is easier than real life

- **It's tiny.** At 54 messages, BM25 word rarity behaves differently than on thousands of messages, and the weak-match bar (`-1.05 × ln(messages)`) works out to about −4.2 here.
- **Messages are clean.** Each one is about a single topic, with no typos, no pasted logs and no long tool output.
- **The same author wrote both sides.** The questions and answers come from one person, and the paraphrases were picked on purpose. Real paraphrases are messier.
- **Small samples.** One question moves a score by about 5 points.

## CI

`tests/ci_test.py` rebuilds this set with `--keyword-only` and checks that keyword, keyword_ungated, mcp and both replays match the newest committed full run. It compares ids, order, outcomes, text hashes and metrics, never raw scores or timings. **Gap:** CI doesn't check meaning, `on_miss` or `always`, because those need the 80 MB model download.

## Other options

- `--keyword-only`: skips the model.
- `--db PATH --golden PATH --config-from config.yaml --out DIR`: scores an existing database file read-only. Its sha256 is checked before and after the run. Use this on a frozen copy of your own database to get private numbers from the same code.
- `--keep`: keeps the temp sandbox so you can inspect it.
