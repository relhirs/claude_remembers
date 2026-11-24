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
