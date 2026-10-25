# How this works

**Hooks.** Claude Code lets you set up scripts that fire automatically at certain points, like when you start a session, when you send a message, when Claude's done answering, or when a session ends. This project uses six of those. You never actually run them yourself, they just happen in the background.

**A database.** Everything gets dumped into one SQLite file sitting on your disk. Every message, every session, a summary written at the end of each session, and if you set up an API key, a list of "lessons" pulled out of whatever you worked on. It's really just a file, you can pop it open with any SQLite tool if you want to poke around.

**An MCP connection.** This is how Claude asks the database stuff like "what did we decide about X" or "find anything about Y." You don't need to memorize tool names or anything, just talk to Claude normally and it figures out which tools to use.

## What happens step by step

**When you open a project and type `claude`:** the session start hook checks if there are any new chat logs it hasn't seen, pulls them in, backs up the database, and gives Claude a quick summary of what happened last time plus anything that got left unfinished.

**Every time you send a message:** a hook digs through the database for anything relevant to what you just typed and quietly tosses it into Claude's context before Claude even sees your message.

And this happens on literally every message, not just right after you run `/clear`. The `/clear` recovery thing below is really just for one specific moment, the message right after you clear mid session. Most of the time when you're bringing up old stuff it has nothing to do with `/clear` at all, like you close your terminal on Monday, open a new one Wednesday, and ask "what did we decide about the database schema" and that's a totally new session that never went near `/clear`. If search only ran right after clearing, that Wednesday question would just come back empty since nothing was ever cleared. This per message search is the whole point of the project, remembering stuff across sessions, not just across a single `/clear`. The `/clear` recovery thing is just a smaller bonus feature on top of that.

This part does keyword search first since it's quick. If keyword search doesn't turn up anything good, it falls back to meaning based search (basically comparing what your message is about instead of the exact words), that one takes a bit longer to spin up so it only kicks in when the fast option finds nothing.

**Every time Claude finishes answering:** the whole exchange gets saved to the database right away. It doesn't wait till the session ends. So if your terminal crashes you don't lose the conversation, just the end of session summary.

**When a session ends**, meaning you ran `/exit`, closed the terminal, or hit `/clear`, there's a background script that reads through the whole conversation and asks Claude Haiku to write an actual summary, what you were doing, what got decided, what's still unfinished, what to pick up next time.

If that call fails for whatever reason (no API key, no internet, hit a rate limit) it falls back to a simpler note instead, putting together just the project name, how many messages were in the session, the time range, and a tiny preview of your first message. That fallback note isn't nearly as useful as a real generated summary, but at least it's clearly marked as a fallback and the session still ends up in the database with something attached to it. The alternative would be writing nothing at all, which means that session disappears from your history since nothing ever points back to it. Either way something gets written, just a question of how useful it ends up being.

**When you run `/clear`:** in the old version, running this command only tried to bring back context if you came back within five minutes, and even then it just dumped the last ten raw messages at you. Now there's no time limit at all, so whenever you come back, even days later, your next message gets the real summary from before mixed with a normal search based on whatever you just typed. It only does this once per `/clear` though, so it won't keep repeating itself on every message after.

## Why "lessons" are kind of a big deal

A session summary is tied to one conversation. A lesson is different, it's more like a standalone fact that's supposed to outlive that one session. Like if you spend an hour figuring out some approach causes a bug, that's worth remembering next time you're anywhere near that problem, not just buried somewhere in one session's notes. The summarization step pulls these out automatically whenever there's something that seems worth keeping.

The catch here is it only works off what gets said, not what actually gets run. Claude Code doesn't save the actual output of tool calls, so stuff like code execution results, notebook cell outputs, whatever a command printed, none of that goes into this memory system. Only the plain text of your messages and Claude's replies gets captured. So if Claude runs something and sees a cool result, that result only makes it into memory if Claude actually types it out in the reply. There's a standing instruction in the global Claude Code config telling Claude to do exactly that, say real findings out loud instead of just quietly moving on, because otherwise a real result can happen and just never get remembered.

## Getting the most out of it

- **Set up the API key.** Without it the session summaries are pretty plain and mechanical. With it they're more useful, providing real reasoning about what happened instead of just a message count.
- **Add your real projects.** Run `python3 scripts/add-project.py` from the root folder for each project you're working in. Without doing this everything just gets dumped into one generic bucket instead of getting its own tracked history and project specific context.
- **Running `python3 scripts/batch_embed.py` every once in a while is still worth it.** New messages, including imported ones, get embedded automatically as they come in now. This script is just a catch up pass for anything that slipped through, like if semantic search was turned off in `config.yaml` during an import and you've since turned it back on. It's pretty quick, like a few seconds for thousands of messages.
- **Just talk normally.** You don't need slash commands or any special phrasing. Just ask "what did we decide about the database schema" the same way you'd ask a person.

## If you want to poke around directly

Anything in `scripts/` that starts with `remember_` is just a plain script you can run yourself, like `python3 scripts/remember_search.py`, `remember_health.py`, `remember_history.py`, stuff like that. Useful if you just want to check something without going through a whole conversation.

Day to day you don't need any of this, the hooks and the four `/remember-*` slash commands cover normal use. These are for when you want something faster than a conversation, a maintenance check, or to change the setup itself.

**Quick lookups, no conversation needed.** Same database the MCP tools query, just from your terminal:
- `python3 scripts/remember_search.py "<keyword>"` -- raw keyword search. Flags: `--project`/`-p` to filter to one project, `--limit`/`-l` for how many results (default 10), `--no-semantic` to skip the meaning based fallback and keep it fast.
- `python3 scripts/remember_query.py "<question>"` -- same idea, phrased as a natural language question instead of keywords. Same `--project`, `--limit`, `--no-semantic` flags.
- `python3 scripts/remember_decide.py "<number or keyword>"` -- fast lookup into just the decisions table, skip everything else.
- `python3 scripts/remember_history.py` -- session timeline. `--project`/`-p` to filter, `--count`/`-c` for how many sessions back (default 10).
- `python3 scripts/remember_recap.py` -- a progress report for a stretch of time. `--today`, `--week`, or `--days <n>` for the range, `--project`/`-p` to scope it to one project.
- `python3 scripts/remember_topics.py [tag]` -- browse sessions by tag, or list all tags if you leave `[tag]` off. `--project` to scope it.

**Maintenance, worth running every so often rather than every session:**
- `python3 scripts/remember_health.py` -- the full 9 point dashboard: database integrity, backup status, embedding coverage, hook and MCP registration, config validity. More detail than `/remember-status`. Add `--json` if you want to pipe the output somewhere instead of reading the table.
- `python3 scripts/batch_embed.py` -- catches up anything that didn't get embedded for meaning based search. No flags, just run it. Takes a few seconds even for thousands of messages.
- `python3 scripts/remember_tag_review.py --generate` -- builds a spreadsheet of suggested tags for review (tags are keyword-suggested, meant to be checked by a human, not trusted blindly forever). Review and edit it, then `--update --map <the xlsx file>` writes your changes back. `--all` includes sessions that already have tags, `--project` scopes it, `--output` picks where the spreadsheet goes.

**Changing the setup itself.** These are plain scripts instead of slash commands on purpose, since they touch `config.yaml` and are infrequent, higher stakes actions compared to normal chat:
- `python3 scripts/add-project.py` -- add a new project: folder, `CLAUDE.md`, config entry, database registration, MCP registration, all in one pass. (All three import scripts reuse this exact same logic whenever you type a brand-new folder name instead of an existing one -- `import_claude_ai.py --folder <name>` collapses it into the single-conversation import command directly, all three scripts' bulk `folder` spreadsheet column triggers it the same way since those cover many conversations at once.)
- `python3 scripts/remove-project.py` -- the reverse. Stops tracking a project (config, database registration, MCP registration) but keeps your actual conversation history by default. Deleting that history is a separate, explicitly confirmed, irreversible step inside the same script, not something that happens automatically.

**Getting data out:**
- `python3 scripts/remember_export.py` -- dumps memory data to plain text files. `--profile` for your facts/preferences, `--decisions` for the decisions table, `--session <id>` for one full transcript, `--search "<query>"` for a set of search results, `--recap-week` for the week's recap.
- `python3 scripts/remember_digest.py` -- only relevant if you've set up the email digest feature in `config.yaml`. Triggers one on demand instead of waiting for its cron schedule: `--daily` for the compact standup, `--project <prefix>` for a project deep dive, `--days <n>` for a custom lookback window, `--dry-run` to see the email on your screen instead of actually sending it, `--test` for a short test send.

What you won't find a reason to run yourself: `write_exchange.py`, `ingest_jsonl.py`, `startup_check.py`, `summarize_session.py`. These are the internals the hooks already trigger automatically at the right moments. Running them by hand wouldn't break anything, they're all deduplicated and safe to re-run, there's just nothing to gain from doing it out of band.

## Bringing in history from other AI tools

If you were talking to Claude, ChatGPT, or Gemini before setting this up, you can pull that history in too. None of it is automatic though, this project only auto captures Claude Code itself. Each platform still needs a manual export and then a one time import you run from the terminal, exactly like before. Nothing about that part changed.

What did change is how the import decides which project a conversation belongs to: now it doesn't decide anything. There used to be a hardcoded keyword guess, then later a meaning-based similarity guess with a confidence score -- both are gone. Instead, each conversation gets a short, free "about" blurb so you can tell at a glance what it actually was (claude.ai conversations sometimes already have their own generated summary, which gets reused as-is; everything else gets a blurb built from the first message, no API calls, no guessing) plus a few suggested tags from simple keyword matching. You read that, then type the destination folder yourself: an existing one, or a brand-new name, which gets created automatically the moment you import. Leave it blank and it just goes into the default project. Nothing ever gets filed without you typing the folder.

**Claude:**
- **Single conversation** (e.g. the "AI Chat Exporter" Chrome extension's export of one open chat):
  1. Run `python3 scripts/import_claude_ai.py <path-to-json>` with no `--folder` flag. This doesn't import anything -- it prints the title, the about blurb, suggested tags, and the list of existing projects (folder name, prefix, label) so you can decide.
  2. Run it again with `--folder <folder-name>` to actually import: `python3 scripts/import_claude_ai.py <path-to-json> --folder <folder-name>`. Type an existing folder name from the list, or a brand-new one -- either way this creates the project on the fly if it doesn't exist (folder, `CLAUDE.md`, config entry, database registration, MCP registration, all of it) and imports into it in the same command.
- **Full account export** (Settings, Privacy, Export Data -- this bundles every conversation you've ever had into one `conversations.json`, not one file per chat):
  1. In Claude, click your initials (bottom-left), Settings, Privacy tab, Export Data, confirm the request. An email arrives with a download link (expires in 24 hours).
  2. That link downloads a `manifest.json`, not the conversations themselves -- it just lists several separate zips (`light_metadata`, `projects`, `memories`, `feedback`, `design_chats`, `frames`, `artifact_data`, `conversations`), each with its own `export_url`. Find the entry with `"category": "conversations"` (filename `conversations-000.zip`), download and extract that one specifically -- the others aren't conversation history and you don't need them for this.
  3. Extracting `conversations-000.zip` gives you `conversations.json`.
  4. Scan it: `python3 scripts/import_claude_ai.py --scan <path-to-conversations.json>`. This builds `claude_ai_import_map.xlsx` with a `title`/`about`/`tags` column per conversation plus an empty `folder` column -- nothing gets imported yet.
  5. Open the xlsx, read the title/about for each row, and type a folder name into the yellow `folder` column for anything you want to keep (existing name, or a new one -- use the same new name across multiple rows to group them into one project). Leave `folder` blank to use the default project. Delete rows you don't want.
  6. Preview: `python3 scripts/import_claude_ai.py --import <path> --map claude_ai_import_map.xlsx --dry-run`
  7. Looks good? Run the exact same command again minus `--dry-run`.

**ChatGPT:**
1. In ChatGPT go to Settings, Data Controls, Export. You'll get an email with a download link, ready within a day or two.
2. Extract the zip somewhere, like `imports/chatgpt-export/`.
3. Scan it first, this just builds a spreadsheet, nothing actually gets imported yet: `python3 scripts/import_chatgpt.py --scan imports/chatgpt-export/`
4. Open up the `chatgpt_import_map.xlsx` it makes. Each row has `title`/`about` columns so you can tell what the conversation actually was, a `tags` column (edit freely), and an empty `folder` column. Type a folder name for anything you want to keep -- an existing one (see the Project Reference tab) or a brand-new name, which gets created automatically on import. Use the same new name across multiple rows to group those conversations into one project together. Leave `folder` blank to use the default project. Delete any rows you don't want brought in.
5. Preview the import without touching the database: `python3 scripts/import_chatgpt.py --import imports/chatgpt-export/ --map chatgpt_import_map.xlsx --dry-run`
6. Looks good? Run the exact same command again minus `--dry-run`.

**Gemini:**
1. Go to Google Takeout, select only "My Activity" then "Gemini Apps," export it, wait for the email, download the zip.
2. Extract it, like to `imports/gemini-export/`.
3. Scan: `python3 scripts/import_gemini.py --scan imports/gemini-export/`
4. Go through `gemini_import_map.xlsx` the same way you did for ChatGPT -- `title`/`about` for context, `tags` to edit, and an empty `folder` column to type your decision into.
5. Dry run then the real run, same two commands as ChatGPT, just swap in `import_gemini.py`.

One weird thing about Gemini specifically, its export isn't split into sessions already, it's just a stream of individual exchanges. The import groups anything within 30 minutes of each other into one session automatically so you don't end up with thousands of one message "sessions."

All three of these also embed everything as it gets imported now, the same embedding step that happens automatically for live Claude Code sessions. That used to only happen later, whenever you got around to running `batch_embed.py`. Imported messages don't need that catch up step anymore, they're already embedded and searchable the moment the import finishes.

## What this doesn't do

- It can't see what happens inside a tool call, only what gets written in a text reply. This isn't really something that can be fixed cleanly either, tool output like code execution, notebook cells and command output can be huge and messy, and trying to capture all of it would just flood the database with junk nobody would ever search through anyway. The real fix is that standing rule mentioned above, telling Claude to say real findings out loud in its own reply instead of letting them sit quietly in a tool's output. That's more of a workaround than an actual fix though, if Claude runs something and doesn't say what it found, that finding just isn't getting remembered.
- It's single user, single machine by design. No multi user support, no syncing between computers built in.
- The first meaning based search in a session takes a few seconds to warm up since it's loading a small local model. After that it's fast. That's why regular keyword search runs first by default.