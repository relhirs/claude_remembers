# claude_remembers

Claude Code just forgets everything the second you close the terminal, run `/clear`, or start a new project. This gives it real memory instead. Every conversation gets saved to a local database on your own machine, and Claude can dig through all of it, stuff like past decisions, what you already tried, what broke and why, so you're not re-explaining yourself every single time.

It runs as a bunch of hooks that fire automatically in the background, plus an MCP connection Claude uses to look stuff up. You don't have to type anything special to use it, it just works on its own.

For the full rundown of how it works under the hood and how to get the most out of it, check out [GO_DEEPER.md](GO_DEEPER.md).

## Why this is better 

The default way Claude Code tries to "remember" stuff is through a file called MEMORY.md that Claude writes to on its own while you work, and the catch is it only actually loads the first 200 lines of that file, or the first 25KB, whichever one it hits first, at the start of every session. So in practice as the thing grows, older stuff just stops getting loaded automatically. It's still technically sitting on disk in separate topic files, but Claude has to decide to go open those itself since there's no real search across them; it's more just file reading than anything else. And it's not like it's static either, Claude keeps writing to it as you go, the real issue is just that whatever's past that 200 line or 25KB mark basically disappears from what gets pulled into context each session.

This project gets around that by not trying to cram everything into one file at all. Instead of a size capped document Claude reads top to bottom every time, everything goes into a real database that can grow as big as it needs to, and instead of Claude reading the whole thing every time, a hook just pulls out the specific stuff that's truly relevant to what you just asked and drops that into context. So you're not stuck choosing what makes the cut into 200 lines - everything gets kept, and the search figures out what's worth surfacing at any given moment instead of you doing that curation by hand.

## Installing it

You need Python 3.10+, Claude Code, and pip.

```bash
git clone <your repo url> claude_remembers
cd claude_remembers
python3 scripts/remember-setup.py
```

The setup script walks you through everything: where you want the database to live, which folders count as "projects," and whether you want semantic search (meaning based search on top of the regular keyword stuff). It makes the database, registers the hooks with Claude Code, and sets up the MCP connection. It'll take a few minutes.

It'll also ask if you want AI generated session summaries instead of the bare bones auto notes, and for that you'll need an Anthropic API key. You can put it in `config.yaml` under `summarization.api_key`, or just set `ANTHROPIC_API_KEY` in your shell. This part's optional, without it you still get working memory, just with simpler notes.

After setup you just use Claude Code like normal. Open a project folder, run `claude`, and talk to it. The memory's already running in the background at that point.

### Configuring it manually

`config.yaml` (the file everything actually reads from) is gitignored, so it never ships in the repo — it only exists once something generates it. `config.yaml.example` is the documented reference copy that does ship, with a comment above every option explaining what it does. You've got two ways to get from one to the other:

- **Run the setup script** (`python3 scripts/remember-setup.py`) and answer its prompts — it writes `config.yaml` for you, interactively, based on your answers.
- **Copy and edit it yourself**: `cp config.yaml.example config.yaml`, then open `config.yaml` and fill in your own values (root path, projects, etc.) by hand, using the comments in the example as your guide.

Either way you end up with the same file; the setup script is just the interactive version of editing the example yourself.

## Security

Everything just lives on your own disk. No cloud service, no third party server, nothing getting uploaded anywhere. The only outside call this ever makes is to Anthropic's API, and only for the optional summarization step, and even then it's just the text of your own conversations going there, same as it already does every time you talk to Claude anyway. Your `config.yaml` (which can have an API key in it) is excluded from git by default so it won't accidentally end up in a repo somewhere.

## Credit

This is a heavily restructured version of [claude-brain](https://github.com/mikeadolan/claude-brain) by Mike Dolan. The original idea of giving Claude Code actual persistent local memory is his. Most of what's here now, the `/clear` recovery logic, the search, the summarization pipeline, has been rebuilt from that starting point.