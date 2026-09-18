# Project context for Hailer

This folder holds the context, skills and prompts that Hailer gives to the AI agent for
**this** project. Commit it alongside the notebook so the whole team gets the same setup.

```
.config/hailer/
├── hailer.toml   (optional: the project config can live here instead of the repo root)
├── context/      always-on: every *.md here is sent to the model at the start of each session
├── skills/       on-demand: <name>/SKILL.md (+ reference/, scripts/) loaded only when needed
└── prompts/      reusable prompts: /prompt <name> [args]
```

## context/

Short Markdown files with facts the agent should always know: what the datasets are, column
meanings, conventions, house style. Files are loaded in filename order (prefix with `00-`,
`10-` to control it) and concatenated up to the `max_context_bytes` limit in `hailer.toml`.

**Everything in `context/` is sent to the configured model endpoint every session.** Do not
put credentials, customer data or anything you would not paste into a chat with the model.
Hailer warns at startup if a file looks like it contains a key or token.

## skills/

Skills use the Agent Skills format that several coding agents share: a folder with a `SKILL.md`
whose frontmatter has `name` and `description`, plus optional `reference/` and `scripts/`.
Hailer puts only the name and description in the agent's instructions; the body is loaded
when the task matches (the agent calls `load_skill`) or when you type `/skill <name>`.

Write the `description` as the trigger: *when* to use the skill and *what* it produces.

## prompts/

Plain Markdown prompts. `{{args}}` is replaced with whatever follows the name:

```
You > /prompt first-look customers.csv
```

## Reloading

`/reload` re-reads this folder without restarting the session. `/context` shows what is loaded.
