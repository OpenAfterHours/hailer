# Context, skills and prompts {#project-context-skills-and-prompts}

`.config/hailer/` holds what the agent knows about *this* project. Commit it with the notebook. Inside
another project these folders are in `.hailer/` instead (see [Inside another project](project.md#hailer-folder)),
and the agent can also read the project's own code and docs ([project files](project.md#project-files)).

```
.config/hailer/
├── context/      always-on: every *.md here is sent to the model at the start of each session
├── skills/       on-demand: <name>/SKILL.md (+ reference/, scripts/) in the Agent Skills format
└── prompts/      reusable prompts: /prompt <name> [args]   ({{args}} is substituted)
```

- **context/**: short Markdown files with column meanings, conventions and house style, loaded in filename
  order and concatenated up to `max_context_bytes`. Hailer warns at startup if a file looks like it contains
  a key or token.
- **skills/**: only the `name` and `description` from each `SKILL.md` frontmatter go into the agent's
  instructions; the body and bundled files are loaded when the task matches (the agent calls `load_skill`
  and `read_skill_file`) or when you type `/skill <name> [message]`.
- **prompts/**: `/prompt first-look customers.csv` sends `prompts/first-look.md` with `{{args}}` replaced.
- `/context` lists what is loaded; `/reload` re-reads the folder and applies from your next message, in the
  same conversation.

Everything in `context/`, and any skill you load, is sent to the configured model endpoint. Keep
credentials, customer names and row-level data out of it.

For a support chat, a context file is the place to say who the users are and how to answer them, for example
*The people asking are analysts: explain how figures are derived in business terms and name the source file*.
