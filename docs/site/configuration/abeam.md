# Running inside abeam {#running-inside-abeam}

[abeam](https://github.com/OpenAfterHours/abeam) runs coding-agent CLIs in a terminal pane beside git status,
a file viewer and a shell. It starts `hailer` from your `PATH`, so install Hailer as a tool first:

```bash
uv tool install hailer
abeam +hailer
```

abeam forwards every argument, so `abeam +hailer notebook --no-browser` or `abeam +hailer --new` behave as
they do in a terminal. abeam starts Hailer in a git worktree; commit `hailer.toml` (and your notebooks) so
each worktree is a workspace of its own, since Hailer uses the nearest folder holding `hailer.toml` or
`pyproject.toml`. Each workspace then gets its own marimo server, `.hailer/` stays out of the git pane, and a
pasted block of several lines arrives as one message.
