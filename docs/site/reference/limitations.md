# Known limitations {#known-limitations}

- `marimo._code_mode` is a private API; Hailer pins marimo 0.24.2 and may need changes for other versions.
- The notebook must be open in a browser; a headless server without a tab has nothing to execute against.
  Switching notebooks opens a new tab; the old one stays open until you close it or `/notebook close` it.
  When the agent switches notebooks and the browser is slow to load, the CLI may open the same notebook a
  second time after the turn (it opens the URL once more when it still sees no kernel session); the extra
  tab is harmless, close it.
- If you press Ctrl+C while the agent is in the middle of `notebook_create` / `notebook_open`, that tool call
  may still finish and switch the active notebook a moment later. Hailer re-reads the state after every
  turn and before `/notebook` and `/status`, so the next command shows the right notebook; a `/notebook
  new` typed in that instant can still be overtaken by the late switch.
- Both `uvx hailer` and `uvx hailer notebook` start or reuse this workspace's marimo server and stop only
  a server they started. An explicitly pinned server must already be running.
- Every request carries about 10 KB of instructions and 5 KB of tool definitions (measured with an empty
  project context) plus the conversation; an endpoint with a strict request-size limit needs room for that.
- The endpoint must support function calling in the standard OpenAI shape. An endpoint that streams tool
  calls in a non-standard way may lose their arguments; `stream = false` on the provider avoids that.
- After Ctrl+C during a long `marimo_execute`, the turn ends at once but the code keeps running in the
  kernel until it finishes, or until you interrupt or restart the kernel from the notebook.
- The docker kernel needs Docker, which is not always an option: Docker Desktop is free for personal use
  and small businesses, but larger organisations need a paid subscription (check Docker's current terms),
  and managed machines often block Docker Desktop, WSL2 or Hyper-V. That is why `local` stays the default
  and fully supported.
- Reading large files through a Windows bind mount is slower than reading them from a local folder; how
  much slower for large Parquet files has not been measured yet.
- The release workflow builds the `linux/arm64` kernel image (Apple Silicon, ARM Linux), but the CI
  integration test only covers amd64. Podman is not supported.
- The docker kernel has only the packages in the image (marimo, Polars, fastexcel, DuckDB, altair, plotly); anything
  else needs an image of your own, built `FROM` Hailer's (so it keeps the version label) and named in
  `[kernel].image`. There is one kernel per workspace, and
  changing `[kernel]` settings while one is kept running needs `uvx hailer kernel stop` first.
