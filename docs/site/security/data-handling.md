# Data handling and security {#security}

**Sent to the configured model endpoint:** your messages; Hailer's system prompt plus everything in
`.config/hailer/context`; the name and description of each skill, and a skill's full body once loaded; every
tool call argument (that is, the Python the agent writes) and the tool results, truncated to
`max_tool_output_chars`; and pages fetched from allowed domains. The instructions name the workspace,
notebooks and data folders by path (with the docker kernel, the container's `/work` paths). With the `openai` provider this goes to OpenAI; with a custom provider,
to the `base_url` you configured, and nowhere else: when older turns are summarised, the same endpoint and
model write the summary.

**Not sent:** your data files or any dataframe, unless code explicitly prints or returns it (the agent is
instructed to inspect schemas, samples and aggregates and to keep outputs compact); API keys or other
secrets (the key stays inside the Hailer process and is masked in logs).

**What the agent can do:** it has Hailer's eleven tools and nothing else: no shell, no file tools, no
tools from other software on your machine. Nothing asks for approval before a tool runs. The tool that
matters is `marimo_execute`: it runs the Python the model writes in the notebook kernel. That is what
makes the analysis possible. What the code can reach depends on the kernel runtime:

- **`docker` (the default): isolated.** The kernel runs in a container with its own copy of your notebooks
  and the data folder (read-only), with no network, none of your environment variables, as a non-root user
  with resource limits (see [Isolated kernel (Docker)](docker.md#isolated-kernel-docker)). It cannot write to
  your machine: Hailer copies notebooks back, and only marimo notebooks with plain names (see
  [Notebook copies](docker.md#notebook-copies)). Hailer refuses to mount a data folder that would expose its
  own files or your credentials (`~/.ssh`, `~/.aws`, `%APPDATA%`, ...), on every start path (see
  [Which data folder can be mounted](docker.md#which-data-folder-can-be-mounted)). Without a usable Docker a
  start stops; it never falls back to running notebook code on your machine.
- **`unsafe-local` (only if you write it): not sandboxed.** The kernel runs in Hailer's own Python, as you,
  with your file and network access. Hailer closes two gaps around it:
  - *Other programs.* The marimo server Hailer starts requires a random token, handed to marimo on stdin
    (never on a command line) and kept in memory by the session that started it, so another program on the
    machine cannot send code to the kernel. Hailer never uses a marimo server it did not start.
  - *Secrets in the environment.* The server's environment leaves out every provider's `env_key` and
    `OPENAI_API_KEY`, the variables named in `env_http_headers`, the names
    `PASSWORD`, `SECRET`, `TOKEN`, `PGPASSWORD` and `MYSQL_PWD`, and every name ending in `_KEY`, `_TOKEN`,
    `_SECRET`, `_PASSWORD`, `_PASSWD`, `_PWD`, `_CREDENTIALS`, `_CONNECTION_STRING` or `APIKEY` (any case).
    `uvx hailer doctor` shows how many were withheld. No setting lets one through.

  Notebook code can still read the OS credential store and every file you can, which is why the `Kernel:`
  line says `not isolated`, in a warning colour.

The instructions forbid destructive file operations and sending data anywhere, but instructions are not an
enforcement boundary: use an endpoint and model you trust, keep the data folder to data the agent may read,
and press Ctrl+C if a turn goes somewhere you did not intend.

**What the docker runtime does not protect against:**

- *Outputs render in your browser, which is online.* The kernel's "no network" does not cover the browser:
  an HTML or image output notebook code produces can contact the internet when it is displayed, so a
  notebook could send data out that way. Treat outputs of untrusted code accordingly.
- *Output goes to the model.* Anything notebook code prints or returns is a tool result and is sent to the
  model endpoint, in either runtime.
- *Notebooks are code.* A notebook the container wrote is copied back to your notebooks folder, and it
  runs on your machine, as you, if it is later opened with the unsafe-local runtime (or with marimo directly,
  or imported by a script: marimo runs a notebook's setup cell on import). After a docker kernel wrote
  notebooks here, an unsafe-local start asks before it runs them (default No; without a terminal to ask in,
  it refuses and says how to go on). Keep the notebooks folder inside your project's git
  repository, so every change is a diff you can review. Only notebooks come back: git and editor settings,
  test-runner hooks (`conftest.py`, `test_*.py`), Python start-up hooks and other files notebook code writes
  stay in the container (see [Notebook copies](docker.md#notebook-copies)). A notebook named like a module
  your own scripts import (`pandas.py`, say) could still shadow it when you run Python in that folder.
- *The token reaches marimo in a file, not on the command line.* Hailer writes it to a folder under
  `.hailer` that only you can open, mounts that one file read-only into the container, and deletes it once
  the kernel answers, so `docker inspect` and the container's process list do not show it. Anyone who can
  use Docker on the machine can still `docker exec` into the container, and notebook code has the kernel's
  own powers.
- *The image is trusted by name.* Hailer checks its tag and kernel contract label, not a signature, and runs
  whatever `[kernel].image` names. The image pins marimo, Polars, fastexcel, DuckDB, altair, plotly and its base
  image, but not their dependencies.
- *It is a default, not a policy.* Anyone can write `runtime = "unsafe-local"`. An organisation that must
  enforce isolation should run Hailer itself in a managed virtual machine or dev container.

**Links and tokens:** the notebook link Hailer opens in your browser, and every link the CLI prints
(`/notebook`, `hailer status`, `doctor`), carries `access_token=<token>`, which signs the browser in. It
works like a password for the kernel while the server runs, so do not paste it anywhere. URLs in tool results
leave the token out, because tool results go to the model endpoint; when the agent asks you to open a
notebook, run `/notebook` for the signed-in link.

**Tracing:** Hailer's agent library (LangChain) can send full conversation traces to the LangSmith service
when variables such as `LANGSMITH_TRACING` are set in the environment. Hailer switches that off for its own
process at startup, so a variable left over from another project cannot send your prompts and results to a
third party. Set `HAILER_TRACING=1` if you do want the tracing variables in your environment to apply.

**On disk:** the conversation (your messages, the agent's replies, tool calls and their truncated results)
is stored unencrypted in `.hailer/threads.sqlite` inside the workspace until `/new` or `hailer --new`
replaces it. While an unsafe-local kernel runs, `.hailer/marimo-<pid>.log` holds its output, including its
signed-in URL; it is deleted when the kernel stops, also after a failed start. The kernel's token is kept
in memory only. A docker kernel's token file exists only while the kernel starts, and
`.hailer/owner-<id>.lock` marks a running docker session (it holds no secret). `.hailer/sandbox-wrote-notebooks` marks notebooks a docker kernel
copied back that you have not yet agreed to run unsafe-local, and `.hailer/notebook-backups/` keeps your versions of notebooks a docker kernel's copy
replaced (see [Notebook copies](docker.md#notebook-copies)). On macOS and Linux these files are readable by you only. `.hailer/` is never mounted into a docker kernel and is
kept out of git by its own `.gitignore` containing `*`; Hailer never edits your repository's `.gitignore`.
