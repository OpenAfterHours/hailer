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

- **`local` (the default): not sandboxed.** The kernel runs in Hailer's own Python, as you, with your file
  and network access. Hailer closes two gaps around it:
  - *Other programs.* The marimo server Hailer starts requires a random token, handed to marimo on stdin
    (never on a command line) and kept in `.hailer/kernel.json`, so another program on the machine cannot
    send code to the kernel. A marimo server you start yourself with `--no-token` has no such protection.
  - *Secrets in the environment.* The server's environment leaves out every provider's `env_key` and
    `OPENAI_API_KEY`, the variables named in `env_http_headers`, `HAILER_MARIMO_TOKEN`, the names
    `PASSWORD`, `SECRET`, `TOKEN`, `PGPASSWORD` and `MYSQL_PWD`, and every name ending in `_KEY`, `_TOKEN`,
    `_SECRET`, `_PASSWORD`, `_PASSWD`, `_PWD`, `_CREDENTIALS`, `_CONNECTION_STRING` or `APIKEY` (any case).
    `uvx hailer doctor` shows how many were withheld. A notebook that needs one of them (a database
    password, say) gets it through `[kernel] pass_env = ["DB_PASSWORD"]`; naming a provider's key or
    header variable, or `HAILER_MARIMO_TOKEN`, there is a `config` warning.

  Notebook code can still read the OS credential store and every file you can, which is why the `Kernel:`
  line says `not isolated`.
- **`docker`: isolated.** The kernel runs in a container that sees only the notebooks folder (read-write)
  and the data folder (read-only), with no network, none of your environment variables, as a non-root user
  with resource limits (see [Isolated kernel (Docker)](docker.md#isolated-kernel-docker)). Hailer refuses to mount
  a folder that would expose its own files or your credentials (`~/.ssh`, `~/.aws`, `%APPDATA%`, ...), a
  data folder inside the notebooks folder, and a notebooks folder that is a git repository, on every start
  path (see [Which folders can be mounted](docker.md#which-folders-can-be-mounted)).

The instructions forbid destructive file operations and sending data anywhere, but instructions are not an
enforcement boundary: use an endpoint and model you trust, keep the data folder to data the agent may read,
and press Ctrl+C if a turn goes somewhere you did not intend.

**What the docker runtime does not protect against:**

- *Output goes to the model.* Anything notebook code prints or returns is a tool result and is sent to the
  model endpoint, in either runtime.
- *Notebooks are code.* A notebook the container wrote runs on your machine, as you, if it is later opened
  with the local runtime (or with marimo directly). Hailer warns on the first local start after a docker
  kernel used the notebooks folder. Keep the notebooks folder inside your project's git repository (a plain
  subfolder, not a repository of its own), so every change is a diff you can review.
- *Other files in the notebooks folder can run code later.* Notebook code can write anything into the
  notebooks folder, including a `.git` folder (hooks and settings such as `core.fsmonitor` run when git
  or an editor that scans nested repositories, like VS Code, touches it), `.vscode`, `.idea` or
  `.devcontainer` settings, or a `hailer.toml` that makes the folder look like another workspace. Hailer
  refuses existing repositories and nested Hailer workspaces anywhere in the notebooks tree. At stop
  and in `doctor`, it warns about repository, editor and workspace controls in that tree. These checks
  do not prevent an editor from acting on a new file while the kernel is running. Disable automatic
  repository discovery and automatic editor tasks for folders holding untrusted notebooks. Review new
  control files before opening them; Hailer never removes them for you.
- *The token is on the container's command line*, so `docker inspect` shows it. Anyone who can use Docker
  on the machine can already `docker exec` into the container, so hiding it would gain nothing.
- *The image is trusted by name.* Hailer checks its tag and version label, not a signature, and runs
  whatever `[kernel].image` names. The image pins marimo, Polars, fastexcel, DuckDB, altair, plotly and its base
  image, but not their dependencies.
- *It is a choice, not a policy.* Anyone can switch back to `local`. An organisation that must enforce
  isolation should run Hailer itself in a managed virtual machine or dev container.

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
replaces it. `.hailer/kernel.json` records the marimo server Hailer started (runtime, URL, token and, for
docker, the container and network ids and the settings it was started with) and is deleted after successful cleanup; failed Docker cleanup keeps the record for retry; `.hailer/last-kernel.json` notes which runtime last used the notebooks folder. `.hailer/marimo.log`
holds the local server's output, including its signed-in URL, and is emptied at each start. On macOS and
Linux these three files are readable by you only. `.hailer/` is never mounted into a docker kernel and is
kept out of git by its own `.gitignore` containing `*`; Hailer never edits your repository's `.gitignore`.
