# Inside another project {#inside-another-project}

Hailer can live inside an existing project, a uv workspace monorepo included, so the chat can answer an
analyst's questions about the data *and* about the product it comes from. Three things make that work:
everything of Hailer's sits in one `.hailer/` folder, the agent can read the project's code and docs, and data
folders anywhere on the machine can be added to the kernel.

## Add Hailer to the project {#add-hailer}

From the project's root (the folder with the root `pyproject.toml`):

```bash
uv add --dev hailer
uv run hailer init
uv run hailer login openai
uv run hailer
```

`uv run hailer` works from any folder of the repository, including a workspace member such as
`packages/billing/`: Hailer looks for its config up to the repository root (the folder with `.git`), and
without one it uses the uv workspace root (the `pyproject.toml` with `[tool.uv.workspace]`).

Adding Hailer as a dependency resolves it together with the project. Hailer pins `marimo` exactly
(`==0.24.2`) and keeps `ruff`, `ty` and the LangChain packages in narrow ranges, so a project that pins other
versions of those cannot install it alongside them. Run it with `uvx hailer` instead: it reads the same
`.hailer/` folder without joining the project's environment. Notebook code never runs in the project's
environment either way: the docker kernel has its own, and notebooks cannot import the project's code.

## The `.hailer/` folder {#hailer-folder}

`init` in a folder with a `pyproject.toml` or `.git` puts everything in `.hailer/` (`--layout project` does
it anywhere; `--layout standalone` keeps the separate `hailer.toml`, `notebooks/`, `data/` and
`.config/hailer/` layout):

```
.hailer/
├── hailer.toml   the settings: commit it
├── notebooks/    the marimo notebooks: commit them
├── context/      always-on context, see Context, skills and prompts
├── skills/       on-demand skills
├── prompts/      reusable prompts
├── data/         the default data folder (ignored by git)
├── local.toml    this machine's settings (ignored by git)
└── state/        conversations, logs and backups of this machine (ignored by git)
```

`init` writes `.hailer/.gitignore` for the last three. Paths in `.hailer/hailer.toml` are relative to the
project root, not to `.hailer/`.

**`.hailer/local.toml`** holds this machine's settings on top of the project's: anything `hailer.toml` can
say, merged table by table, and the environment variables still win over both. It is where each person's
data folder paths go (see below), and it can also hold a personal provider or `[project] enabled = false`.
A standalone workspace reads `.hailer/local.toml` too.

## Let the agent read the project {#project-files}

With a `[project]` section (the project layout's `hailer.toml` has one) the agent gets three read-only tools:
`project_files` lists the project's code, docs and configuration, `project_search` finds lines by regular
expression and `project_read` reads part of a file with line numbers. Ask, for example, *How is net revenue
calculated?* or *Which orders does the churn report leave out?* and the agent searches the code, reads the
relevant lines and answers with file and line references. What it learns also guides the notebooks it writes,
but notebook code cannot import or open these files: the agent re-creates the logic in the notebook.

```toml
[project]
root    = "."                        # the folder to read (default: the workspace)
include = ["**/*.py", "**/*.sql", "docs/**"]   # only these (default: code, docs and config files)
exclude = ["tests/fixtures/**"]      # never these
```

What the agent can read:

- In a git repository, the files git lists: tracked ones, and new ones git does not ignore. Elsewhere, the
  folder's files, without virtual environments.
- Never: `.git`, `node_modules`, virtual environments and caches; Hailer's state, `local.toml`, the notebooks
  folder and every data folder; files named like credentials (`.env`, `*.pem`, `*.key`, `id_rsa`,
  `credentials.json`, `secrets.yaml`, `*.tfstate`, ...; `.env.example` is fine); links that lead out of the
  folder; files over 1 MB.
- Without `include`: code, docs and config files by type (`.py`, `.sql`, `.md`, `.toml`, `.yaml`, `.json`,
  `.ts`, `Dockerfile`, ...), so data files such as CSV or Parquet are left out unless you include them.

Everything the agent reads is sent to the model endpoint, like any other tool result. A line that looks like a
credential is replaced with `[line withheld: it looks like a credential]` (an API key, a private key block, or
`password = "..."`; in source code only a quoted value counts, so `token = request.headers.get(...)` stays
readable). This is a heuristic, not a guarantee: keep secrets out of the repository. A `root` that is a whole
drive, holds your home folder, or is or is inside a credential folder (`~/.ssh`, `~/.aws`, ...) is refused.
Without `[project]`, the tools refuse and the agent says how to turn them on. The startup panel's `Project:`
line shows which folder the agent can read.

## Add data from anywhere on this machine {#data-folders}

The data folder (`[hailer].data_dir`) can be anywhere already; data folders add more, each with a name:

```bash
uv run hailer data add finance "~/OneDrive/Finance exports" --description "Monthly finance extracts"
uv run hailer data              # list them
uv run hailer data remove finance
```

`data add` writes the folder to `.hailer/local.toml`, so the project's config and other people's machines
are not changed. In the chat, `/data` lists them and `/data add <name> <folder>` adds one. A running chat sees
a new folder at its next start (`/exit`, then start it again): a container's mounts are fixed when it starts.

Notebooks find each folder as `DATA_FOLDERS["<name>"]`; the docker kernel mounts it read-only at
`/work/folders/<name>`, and the unsafe-local kernel reads it where it is. `list_periods` lists the folders
and describes one with `list_periods(folder="finance")`.

A team can declare the folders in the project's `hailer.toml` and leave the path to each machine:

```toml
[data_folders]
crm = { description = "CRM extracts, refreshed nightly" }          # each machine: hailer data add crm <folder>
warehouse = { path = "exports/parquet", description = "Warehouse exports" }   # inside the repository
```

A folder with no path on this machine, or whose path does not exist, is a warning and is left out. Each
folder follows the same rules as the data folder (see
[Which data folder can be mounted](../security/docker.md#which-data-folder-can-be-mounted)): it cannot be a
whole drive, your home folder, the project folder or a folder with credentials, and Docker cannot mount a
network share (`\\server\share`), so copy such data to a local folder first. Names are lower-case letters,
digits, `-` and `_`.
