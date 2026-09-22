# Hailer

Hailer is a Python command-line package for chatting with your data. Ask questions about local CSV,
Parquet, JSON or Excel files in plain English, and it explores the data and adds tables, charts and summaries
to a live [marimo](https://marimo.io/) notebook in your browser. You chat in the terminal; the notebook
keeps the analysis and its Python code.

[Documentation](https://openafterhours.github.io/hailer/) · [Quickstart](https://openafterhours.github.io/hailer/getting-started/quickstart/) · [Command reference](https://openafterhours.github.io/hailer/reference/commands/)

## Quick start

You'll need:

- [uv](https://docs.astral.sh/uv/getting-started/installation/), which provides the `uvx` command.
  Install it, then reopen your terminal.
- **Docker**, installed and running: [Docker Desktop](https://docs.docker.com/desktop/) on Windows or
  macOS, [Docker Engine](https://docs.docker.com/engine/install/) on Linux. Your notebook code runs in a
  Docker container that sees only your data folder, read-only, with no network.
- An [OpenAI API key](https://platform.openai.com/api-keys) for the default setup.
- A web browser.

These steps work in PowerShell on Windows and in a terminal on macOS or Linux. `uvx` downloads Hailer,
its dependencies and a suitable Python version when needed. You do not need to clone this repository.

### 1. Create a workspace and sign in

Run these commands in your terminal:

```bash
mkdir my-analysis
cd my-analysis
uvx hailer init
uvx hailer login openai
```

`init` creates your settings (`hailer.toml`), a starter notebook and a `data/` folder. At the login prompt,
paste your API key; input is hidden and the key is saved in your operating system's credential store.
The starter settings use OpenAI with `gpt-5.5` and the Docker kernel (`[kernel] runtime = "docker"`).
If Docker is not on your machine, `init` says so and how to get it.

For a company or another model endpoint, follow [custom endpoint setup](https://openafterhours.github.io/hailer/configuration/models/#a-custom-or-internal-endpoint)
after `init`, using that provider's login command, then continue below.

### 2. Add your data

Copy the CSV, Parquet, JSON or Excel files you want to analyse into the `data/` folder inside `my-analysis`.
Any filenames work.

For a small example, save this as `data/sales.csv` in your workspace:

```csv
month,region,revenue
2026-01,North,1200
2026-01,South,900
2026-02,North,1500
2026-02,South,1100
```

### 3. Start chatting

From the same terminal, run:

```bash
uvx hailer
```

The first start downloads Hailer's kernel image, which takes a minute or two. Then Hailer opens the notebook
in your browser and starts the chat in your terminal. **Keep the notebook tab open while you work**; it
provides the live session that runs the analysis.

Ask a question in the terminal, for example:

```text
What data do I have? Summarise the columns and any missing values.
```

With the sample CSV, try:

```text
Chart total revenue by month, split by region.
```

Tables and charts appear in the notebook. Type `/exit` to finish. To continue later, open a terminal
in `my-analysis` and run `uvx hailer` again; it resumes your conversation and active notebook.
Use `/new` for a fresh conversation, or `/help` to see the chat commands.

If startup fails, run `uvx hailer doctor` for checks and suggested fixes, or see
[Troubleshooting](https://openafterhours.github.io/hailer/reference/troubleshooting/#troubleshooting). `Docker is not installed.`
or `Docker is not running.` means exactly that: install or start Docker and run `uvx hailer` again. `doctor`
starts no kernel: every `uvx hailer` session starts its own and stops it when the chat ends.

**Data and code:** files are read locally, but your messages, code and notebook tool outputs (which can
include data samples) go to the configured model endpoint. Notebook code runs in the Docker container,
which sees your data folder read-only and has no network. Where Docker is not an option,
`runtime = "unsafe-local"` runs it on your machine as you, with your files and network, if you accept that.
See [Security](https://openafterhours.github.io/hailer/security/data-handling/#security) and
[kernel runtimes](https://openafterhours.github.io/hailer/security/runtimes/#kernel-runtimes) for details.

## Documentation

- **Using Hailer:** [everyday workflow](https://openafterhours.github.io/hailer/using/workflow/), [notebooks and sessions](https://openafterhours.github.io/hailer/using/notebooks/), [data and Excel](https://openafterhours.github.io/hailer/using/data/).
- **Configuration:** [models and endpoints](https://openafterhours.github.io/hailer/configuration/models/), [context, skills and prompts](https://openafterhours.github.io/hailer/configuration/context/).
- **Security:** [data handling](https://openafterhours.github.io/hailer/security/data-handling/), [kernel runtimes](https://openafterhours.github.io/hailer/security/runtimes/).
- **Help:** [installation and upgrades](https://openafterhours.github.io/hailer/getting-started/installation/), [troubleshooting](https://openafterhours.github.io/hailer/reference/troubleshooting/), [known limitations](https://openafterhours.github.io/hailer/reference/limitations/).

## Development

```bash
git clone https://github.com/OpenAfterHours/hailer.git
cd hailer
uv sync --locked
uv run hailer login openai
uv run hailer
uv run pytest
```

Read the [development guide](docs/site/development/contributing.md) for sample data and test details,
and the [release guide](docs/site/development/releasing.md) for the release process.
Before changing the agent, providers, tools or conversation lifecycle, read
[docs/LEARNINGS.md](docs/LEARNINGS.md), [docs/INTERFACES.md](docs/INTERFACES.md) and [PLAN.md](PLAN.md).

Preview the documentation:

```bash
uv run --locked --group docs zensical serve
```

See [Maintaining the docs](docs/site/development/docs.md) for build and publishing instructions.
