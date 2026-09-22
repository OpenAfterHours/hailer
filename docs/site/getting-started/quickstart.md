# Your first analysis

Explore a small dataset, ask a follow-up question and keep the result in a Python notebook.

You'll need:

- [uv](https://docs.astral.sh/uv/getting-started/installation/), which provides the `uvx` command.
  Install it, then reopen your terminal.
- An [OpenAI API key](https://platform.openai.com/api-keys) for the default setup.
- A web browser.

These steps work in PowerShell on Windows and in a terminal on macOS or Linux. `uvx` downloads Hailer,
its dependencies and a suitable Python version when needed. You do not need to clone this repository
or install Docker.

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
The starter settings use OpenAI with `gpt-5.5`.

For a company or another model endpoint, follow [custom endpoint setup](../configuration/models.md#a-custom-or-internal-endpoint)
after `init`, using that provider's login command, then continue below.

### 2. Add your data

Copy the CSV, Parquet, JSON or Excel files you want to analyse into the `data/` folder inside `my-analysis`.
Any filenames work.

For a small example, [download sales.csv](../assets/examples/sales.csv) into `data/`, or save the following as `data/sales.csv`:

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

Hailer opens the notebook in your browser and starts the chat in your terminal. **Keep the notebook tab
open while you work**; it provides the live session that runs the analysis.

You can type or paste your first question while chat preparation continues. Pressing Enter keeps one
message waiting and sends it when preparation finishes; you can keep editing your next draft meanwhile.

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
[Troubleshooting](../reference/troubleshooting.md#troubleshooting). `doctor` starts no
kernel: every `uvx hailer` session starts its own and stops it when the chat ends.

**Data and code:** files are read locally, but your messages, code and notebook tool outputs (which can
include data samples) go to the configured model endpoint. By default, notebook code runs with your
account's file and network access. See [Security](../security/data-handling.md#security) and the optional
[Docker runtime](../security/docker.md#isolated-kernel-docker) for details.
