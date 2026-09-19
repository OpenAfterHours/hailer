# Allowed web domains {#allowed-web-domains}

By default the agent's **`fetch_page` tool is disabled**: it refuses every URL. To let that tool read
specific sites for context:

```toml
[web]
allowed_domains = ["docs.pola.rs", "duckdb.org", "**.marimo.io"]
max_page_bytes = 200000
```

- Exact hosts match themselves; `*.example.com` matches subdomains only; `**.example.com` matches the apex
  and its subdomains. Matching is case-insensitive and ignores ports. Loopback and private addresses are
  only allowed when listed literally (for example `127.0.0.1`).
- `fetch_page` checks the host before any network I/O and on every redirect, converts HTML to readable
  text, caps the size, and returns the text to the model. A refused URL tells the agent which domains are
  allowed so it can ask you to extend the list.

Text fetched from an allowed site becomes a tool result and is sent to the model endpoint like any other
result. `/context`, `/status` and the startup panel show the active allowlist. The allowlist governs the
agent's own web tool, not what notebook code can do: with the default local kernel, code the agent runs
in the notebook is ordinary Python on your machine with your network access; with the docker kernel it
has no network at all unless `[kernel] network = true` (see [Security](../security/data-handling.md#security)).
