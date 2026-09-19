# Credentials {#secrets}

`uvx hailer login <provider>` stores the API key in the operating system's credential store through
`keyring` (Windows Credential Manager on Windows) under the service name `hailer`. Hailer reads it when the
agent starts and hands it to the HTTP client inside its own process, which sends it only as the
`Authorization` header of requests to that provider's endpoint. It is never written to `hailer.toml`, the
session file, command history or logs, and never passed on a command line or to another process.
`uvx hailer logout <provider>` removes it.

An environment variable named by `env_key` takes precedence over the credential store, which keeps scripted
and CI use simple. Hailer leaves that variable (and other secret-looking ones) out of the environment of the
marimo server it starts, so it is not in notebook code's environment (see [Security](../security/data-handling.md#security)). Note that any
process running as the same user can read both credential-store entries and environment variables; the
gain is keeping the key out of files and history, not isolation from your own account.

Every provider needs a key, including the built-in `openai` one. A missing key stops `hailer` at startup
with the `login` command to run.
