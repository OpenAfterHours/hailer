"""Typed internal models shared across Hailer modules.

Keep this module dependency-free (standard library only) so every other module
can import it without side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


# --------------------------------------------------------------------------- #
# Data / periods
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, order=True)
class Period:
    """A reporting month encoded in a filename as ``YY-MM``."""

    year: int
    month: int

    @property
    def label(self) -> str:
        """Long form, e.g. ``2025-03``."""
        return f"{self.year:04d}-{self.month:02d}"

    @property
    def short(self) -> str:
        """Filename form, e.g. ``25-03``."""
        return f"{self.year % 100:02d}-{self.month:02d}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.label


@dataclass(frozen=True)
class PeriodFile:
    """A data file whose period is encoded in its name, e.g. ``25-03 pra101.parquet``."""

    path: Path
    period: Period
    stem: str  # the dataset name after the period, lower-cased, e.g. "pra101"


# --------------------------------------------------------------------------- #
# Marimo
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MarimoServer:
    """A running marimo server discovered from the local registry or config."""

    url: str  # base URL without trailing slash, e.g. http://127.0.0.1:2718
    server_id: str = ""
    pid: int | None = None
    version: str = ""
    source: Literal["config", "registry", "env"] = "config"


@dataclass(frozen=True)
class MarimoSession:
    """An active kernel session (exists only while the notebook is open in a browser)."""

    session_id: str
    filename: str | None
    path: str | None


@dataclass
class ExecResult:
    """Result of running code in the marimo scratchpad via ``/api/kernel/execute``."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    output: str = ""  # rendered value of the scratch cell (text form)
    mimetype: str = "text/plain"
    truncated: bool = False

    def as_text(self, max_chars: int | None = None) -> str:
        """Compact, model-friendly rendering. Truncation is applied by the caller."""
        parts: list[str] = []
        if self.stdout.strip():
            parts.append(self.stdout.rstrip())
        if self.output.strip() and self.output.strip() != self.stdout.strip():
            parts.append(self.output.rstrip())
        if self.stderr.strip():
            parts.append("[stderr]\n" + self.stderr.rstrip())
        text = "\n".join(parts) if parts else ("(no output)" if self.success else "(failed, no output)")
        if max_chars is not None and len(text) > max_chars:
            return text[:max_chars]
        return text


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProviderConfig:
    """A model provider, passed through to Codex as ``model_providers.<id>.*``."""

    id: str
    base_url: str | None = None
    wire_api: str = "responses"
    env_key: str | None = None
    requires_openai_auth: bool = False
    name: str | None = None
    http_headers: dict[str, str] = field(default_factory=dict)
    env_http_headers: dict[str, str] = field(default_factory=dict)
    query_params: dict[str, str] = field(default_factory=dict)

    @property
    def is_builtin_openai(self) -> bool:
        return self.id == "openai" and self.base_url is None


@dataclass(frozen=True)
class ModelConfig:
    name: str = "gpt-5.5"
    provider: str = "openai"
    reasoning_effort: str | None = "medium"


@dataclass(frozen=True)
class WebConfig:
    """Websites the agent may read for context. Empty list == no internet access."""

    allowed_domains: tuple[str, ...] = ()
    max_page_bytes: int = 200_000
    allow_shell_network: bool = False


@dataclass(frozen=True)
class HailerConfig:
    """Fully resolved configuration (file + environment overrides). Paths are absolute."""

    workspace: Path
    notebook: Path
    data_dir: Path
    context_dir: Path
    skills_dir: Path
    prompts_dir: Path
    model: ModelConfig = field(default_factory=ModelConfig)
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    web: WebConfig = field(default_factory=WebConfig)
    marimo_url: str | None = None
    marimo_token: str | None = None  # value is read from env only, never from file
    codex_home: Path | None = None
    log_level: str = "WARNING"
    config_path: Path | None = None
    max_tool_output_chars: int = 12_000
    max_context_bytes: int = 24_000

    @property
    def provider(self) -> ProviderConfig:
        """The active provider; a bare ``openai`` provider is synthesised if not declared."""
        pid = self.model.provider
        if pid in self.providers:
            return self.providers[pid]
        if pid == "openai":
            return ProviderConfig(id="openai", env_key="OPENAI_API_KEY", requires_openai_auth=True)
        raise KeyError(pid)


# --------------------------------------------------------------------------- #
# Context / skills
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SkillInfo:
    name: str
    description: str
    path: Path  # directory containing SKILL.md


@dataclass
class ContextBundle:
    """Everything loaded from the ``.config/hailer`` folder for one session."""

    context_text: str = ""
    context_files: list[Path] = field(default_factory=list)
    skills: list[SkillInfo] = field(default_factory=list)
    prompts: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


AgentEventKind = Literal["message_delta", "command", "command_output", "tool_call", "reasoning", "status"]


@dataclass(frozen=True)
class AgentEvent:
    """A progress event surfaced to the terminal while a turn is running."""

    kind: AgentEventKind
    text: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnSummary:
    final_response: str
    thread_id: str
    turn_id: str
    duration_ms: int | None = None
    commands: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    status: str = "completed"


# --------------------------------------------------------------------------- #
# CLI / session
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Command:
    """A parsed slash command such as ``/model gpt-5.5``."""

    name: str
    args: str = ""


@dataclass
class SessionState:
    thread_id: str | None = None
    model: str | None = None
    provider: str | None = None
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class Check:
    """One preflight/doctor check result."""

    name: str
    ok: bool
    summary: str
    hint: str = ""
    fatal: bool = True
