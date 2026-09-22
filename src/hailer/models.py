"""Typed internal models shared across Hailer modules.

Keep this module dependency-free (standard library only) so every other module
can import it without side effects. :class:`~hailer.kernel.PathMap` is only named in
annotations here (a ``TYPE_CHECKING`` import): ``hailer.kernel`` imports this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover - annotations only; hailer.kernel imports this module
    from hailer.kernel import PathMap


# --------------------------------------------------------------------------- #
# Marimo
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MarimoServer:
    """The marimo server this Hailer process started, held in memory for as long as it runs:
    nothing else finds or attaches to it."""

    url: str  # base URL without trailing slash, e.g. http://127.0.0.1:2718
    pid: int | None = None  # local runtime: the marimo process
    #: The server's random auth token. Kept out of ``repr`` so it never lands in a log line or a
    #: test failure.
    token: str | None = field(default=None, repr=False)
    #: Where the kernel runs: "local" (Hailer's own Python) or "docker".
    runtime: str = "local"
    #: Host <-> kernel paths; ``None`` means identity (the kernel sees the host's paths).
    paths: PathMap | None = None
    #: Docker only: whether the kernel was started with ``[kernel] network = true``.
    network_access: bool = False


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


#: Protocols a provider endpoint can speak: the OpenAI Responses API (``POST /responses``) or
#: Chat Completions (``POST /chat/completions``). Hailer talks either one directly.
WIRE_API_RESPONSES = "responses"
WIRE_API_CHAT = "chat"
VALID_WIRE_APIS = (WIRE_API_RESPONSES, WIRE_API_CHAT)


@dataclass(frozen=True)
class ProviderConfig:
    """A model endpoint: the built-in OpenAI one, or a ``[model_providers.<id>]`` table with a ``base_url``."""

    id: str
    base_url: str | None = None
    wire_api: str = WIRE_API_RESPONSES  # one of VALID_WIRE_APIS
    #: False sends ``stream: false`` and reads one JSON reply, for endpoints that reject or
    #: cannot deliver server-sent events.
    stream: bool = True
    #: False omits ``stream_options`` from streamed requests, for endpoints that reject the
    #: field (token counts are then whatever the final chunk carries).
    stream_options: bool = True
    #: Environment variable (and credential-store entry) that holds the API key.
    env_key: str | None = None
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
    #: Older turns are summarised once the conversation passes this many tokens; 0 turns it off.
    #: Lower it for endpoints whose models have a small context window.
    summarize_after_tokens: int = 100_000


@dataclass(frozen=True)
class WebConfig:
    """Websites the agent may read for context. Empty list == no internet access."""

    allowed_domains: tuple[str, ...] = ()
    max_page_bytes: int = 200_000


#: Where the notebook kernel runs: in Hailer's own Python, as the user (the default), or in a
#: Docker container that sees only the notebooks folder and, read-only, the data folder.
KERNEL_RUNTIME_LOCAL = "local"
KERNEL_RUNTIME_DOCKER = "docker"
VALID_KERNEL_RUNTIMES = (KERNEL_RUNTIME_LOCAL, KERNEL_RUNTIME_DOCKER)
#: The published kernel image; the tag is the kernel contract (see ``hailer.kernel_image``).
KERNEL_IMAGE_REPOSITORY = "ghcr.io/openafterhours/hailer-kernel"


@dataclass(frozen=True)
class KernelConfig:
    """The ``[kernel]`` table: where notebook code runs. Only ``runtime`` matters for ``local``."""

    runtime: str = KERNEL_RUNTIME_LOCAL  # one of VALID_KERNEL_RUNTIMES (validate() reports others)
    image: str | None = None  # None: the published image for this Hailer's kernel contract
    memory: str = "4g"  # docker --memory format: <number>[b|k|m|g], as written (any case)
    cpus: float = 2.0
    network: bool = False  # docker only: True puts the kernel on the default bridge network
    #: Local only: environment variables the kernel gets even though their names look secret.
    pass_env: tuple[str, ...] = ()

    @property
    def effective_image(self) -> str:
        """``image``, else ``ghcr.io/openafterhours/hailer-kernel:marimo<version>-<fingerprint>``."""
        from hailer.kernel_image import default_image  # lazy: kernel_image imports this module

        return self.image or default_image()


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
    kernel: KernelConfig = field(default_factory=KernelConfig)
    log_level: str = "WARNING"
    config_path: Path | None = None
    max_tool_output_chars: int = 12_000
    max_context_bytes: int = 24_000
    notebooks_dir: Path | None = None  # load_config always fills it; None only in hand-built test configs

    @property
    def notebooks_root(self) -> Path:
        """The folder notebooks live in (``notebooks_dir``, else the configured notebook's folder)."""
        return self.notebooks_dir if self.notebooks_dir is not None else self.notebook.parent

    @property
    def provider(self) -> ProviderConfig:
        """The active provider; a bare ``openai`` provider is synthesised if not declared."""
        pid = self.model.provider
        if pid in self.providers:
            return self.providers[pid]
        if pid == "openai":
            return ProviderConfig(id="openai", env_key="OPENAI_API_KEY")
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


AgentEventKind = Literal["message_delta", "tool_call", "status"]


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
    duration_ms: int | None = None
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
