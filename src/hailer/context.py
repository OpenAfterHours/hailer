"""User-supplied context from the ``.config/hailer`` folder.

- ``context/*.md``  always-on text appended to the agent's developer instructions
- ``skills/<name>/SKILL.md``  on-demand skills (Agent Skills format); only an index goes in the prompt
- ``prompts/<name>.md``  reusable prompts invoked with ``/prompt <name> [args]``

Everything loaded here is sent to the configured model endpoint, so the loader
warns when a file looks like it contains a secret.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

from .errors import HailerError
from .models import ContextBundle, HailerConfig, SkillInfo

SKILL_FILE = "SKILL.md"

# (pattern, apply_exclusions): fixed-format credentials match outright; ``key = value``
# style matches are filtered so env var *names* and placeholders do not trigger.
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], bool], ...] = (
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), False),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), False),
    (re.compile(r"\b[A-Fa-f0-9]{40,}\b"), False),
    (re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._\-]{16,})"), True),
    (re.compile(r"(?i)\b(api[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*['\"]?([A-Za-z0-9_\-./+=]{12,})"), True),
)
_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_PLACEHOLDER_RE = re.compile(r"^[<$\{].*|.*(your|example|placeholder|xxx|\.\.\.).*$", re.IGNORECASE)


def looks_like_secret(text: str) -> bool:
    """Heuristic: does the text contain something that looks like a credential?"""
    for pattern, apply_exclusions in _SECRET_PATTERNS:
        for match in pattern.finditer(text):
            if not apply_exclusions:
                return True
            value = match.group(match.lastindex) if match.lastindex else match.group(0)
            if _ENV_NAME_RE.match(value) or _PLACEHOLDER_RE.match(value):
                continue  # an env var *name* or an obvious placeholder, not a value
            return True
    return False


# --------------------------------------------------------------------------- #
# Frontmatter
# --------------------------------------------------------------------------- #


def parse_skill_frontmatter(text: str) -> dict[str, str]:
    """Parse the ``---`` delimited YAML-style header of a SKILL.md with the standard library.

    Supports ``key: value``, quoted values, and the folded/literal block forms
    ``key: >-`` / ``key: >`` / ``key: |`` followed by indented lines.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}

    result: dict[str, str] = {}
    key: str | None = None
    block_style: str | None = None  # ">" folded, "|" literal, None plain continuation
    buffer: list[str] = []

    def flush() -> None:
        nonlocal key, block_style, buffer
        if key is None:
            return
        parts = [b.strip() for b in buffer if b.strip()]
        if block_style == "|":
            value = textwrap.dedent("\n".join(b.rstrip() for b in buffer)).strip()
        else:
            value = " ".join(parts)
        if value:
            result[key] = value if key not in result else (result[key] + " " + value).strip()
        key, block_style, buffer = None, None, []

    for line in lines[1:end]:
        if not line.strip():
            if key is not None and block_style is not None:
                buffer.append("")
            continue
        if line[0] in " \t":
            if key is not None:
                buffer.append(line)
            continue
        flush()
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        k, _, v = line.partition(":")
        key = k.strip()
        v = v.strip()
        if v in (">", ">-", ">+", "|", "|-", "|+"):
            block_style = v[0]
            continue
        block_style = None
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        result[key] = v
        key = None
    flush()
    return result


def _skill_dirs(config: HailerConfig) -> list[Path]:
    root = config.skills_dir
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / SKILL_FILE).is_file())


def _skill_info(directory: Path) -> SkillInfo:
    text = (directory / SKILL_FILE).read_text(encoding="utf-8", errors="replace")
    fm = parse_skill_frontmatter(text)
    return SkillInfo(name=fm.get("name") or directory.name, description=fm.get("description", ""), path=directory)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_context(config: HailerConfig) -> ContextBundle:
    """Load always-on context, the skills index and the prompt map for one session."""
    bundle = ContextBundle()

    ctx_dir = config.context_dir
    if ctx_dir.is_dir():
        files = sorted(p for p in ctx_dir.iterdir() if p.is_file() and p.suffix.lower() == ".md")
        parts: list[str] = []
        total = 0
        skipped: list[str] = []
        for f in files:
            if skipped:
                skipped.append(f.name)
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
            if looks_like_secret(text):
                bundle.warnings.append(
                    f"{f.name} looks like it contains a secret; everything in context/ is sent to the model endpoint"
                )
            chunk = f"## {f.name}\n\n{text.strip()}\n"
            size = len(chunk.encode("utf-8"))
            if total + size > config.max_context_bytes:
                skipped.append(f.name)
                continue
            parts.append(chunk)
            total += size
            bundle.context_files.append(f)
        if skipped:
            bundle.warnings.append(
                f"Context limit of {config.max_context_bytes} bytes reached; not loaded: " + ", ".join(skipped)
            )
        bundle.context_text = "\n".join(parts).strip()

    for d in _skill_dirs(config):
        try:
            bundle.skills.append(_skill_info(d))
        except OSError as exc:  # pragma: no cover - unreadable file
            bundle.warnings.append(f"Could not read skill {d.name}: {exc}")

    prompts_dir = config.prompts_dir
    if prompts_dir.is_dir():
        bundle.prompts = {p.stem: p for p in sorted(prompts_dir.glob("*.md")) if p.is_file()}

    return bundle


# --------------------------------------------------------------------------- #
# Skills
# --------------------------------------------------------------------------- #


def _find_skill(config: HailerConfig, name: str) -> tuple[SkillInfo, list[SkillInfo]]:
    skills = [_skill_info(d) for d in _skill_dirs(config)]
    wanted = name.strip().lower()
    for s in skills:
        if s.name.lower() == wanted or s.path.name.lower() == wanted:
            return s, skills
    available = ", ".join(s.name for s in skills) or "(none)"
    raise HailerError(
        f"No skill named '{name}'",
        f"Available skills: {available}. Skills live in {config.skills_dir} as <name>/SKILL.md.",
    )


def _skill_files(directory: Path) -> list[str]:
    files: list[str] = []
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.name != SKILL_FILE and "__pycache__" not in p.parts:
            files.append(p.relative_to(directory).as_posix())
    return files


def read_skill(config: HailerConfig, name: str) -> str:
    """Return the SKILL.md body followed by the list of files bundled with the skill."""
    skill, _ = _find_skill(config, name)
    body = (skill.path / SKILL_FILE).read_text(encoding="utf-8", errors="replace").strip()
    files = _skill_files(skill.path)
    listing = "\n".join(f"- {f}" for f in files) if files else "- (no additional files)"
    return f"{body}\n\nFiles (read with read_skill_file('{skill.name}', <path>)):\n{listing}"


def read_skill_file(config: HailerConfig, name: str, relative: str) -> str:
    """Read one file bundled with a skill. Rejects absolute paths and traversal."""
    skill, _ = _find_skill(config, name)
    # Accept either separator: the agent may send Windows-style paths, and on POSIX a
    # backslash would otherwise be a literal character in the file name.
    cleaned = relative.strip().replace("\\", "/")
    rel = Path(cleaned)
    if not cleaned or rel.is_absolute() or rel.drive or cleaned.startswith("/"):
        raise HailerError(
            f"Skill file path must be relative to the skill folder: '{relative}'",
            "Use a path such as reference/checks.md.",
        )
    base = skill.path.resolve()
    target = (base / rel).resolve()
    if base != target and base not in target.parents:
        raise HailerError(
            f"'{relative}' is outside the skill folder",
            "Only files under the skill's own folder can be read.",
        )
    if not target.is_file():
        files = _skill_files(skill.path)
        raise HailerError(
            f"No file '{relative}' in skill '{skill.name}'",
            "Available files: " + (", ".join(files) if files else "(none)"),
        )
    return target.read_text(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


def render_prompt(config: HailerConfig, name: str, args: str) -> str:
    """Return the named prompt with ``{{args}}`` substituted."""
    prompts_dir = config.prompts_dir
    prompts = {p.stem: p for p in sorted(prompts_dir.glob("*.md"))} if prompts_dir.is_dir() else {}
    path = prompts.get(name.strip())
    if path is None:
        available = ", ".join(prompts) or "(none)"
        raise HailerError(
            f"No prompt named '{name}'",
            f"Available prompts: {available}. Prompts live in {prompts_dir} as <name>.md.",
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    return text.replace("{{args}}", args.strip()).strip()
