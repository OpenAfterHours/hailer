"""Tests for hailer.context: context files, skills, prompts, secret detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from hailer.context import (
    load_context,
    looks_like_secret,
    parse_skill_frontmatter,
    read_skill,
    read_skill_file,
    render_prompt,
)
from hailer.errors import HailerError
from hailer.models import HailerConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_config(tmp_path: Path, **kw) -> HailerConfig:
    root = tmp_path / ".config" / "hailer"
    return HailerConfig(
        workspace=tmp_path,
        notebook=tmp_path / "notebooks" / "analysis.py",
        data_dir=tmp_path / "data",
        context_dir=root / "context",
        skills_dir=root / "skills",
        prompts_dir=root / "prompts",
        **kw,
    )


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# frontmatter
# --------------------------------------------------------------------------- #


def test_frontmatter_plain_and_quoted():
    fm = parse_skill_frontmatter('---\nname: my-skill\ndescription: "Does a thing"\nextra: 42\n---\n# body\n')
    assert fm == {"name": "my-skill", "description": "Does a thing", "extra": "42"}


def test_frontmatter_folded_block_like_marimo_pair():
    text = (
        "---\n"
        "name: marimo-pair\n"
        "description: >-\n"
        "  Drive a live marimo notebook as a workspace: run Python in the same kernel\n"
        "  the user does, inspect live notebook state, and commit durable notebook\n"
        "  changes.\n"
        "allowed-tools: Bash(bash **/scripts/execute-code.sh *), Read\n"
        "---\n"
        "\nmarimo is a reactive Python runtime.\n"
    )
    fm = parse_skill_frontmatter(text)
    assert fm["name"] == "marimo-pair"
    assert fm["description"].startswith("Drive a live marimo notebook as a workspace: run Python in the same kernel the user does")
    assert fm["description"].endswith("commit durable notebook changes.")
    assert fm["allowed-tools"].startswith("Bash(")


def test_frontmatter_literal_block_and_missing():
    fm = parse_skill_frontmatter("---\nname: x\nnotes: |\n  line one\n  line two\n---\n")
    assert fm["notes"] == "line one\nline two"
    assert parse_skill_frontmatter("# no frontmatter\n") == {}
    assert parse_skill_frontmatter("---\nname: unterminated\n") == {}


# --------------------------------------------------------------------------- #
# secrets
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "key: sk-abcdefghijklmnopqrstuvwxyz0123",
        "api_key = abc123def456ghi789jkl",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9abc",
        "token=" + "a1" * 25,
        "AKIAIOSFODNN7QWERTYU",
        "password: hunter2hunter2hunter2",
    ],
)
def test_looks_like_secret_positive(text):
    assert looks_like_secret(text)


@pytest.mark.parametrize(
    "text",
    [
        "The env var INTERNAL_MODEL_API_KEY holds the key.",
        "env_key = INTERNAL_MODEL_API_KEY",
        "api_key = <your key here>",
        "api_key = ${INTERNAL_MODEL_API_KEY}",
        "Report RWA in GBP millions with one decimal place.",
        "The token budget is 10000 tokens per skill.",
    ],
)
def test_looks_like_secret_negative(text):
    assert not looks_like_secret(text)


# --------------------------------------------------------------------------- #
# load_context
# --------------------------------------------------------------------------- #


def test_load_context_empty_when_dirs_missing(tmp_path):
    bundle = load_context(make_config(tmp_path))
    assert bundle.context_text == ""
    assert bundle.skills == [] and bundle.prompts == {} and bundle.warnings == []


def test_load_context_orders_and_headings(tmp_path):
    config = make_config(tmp_path)
    write(config.context_dir / "10-second.md", "second body")
    write(config.context_dir / "00-first.md", "first body")
    write(config.context_dir / "notes.txt", "ignored")
    bundle = load_context(config)
    assert bundle.context_text.index("## 00-first.md") < bundle.context_text.index("## 10-second.md")
    assert "first body" in bundle.context_text and "second body" in bundle.context_text
    assert "ignored" not in bundle.context_text
    assert [p.name for p in bundle.context_files] == ["00-first.md", "10-second.md"]


def test_load_context_cap_and_secret_warning(tmp_path):
    config = make_config(tmp_path, max_context_bytes=120)
    write(config.context_dir / "00-a.md", "x" * 60)
    write(config.context_dir / "01-b.md", "y" * 60)  # pushes over the cap -> skipped
    write(config.context_dir / "02-c.md", "api_key = abc123def456ghi789jkl")
    bundle = load_context(config)
    assert "x" * 60 in bundle.context_text
    assert "y" * 60 not in bundle.context_text
    assert any("not loaded: 01-b.md, 02-c.md" in w for w in bundle.warnings)
    # the secret warning must not fire for a file that was never read past the cap
    assert not any("02-c.md looks like" in w for w in bundle.warnings)


def test_load_context_warns_on_secret_but_keeps_file(tmp_path):
    config = make_config(tmp_path)
    write(config.context_dir / "00-a.md", "api_key = abc123def456ghi789jkl")
    bundle = load_context(config)
    assert "abc123def456ghi789jkl" in bundle.context_text
    assert any("00-a.md looks like it contains a secret" in w for w in bundle.warnings)


def test_load_context_skills_and_prompts(tmp_path):
    config = make_config(tmp_path)
    write(config.skills_dir / "alpha" / "SKILL.md", "---\nname: alpha-skill\ndescription: Alpha things\n---\nbody")
    write(config.skills_dir / "beta" / "SKILL.md", "no frontmatter here")
    write(config.skills_dir / "not-a-skill" / "README.md", "x")
    write(config.prompts_dir / "monthly.md", "Do the monthly pack for {{args}}")
    bundle = load_context(config)
    assert [(s.name, s.description) for s in bundle.skills] == [("alpha-skill", "Alpha things"), ("beta", "")]
    assert bundle.skills[0].path == config.skills_dir / "alpha"
    assert list(bundle.prompts) == ["monthly"]


def test_repo_example_skill_parses():
    root = REPO_ROOT / ".config" / "hailer"
    config = HailerConfig(
        workspace=REPO_ROOT,
        notebook=REPO_ROOT / "notebooks" / "analysis.py",
        data_dir=REPO_ROOT / "data",
        context_dir=root / "context",
        skills_dir=root / "skills",
        prompts_dir=root / "prompts",
    )
    bundle = load_context(config)
    names = [s.name for s in bundle.skills]
    assert "sales-month-on-month" in names
    skill = next(s for s in bundle.skills if s.name == "sales-month-on-month")
    assert "month-on-month" in skill.description.lower()
    assert "first-look" in bundle.prompts
    assert "## 00-example.md" in bundle.context_text
    assert not [w for w in bundle.warnings if "secret" in w]


# --------------------------------------------------------------------------- #
# read_skill / read_skill_file
# --------------------------------------------------------------------------- #


def test_read_skill_body_and_files(tmp_path):
    config = make_config(tmp_path)
    write(config.skills_dir / "alpha" / "SKILL.md", "---\nname: alpha-skill\ndescription: d\n---\n# Alpha\nsteps")
    write(config.skills_dir / "alpha" / "reference" / "checks.md", "check 1")
    write(config.skills_dir / "alpha" / "scripts" / "run.py", "print(1)")
    text = read_skill(config, "alpha-skill")
    assert text.startswith("---\nname: alpha-skill")
    assert "# Alpha" in text
    assert "- reference/checks.md" in text and "- scripts/run.py" in text
    # lookup by folder name also works, case-insensitively
    assert read_skill(config, "ALPHA") == text


def test_read_skill_unknown_lists_available(tmp_path):
    config = make_config(tmp_path)
    write(config.skills_dir / "alpha" / "SKILL.md", "---\nname: alpha-skill\n---\n")
    with pytest.raises(HailerError) as exc:
        read_skill(config, "nope")
    assert "alpha-skill" in exc.value.hint


def test_read_skill_file_ok_and_guards(tmp_path):
    config = make_config(tmp_path)
    write(config.skills_dir / "alpha" / "SKILL.md", "---\nname: alpha\n---\n")
    write(config.skills_dir / "alpha" / "reference" / "checks.md", "check 1")
    write(tmp_path / "outside.txt", "secret")
    assert read_skill_file(config, "alpha", "reference/checks.md") == "check 1"
    assert read_skill_file(config, "alpha", "reference\\checks.md") == "check 1"
    for bad in ["../../outside.txt", "reference/../../../outside.txt", str(tmp_path / "outside.txt"), "/etc/passwd", "\\x", ""]:
        with pytest.raises(HailerError):
            read_skill_file(config, "alpha", bad)
    with pytest.raises(HailerError) as exc:
        read_skill_file(config, "alpha", "reference/missing.md")
    assert "reference/checks.md" in exc.value.hint


# --------------------------------------------------------------------------- #
# render_prompt
# --------------------------------------------------------------------------- #


def test_render_prompt(tmp_path):
    config = make_config(tmp_path)
    write(config.prompts_dir / "monthly.md", "Run the pack for {{args}}.\n")
    assert render_prompt(config, "monthly", " 2025-06 ") == "Run the pack for 2025-06."
    assert render_prompt(config, "monthly", "") == "Run the pack for ."
    with pytest.raises(HailerError) as exc:
        render_prompt(config, "weekly", "")
    assert "monthly" in exc.value.hint
