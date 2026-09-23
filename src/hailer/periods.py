"""Utilities for the files in the data folder.

:func:`list_data_files` finds every data file, whatever its name (CSV, Parquet, JSON, ...).
The rest of the module is for an optional convention: monthly files named
``YY-MM <dataset>.parquet``, e.g. ``25-03 sales.parquet`` is the ``sales`` dataset for
March 2025. The parquet data itself is not assumed to contain a period column; these
helpers add one, and they cope with columns appearing in later months (schema evolution).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from hailer.errors import MalformedParquetError

PERIOD_RE = re.compile(r"^(?P<yy>\d{2})-(?P<mm>\d{2})\s+(?P<stem>.+)$")

_READ_ERRORS: tuple[type[BaseException], ...] = (pl.exceptions.PolarsError, OSError, ValueError, TypeError)

#: File types :func:`list_data_files` reports (Excel is read by Polars with the bundled fastexcel).
DATA_SUFFIXES: tuple[str, ...] = (
    ".csv", ".tsv", ".parquet", ".json", ".jsonl", ".ndjson", ".xlsx", ".xls", ".xlsb", ".arrow", ".feather", ".ipc",
)


# --------------------------------------------------------------------------- #
# Periods
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
    """A data file whose period is encoded in its name, e.g. ``25-03 sales.parquet``."""

    path: Path
    period: Period
    stem: str  # the dataset name after the period, lower-cased, e.g. "sales"


# --------------------------------------------------------------------------- #
# Any data file
# --------------------------------------------------------------------------- #


def list_data_files(data_dir: Path) -> list[Path]:
    """The data files directly in ``data_dir`` (any name, a type in :data:`DATA_SUFFIXES`), sorted by name."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return []
    files = [p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in DATA_SUFFIXES]
    return sorted(files, key=lambda p: p.name.lower())


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #


def _file_stem(name: str | Path) -> str:
    path = Path(name)
    return path.stem if path.suffix else path.name


def parse_period(name: str | Path) -> Period | None:
    """``"25-03 sales.parquet"`` → ``Period(2025, 3)``; ``None`` when the name does not follow the convention."""
    match = PERIOD_RE.match(_file_stem(name).strip())
    if not match:
        return None
    month = int(match.group("mm"))
    if not 1 <= month <= 12:
        return None
    return Period(year=2000 + int(match.group("yy")), month=month)


def parse_period_file(path: Path) -> PeriodFile | None:
    path = Path(path)
    match = PERIOD_RE.match(_file_stem(path).strip())
    period = parse_period(path)
    if match is None or period is None:
        return None
    return PeriodFile(path=path, period=period, stem=match.group("stem").strip().lower())


def scan_period_files(data_dir: Path, name: str | None = None, *, suffix: str = ".parquet") -> list[PeriodFile]:
    """All period-named files in ``data_dir`` (optionally one dataset), sorted by period."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return []
    wanted = name.strip().lower() if name else None
    found: list[PeriodFile] = []
    for path in data_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != suffix.lower():
            continue
        pf = parse_period_file(path)
        if pf is None or (wanted is not None and pf.stem != wanted):
            continue
        found.append(pf)
    found.sort(key=lambda f: (f.period, f.stem, f.path.name))
    return found


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _malformed(path: Path, err: BaseException) -> MalformedParquetError:
    return MalformedParquetError(
        f"Could not read {path.name}: {err}",
        hint=f"Check that {path} is a valid parquet file (it may be truncated, not parquet, or still being written).",
    )


def _read_one(pf: PeriodFile, columns: Sequence[str] | None) -> pl.DataFrame:
    try:
        df = pl.read_parquet(pf.path)
    except _READ_ERRORS as err:
        raise _malformed(pf.path, err) from err
    if columns is not None:
        keep = [c for c in columns if c in df.columns]
        df = df.select(keep)
    return df.select([pl.lit(pf.period.label).alias("period"), pl.all()])


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema={"period": pl.Utf8})


def load_periods(files: Sequence[PeriodFile], *, columns: Sequence[str] | None = None) -> pl.DataFrame:
    """Concatenate the files with a leading ``period`` (``YYYY-MM``) column; tolerant of new columns over time."""
    frames = [_read_one(pf, columns) for pf in files]
    if not frames:
        return _empty()
    return pl.concat(frames, how="diagonal_relaxed")


def scan_periods(files: Sequence[PeriodFile]) -> pl.LazyFrame:
    """Lazy variant of :func:`load_periods`."""
    lazies: list[pl.LazyFrame] = []
    for pf in files:
        try:
            lf = pl.scan_parquet(pf.path)
            lf.collect_schema()  # surfaces unreadable files eagerly
        except _READ_ERRORS as err:
            raise _malformed(pf.path, err) from err
        lazies.append(lf.select([pl.lit(pf.period.label).alias("period"), pl.all()]))
    if not lazies:
        return _empty().lazy()
    return pl.concat(lazies, how="diagonal_relaxed")


def _sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def duckdb_periods_view(con: duckdb.DuckDBPyConnection, files: Sequence[PeriodFile], view_name: str = "periods") -> str:
    """Create (or replace) a DuckDB view over the files with a derived ``period`` column. Returns the view name."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", view_name):
        raise ValueError(f"invalid view name: {view_name!r}")
    if not files:
        con.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT CAST(NULL AS VARCHAR) AS period WHERE false")
        return view_name
    paths = [pf.path.resolve().as_posix() for pf in files]
    file_list = ", ".join(_sql_str(p) for p in paths)
    mapping = ", ".join(f"({_sql_str(p)}, {_sql_str(pf.period.label)})" for p, pf in zip(paths, files, strict=True))
    sql = (
        f"CREATE OR REPLACE VIEW {view_name} AS "
        f"SELECT m.period, d.* EXCLUDE (filename) "
        f"FROM read_parquet([{file_list}], union_by_name=true, filename=true) AS d "
        f"JOIN (VALUES {mapping}) AS m(filename, period) "
        f"ON replace(d.filename, '\\', '/') = m.filename"
    )
    try:
        con.execute(sql)
    except duckdb.Error as err:  # pragma: no cover - depends on file contents
        bad = next((pf.path for pf in files if pf.path.name in str(err)), files[0].path)
        raise _malformed(bad, err) from err
    return view_name


# --------------------------------------------------------------------------- #
# Description
# --------------------------------------------------------------------------- #


def _schema(pf: PeriodFile) -> dict[str, pl.DataType]:
    try:
        return dict(pl.read_parquet_schema(pf.path))
    except _READ_ERRORS as err:
        raise _malformed(pf.path, err) from err


def describe_periods(files: Sequence[PeriodFile], schema: Callable[[PeriodFile], Mapping[str, Any]] | None = None) -> str:
    """Compact text summary: count, span, common columns, and columns that only appear in some periods.

    ``schema`` reads a file's columns and types (default: the Parquet schema at ``pf.path``); Hailer's
    ``list_periods`` tool passes one that asks the kernel's sandbox, so it never opens the data folder.
    """
    if not files:
        return "No period files found (expected names like '25-01 sales.parquet')."
    ordered = sorted(files, key=lambda f: (f.period, f.stem))
    read = schema or _schema
    schemas = [(pf, read(pf)) for pf in ordered]
    stems = sorted({pf.stem for pf in ordered})
    first, last = ordered[0].period.label, ordered[-1].period.label
    lines = [f"{len(ordered)} period file(s) [{', '.join(stems)}]: {first} -> {last}"]
    if len(ordered) <= 24:
        lines.append("Periods: " + ", ".join(pf.period.label for pf in ordered))

    common: list[str] = [c for c in schemas[0][1] if all(c in s for _, s in schemas)]
    lines.append(f"Common columns ({len(common)}): " + ", ".join(f"{c} {schemas[0][1][c]}" for c in common))

    partial: dict[str, list[str]] = {}
    for pf, schema in schemas:
        for col in schema:
            if col not in common:
                partial.setdefault(col, []).append(pf.period.label)
    for col, periods in partial.items():
        span = f"{periods[0]}..{periods[-1]}" if len(periods) > 1 else periods[0]
        lines.append(f"Only in some periods: {col} ({len(periods)} file(s), {span})")

    changed: list[str] = []
    for col in common:
        types = {str(s[col]) for _, s in schemas}
        if len(types) > 1:
            changed.append(f"{col}: {' / '.join(sorted(types))}")
    if changed:
        lines.append("Type differs across periods: " + "; ".join(changed))
    return "\n".join(lines)


__all__ = [
    "DATA_SUFFIXES",
    "PERIOD_RE",
    "describe_periods",
    "duckdb_periods_view",
    "list_data_files",
    "load_periods",
    "parse_period",
    "parse_period_file",
    "scan_period_files",
    "scan_periods",
]
