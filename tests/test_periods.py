"""Tests for hailer.periods (filename periods, schema-tolerant loading, DuckDB view)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
import pytest

from hailer import periods as pf
from hailer.errors import MalformedParquetError
from hailer.periods import Period, PeriodFile

# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("25-01 pra101.parquet", Period(2025, 1)),
        ("25-12 pra101.parquet", Period(2025, 12)),
        ("99-06 x.parquet", Period(2099, 6)),
        ("25-03 PRA101.PARQUET", Period(2025, 3)),
        ("25-03   spaced name.parquet", Period(2025, 3)),
        ("25-03 pra.101.parquet", Period(2025, 3)),
        ("25-13 x.parquet", None),
        ("25-00 x.parquet", None),
        ("2025-01 x.parquet", None),
        ("25-01.parquet", None),
        ("pra101 25-01.parquet", None),
        ("25_01 x.parquet", None),
        ("", None),
    ],
)
def test_parse_period(name, expected):
    assert pf.parse_period(name) == expected
    assert pf.parse_period(Path("some") / "dir" / name) == expected if name else True


def test_period_labels():
    p = Period(2025, 3)
    assert p.label == "2025-03" and p.short == "25-03" and str(p) == "2025-03"
    assert Period(2025, 1) < Period(2025, 2) < Period(2026, 1)


def test_parse_period_file_stem_lowercased(tmp_path):
    path = tmp_path / "25-03 PRA101.parquet"
    result = pf.parse_period_file(path)
    assert result == PeriodFile(path=path, period=Period(2025, 3), stem="pra101")
    assert pf.parse_period_file(tmp_path / "notes.parquet") is None


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def _write(path: Path, **columns) -> Path:
    pl.DataFrame(columns).write_parquet(path)
    return path


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    _write(d / "25-02 pra101.parquet", exposure_class=["Corporate", "Retail"], rwa=[100, 50])
    _write(d / "25-01 pra101.parquet", exposure_class=["Corporate", "Retail"], rwa=[90, 40])
    # later month adds a column and changes rwa to float (schema evolution)
    _write(d / "25-03 pra101.parquet", exposure_class=["Corporate"], rwa=[120.5], risk_weight=[1.0])
    _write(d / "25-01 pra102.parquet", other=[1])
    (d / "25-01 notes.txt").write_text("not data", encoding="utf-8")
    (d / "readme.parquet").write_bytes(b"")  # no period in name -> ignored by scan
    return d


# --------------------------------------------------------------------------- #
# scanning / loading
# --------------------------------------------------------------------------- #


def test_scan_sorted_and_filtered(data_dir):
    all_files = pf.scan_period_files(data_dir)
    assert [(f.period.label, f.stem) for f in all_files] == [("2025-01", "pra101"), ("2025-01", "pra102"), ("2025-02", "pra101"), ("2025-03", "pra101")]
    only = pf.scan_period_files(data_dir, "PRA101")
    assert [f.path.name for f in only] == ["25-01 pra101.parquet", "25-02 pra101.parquet", "25-03 pra101.parquet"]
    assert pf.scan_period_files(data_dir / "missing") == []
    # suffix filter is case-insensitive and other suffixes are ignored by default
    assert [f.stem for f in pf.scan_period_files(data_dir, None, suffix=".TXT")] == ["notes"]
    assert all(f.path.suffix == ".parquet" for f in all_files)


def test_list_data_files_any_name(data_dir):
    (data_dir / "Customers.CSV").write_text("id\n1\n", encoding="utf-8")
    (data_dir / "survey.jsonl").write_text("{}\n", encoding="utf-8")
    (data_dir / "archive.parquet").mkdir()  # folders are skipped, whatever their suffix
    names = [p.name for p in pf.list_data_files(data_dir)]
    assert names == [
        "25-01 pra101.parquet",
        "25-01 pra102.parquet",
        "25-02 pra101.parquet",
        "25-03 pra101.parquet",
        "Customers.CSV",
        "readme.parquet",
        "survey.jsonl",
    ]
    assert pf.list_data_files(data_dir / "missing") == []


def test_load_periods_adds_period_first_and_tolerates_schema_change(data_dir):
    files = pf.scan_period_files(data_dir, "pra101")
    df = pf.load_periods(files)
    assert df.columns[0] == "period"
    assert df.get_column("period").to_list() == ["2025-01", "2025-01", "2025-02", "2025-02", "2025-03"]
    assert "risk_weight" in df.columns
    assert df.get_column("risk_weight").null_count() == 4
    assert df.get_column("rwa").dtype == pl.Float64  # relaxed int -> float
    assert df.filter(pl.col("period") == "2025-03").get_column("rwa").to_list() == [120.5]


def test_load_periods_column_subset_and_empty(data_dir):
    files = pf.scan_period_files(data_dir, "pra101")
    df = pf.load_periods(files, columns=["rwa", "risk_weight"])
    assert df.columns == ["period", "rwa", "risk_weight"]
    empty = pf.load_periods([])
    assert empty.columns == ["period"] and empty.height == 0


def test_scan_periods_matches_eager(data_dir):
    files = pf.scan_period_files(data_dir, "pra101")
    lazy = pf.scan_periods(files).collect()
    eager = pf.load_periods(files)
    assert lazy.columns == eager.columns
    assert lazy.sort(["period", "exposure_class"]).equals(eager.sort(["period", "exposure_class"]))
    assert pf.scan_periods([]).collect().columns == ["period"]


def test_malformed_parquet_names_file(data_dir):
    bad = data_dir / "25-04 pra101.parquet"
    bad.write_bytes(b"this is not parquet at all")
    files = pf.scan_period_files(data_dir, "pra101")
    with pytest.raises(MalformedParquetError) as exc:
        pf.load_periods(files)
    assert "25-04 pra101.parquet" in str(exc.value) and exc.value.hint
    with pytest.raises(MalformedParquetError):
        pf.scan_periods(files)
    with pytest.raises(MalformedParquetError):
        pf.describe_periods(files)


# --------------------------------------------------------------------------- #
# duckdb
# --------------------------------------------------------------------------- #


def test_duckdb_view_derives_period_from_filename(data_dir):
    files = pf.scan_period_files(data_dir, "pra101")
    con = duckdb.connect()
    name = pf.duckdb_periods_view(con, files, "pra101")
    assert name == "pra101"
    rows = con.execute("SELECT period, count(*) AS n, sum(rwa) AS rwa FROM pra101 GROUP BY period ORDER BY period").fetchall()
    assert rows == [("2025-01", 2, 130.0), ("2025-02", 2, 150.0), ("2025-03", 1, 120.5)]
    cols = [c[0] for c in con.execute("DESCRIBE pra101").fetchall()]
    assert cols[0] == "period" and "risk_weight" in cols and "filename" not in cols
    assert con.execute("SELECT count(*) FROM pra101 WHERE risk_weight IS NULL").fetchone()[0] == 4


def test_duckdb_view_empty_and_bad_name(data_dir):
    con = duckdb.connect()
    assert pf.duckdb_periods_view(con, [], "empty") == "empty"
    assert con.execute("SELECT count(*) FROM empty").fetchone()[0] == 0
    with pytest.raises(ValueError):
        pf.duckdb_periods_view(con, [], "bad name; drop")


# --------------------------------------------------------------------------- #
# describe
# --------------------------------------------------------------------------- #


def test_describe_periods_text(data_dir):
    files = pf.scan_period_files(data_dir, "pra101")
    text = pf.describe_periods(files)
    assert text.startswith("3 period file(s) [pra101]: 2025-01 -> 2025-03")
    assert "Common columns (2): exposure_class String, rwa" in text
    assert "Only in some periods: risk_weight (1 file(s), 2025-03)" in text
    assert "Type differs across periods: rwa: Float64 / Int64" in text
    assert pf.describe_periods([]).startswith("No period files found")
