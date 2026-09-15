"""Generate synthetic PRA101-style monthly parquet files for demos and tests.

Writes files named ``YY-MM pra101.parquet`` (default: six months from 2025-01)
with credit-risk-like columns. Later months add a ``risk_weight`` column so the
schema-evolution handling in ``hailer.periods`` is exercised. Deterministic.

Usage:
    uv run python scripts/make_sample_data.py            # writes into ./data
    uv run python scripts/make_sample_data.py --out tmp --months 3 --rows 200
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import polars as pl

EXPOSURE_CLASSES = ["Corporate", "Retail", "Sovereign", "Institution", "Equity"]
COUNTRIES = ["GB", "US", "DE", "FR", "JP", "NL"]
BASE_RISK_WEIGHT = {"Corporate": 1.0, "Retail": 0.75, "Sovereign": 0.0, "Institution": 0.5, "Equity": 2.5}
# Month-on-month growth factors per class; corporates drive most of the RWA increase.
GROWTH = {"Corporate": 1.045, "Retail": 1.01, "Sovereign": 1.0, "Institution": 1.005, "Equity": 0.99}


def _parse_start(value: str) -> tuple[int, int]:
    yy, mm = value.split("-")
    year, month = 2000 + int(yy), int(mm)
    if not 1 <= month <= 12:
        raise argparse.ArgumentTypeError("month must be 01-12")
    return year, month


def _months(start: tuple[int, int], count: int) -> list[tuple[int, int]]:
    year, month = start
    out = []
    for _ in range(count):
        out.append((year, month))
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return out


def build_month(index: int, rows: int, rng: random.Random, *, with_risk_weight: bool) -> pl.DataFrame:
    counterparty_ids = [f"CP{n:05d}" for n in range(1, rows + 1)]
    classes = [EXPOSURE_CLASSES[n % len(EXPOSURE_CLASSES)] for n in range(rows)]
    countries = [rng.choice(COUNTRIES) for _ in range(rows)]
    default_flags = [rng.random() < 0.04 for _ in range(rows)]
    exposures: list[float] = []
    rwas: list[float] = []
    risk_weights: list[float] = []
    for cls, defaulted in zip(classes, default_flags, strict=True):
        base = rng.uniform(50_000, 5_000_000) * (1.2 if cls == "Corporate" else 1.0)
        exposure = base * (GROWTH[cls] ** index)
        weight = BASE_RISK_WEIGHT[cls] * (1.5 if defaulted else 1.0) * rng.uniform(0.9, 1.1)
        exposures.append(round(exposure, 2))
        risk_weights.append(round(weight, 4))
        rwas.append(round(exposure * weight, 2))
    data: dict[str, object] = {
        "counterparty_id": counterparty_ids,
        "exposure_class": classes,
        "country": countries,
        "default_flag": default_flags,
        "exposure_value": exposures,
        "rwa": rwas,
    }
    if with_risk_weight:
        data["risk_weight"] = risk_weights
    return pl.DataFrame(data)


def write_sample_data(out: Path, *, months: int = 6, rows: int = 400, start: tuple[int, int] = (2025, 1), seed: int = 42, dataset: str = "pra101") -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    written: list[Path] = []
    for index, (year, month) in enumerate(_months(start, months)):
        df = build_month(index, rows, rng, with_risk_weight=index >= 3)
        path = out / f"{year % 100:02d}-{month:02d} {dataset}.parquet"
        df.write_parquet(path)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("data"), help="output directory (default: data)")
    parser.add_argument("--months", type=int, default=6, help="number of monthly files (default: 6)")
    parser.add_argument("--rows", type=int, default=400, help="rows per file (default: 400)")
    parser.add_argument("--start", type=_parse_start, default=(2025, 1), help="first period as YY-MM (default: 25-01)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="pra101", help="dataset name used in filenames (default: pra101)")
    args = parser.parse_args(argv)
    written = write_sample_data(args.out, months=args.months, rows=args.rows, start=args.start, seed=args.seed, dataset=args.dataset)
    for path in written:
        print(path)
    print(f"Wrote {len(written)} file(s) to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
