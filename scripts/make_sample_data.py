"""Generate synthetic monthly sales files for demos and tests.

Writes files named ``YY-MM sales.parquet`` (default: six months from 2025-01), one row per
order. Later months add a ``discount`` column so the schema-evolution handling in
``hailer.periods`` is exercised. Deterministic.

Usage:
    uv run python scripts/make_sample_data.py            # writes into ./data
    uv run python scripts/make_sample_data.py --out tmp --months 3 --rows 200
"""

from __future__ import annotations

import argparse
import calendar
import random
from datetime import date
from pathlib import Path

import polars as pl

REGIONS = ["North", "South", "East", "West"]
CATEGORIES = ["Electronics", "Home", "Clothing", "Sports", "Books"]
CHANNELS = ["Online", "Store"]
UNIT_PRICE = {"Electronics": 90.0, "Home": 45.0, "Clothing": 35.0, "Sports": 55.0, "Books": 20.0}
DISCOUNTS = [0.0, 0.0, 0.0, 0.05, 0.1, 0.2]
# Month-on-month growth factors per region; the North drives most of the revenue increase.
GROWTH = {"North": 1.08, "South": 1.01, "East": 1.0, "West": 0.99}


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


def build_month(year: int, month: int, index: int, rows: int, rng: random.Random, *, with_discount: bool) -> pl.DataFrame:
    last_day = calendar.monthrange(year, month)[1]
    order_ids = [f"{year % 100:02d}{month:02d}-{n:05d}" for n in range(1, rows + 1)]
    order_dates = [date(year, month, rng.randint(1, last_day)) for _ in range(rows)]
    regions = [REGIONS[n % len(REGIONS)] for n in range(rows)]
    categories = [rng.choice(CATEGORIES) for _ in range(rows)]
    channels = [CHANNELS[0] if rng.random() < 0.55 else CHANNELS[1] for _ in range(rows)]
    returned = [rng.random() < 0.03 for _ in range(rows)]
    units: list[int] = []
    revenue: list[float] = []
    discounts: list[float] = []
    for region, category in zip(regions, categories, strict=True):
        quantity = rng.randint(1, 3)
        discount = rng.choice(DISCOUNTS)
        price = UNIT_PRICE[category] * rng.uniform(0.9, 1.1)
        units.append(quantity)
        discounts.append(discount)
        revenue.append(round(quantity * price * (1 - discount) * GROWTH[region] ** index, 2))
    data: dict[str, object] = {
        "order_id": order_ids,
        "order_date": order_dates,
        "region": regions,
        "category": categories,
        "channel": channels,
        "units": units,
        "revenue": revenue,
        "returned": returned,
    }
    if with_discount:
        data["discount"] = discounts
    return pl.DataFrame(data)


def write_sample_data(out: Path, *, months: int = 6, rows: int = 400, start: tuple[int, int] = (2025, 1), seed: int = 42, dataset: str = "sales") -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    written: list[Path] = []
    for index, (year, month) in enumerate(_months(start, months)):
        df = build_month(year, month, index, rows, rng, with_discount=index >= 3)
        path = out / f"{year % 100:02d}-{month:02d} {dataset}.parquet"
        df.write_parquet(path)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("data"), help="output directory (default: data)")
    parser.add_argument("--months", type=int, default=6, help="number of monthly files (default: 6)")
    parser.add_argument("--rows", type=int, default=400, help="orders per file (default: 400)")
    parser.add_argument("--start", type=_parse_start, default=(2025, 1), help="first period as YY-MM (default: 25-01)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="sales", help="dataset name used in filenames (default: sales)")
    args = parser.parse_args(argv)
    written = write_sample_data(args.out, months=args.months, rows=args.rows, start=args.start, seed=args.seed, dataset=args.dataset)
    for path in written:
        print(path)
    print(f"Wrote {len(written)} file(s) to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
