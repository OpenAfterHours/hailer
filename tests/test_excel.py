"""Excel capability checks against the installed dependencies, without optional test packages."""

from datetime import date

import fastexcel
import polars as pl
import pytest

from fake_excel import write_workbook
from hailer.periods import list_data_files


def test_excel_sheets_can_be_inspected_and_read_with_polars(tmp_path):
    path = write_workbook(tmp_path / "Regional sales.xlsx")
    assert list_data_files(tmp_path) == [path]
    assert fastexcel.read_excel(path).sheet_names == ["Sales", "Budget"]

    # The default read_excel engine must work without installing anything at runtime.
    sales = pl.read_excel(path)
    assert sales.shape == (3, 4)
    assert sales["region"].to_list() == ["North", "South", "North"]
    assert sales["revenue"].to_list() == [120.5, 80.0, None]
    assert sales["sold_on"].to_list() == [date(2026, 9, day) for day in (1, 2, 3)]

    selected = pl.read_excel(path, sheet_name="Sales", schema_overrides={"customer_id": pl.String})
    assert selected["customer_id"].to_list() == ["001", "002", "003"]
    assert selected["revenue"].sum() == 200.5

    budget = pl.read_excel(path, sheet_name="Budget")
    assert budget["target"].to_list() == [150, 100], "the formula's saved result is readable"
    sheets = pl.read_excel(path, sheet_id=0)
    assert list(sheets) == ["Sales", "Budget"]
    assert sheets["Sales"].equals(sales)
    assert sheets["Budget"].equals(budget)


@pytest.mark.parametrize("suffix", [".xlsx", ".XLSX", ".xls", ".XLS", ".xlsb", ".XLSB"])
def test_excel_formats_are_discoverable(tmp_path, suffix):
    path = tmp_path / f"sales{suffix}"
    path.touch()  # Discovery is by suffix; it must not parse the workbook.
    (tmp_path / f"folder{suffix}").mkdir()
    assert list_data_files(tmp_path) == [path]
