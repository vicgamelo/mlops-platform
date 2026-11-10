import csv
import io
from pathlib import Path

import numpy as np
import pytest

from mlops_platform.batch import Limits, parse, report, safe_cell, score, to_csv
from mlops_platform.data import FEATURES

EXAMPLES = Path(__file__).parent.parent / "examples"
VALID = (EXAMPLES / "batch-valid.csv").read_bytes()
ERRORS = (EXAMPLES / "batch-errors.csv").read_bytes()
HEADER = ",".join(FEATURES)
ROW = "36,5711,4,2,38,2,1,no checking,existing paid,used car,>=1000,>=7,none,car,none,own,skilled,yes"


def csv_of(*rows: str, header: str = HEADER) -> bytes:
    return ("\n".join([header, *rows]) + "\n").encode()


def file_codes(body: bytes, limits: Limits | None = None) -> list[str]:
    return [e["code"] for e in parse(body, limits=limits).file_errors]


def cell_codes(body: bytes) -> set[tuple[int, str, str]]:
    return {(p["row"], p["column"], p["code"]) for p in parse(body).problems}


class FakeModel:
    """predict_proba = credit_amount / 10_000, so the test knows each row's score."""

    def predict_proba(self, frame):  # type: ignore[no-untyped-def]
        assert list(frame.columns) == FEATURES
        p = np.clip(frame["credit_amount"].to_numpy(dtype=float) / 10_000, 0, 1)
        return np.column_stack([1 - p, p])


def test_valid_example_is_fully_valid() -> None:
    r = report(parse(VALID))
    assert r["ok"] and r["rows"] == 12 and r["valid_rows"] == 12 and r["problems"] == []


def test_errors_example_points_at_each_cell() -> None:
    parsed = parse(ERRORS)
    r = report(parsed)
    assert not r["ok"] and r["rows"] == 12
    assert cell_codes(ERRORS) == {
        (3, "age", "not_integer"),
        (4, "purpose", "out_of_domain"),
        (5, "duration", "out_of_range"),
        (6, "credit_amount", "missing"),
        (7, "housing", "out_of_domain"),
        (8, "credit_amount", "decimal_comma"),
        (9, "job", "out_of_domain"),
        (10, "installment_commitment", "not_integer"),
        (11, "duration", "outside_training"),  # a warning: the row is still scored
    }
    assert r["valid_rows"] == 4 and r["errors_total"] == 8
    assert r["column_warnings"] == [
        {"column": "personal_status", "code": "protected_column", "message": parsed.column_warnings[0]["message"]}
    ]
    housing = next(p for p in parsed.problems if p["column"] == "housing")
    assert "'own'" in housing["message"]  # tells what to type instead


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (b"", "empty"),
        (b"\n\n  \n", "empty"),
        (HEADER.encode() + b"\n", "no_rows"),
        (b"\x00\x01binary", "binary"),
        (csv_of(ROW).replace(b"own", "própria".encode("latin-1")), "encoding"),
        (csv_of(ROW.replace("skilled", '"skilled')), "malformed"),
        (csv_of(ROW, header=HEADER.replace("age", "idade")), "missing_columns"),
        (csv_of(ROW + ",x", header=HEADER + ",age"), "duplicate_column"),
        (csv_of(ROW + ",x", header=HEADER + ","), "empty_header"),
    ],
)
def test_file_level_errors(body: bytes, code: str) -> None:
    assert file_codes(body) == [code]
    assert report(parse(body))["valid_rows"] == 0


def test_missing_column_suggests_the_misspelled_one() -> None:
    parsed = parse(csv_of(ROW, header=HEADER.replace("age", "agee")))
    assert [e["code"] for e in parsed.file_errors] == ["missing_columns"]
    assert "age" in parsed.file_errors[0]["message"]
    assert parsed.column_warnings[0]["code"] == "unknown_column" and "'age'" in parsed.column_warnings[0]["message"]


def test_size_limit_is_checked_before_parsing() -> None:
    assert file_codes(VALID, Limits(max_bytes=100)) == ["too_large"]


def test_row_limit() -> None:
    limits = Limits(max_rows=3)
    assert file_codes(csv_of(*[ROW] * 3), limits) == []
    assert file_codes(csv_of(*[ROW] * 4), limits) == ["too_many_rows"]
    # A huge file stops being read early instead of being parsed to the end.
    big = parse(csv_of(*[ROW] * 5000), limits=Limits(max_rows=3, max_bytes=10**7))
    assert [e["code"] for e in big.file_errors] == ["too_many_rows"] and "3+" in big.file_errors[0]["message"]


def test_semicolon_and_bom_and_crlf() -> None:
    body = ("﻿" + HEADER.replace(",", ";") + "\r\n" + ROW.replace(",", ";") + "\r\n").encode()
    parsed = parse(body)
    assert parsed.delimiter == ";" and parsed.file_errors == [] and len(parsed.valid) == 1


def test_header_is_case_and_space_insensitive() -> None:
    header = ",".join(f" {c.upper()} " for c in FEATURES)
    assert len(parse(csv_of(ROW, header=header)).valid) == 1


def test_wrong_field_count_and_blank_lines_keep_file_line_numbers() -> None:
    body = csv_of(ROW, "", ROW + ",extra", ROW.rsplit(",", 1)[0], ROW)
    parsed = parse(body)
    assert [r.line for r in parsed.rows] == [2, 4, 5, 6]
    assert {(p["row"], p["code"]) for p in parsed.problems} == {(4, "wrong_field_count"), (5, "wrong_field_count")}
    assert [r.line for r in parsed.valid] == [2, 6]


def test_id_column_is_kept_and_capped() -> None:
    parsed = parse(csv_of("x" * 100 + "," + ROW, header="id," + HEADER))
    assert parsed.rows[0].id == "x" * 64 and len(parsed.valid) == 1


def test_problem_list_is_capped_but_counted() -> None:
    bad = ROW.replace("38", "7")
    r = report(parse(csv_of(*[bad] * 10)), Limits(max_listed=4))
    assert len(r["problems"]) == 4 and r["problems_total"] == 10 and r["truncated"]
    assert r["summary"] == [{"column": "age", "code": "out_of_range", "severity": "error", "count": 10}]


def test_scores_only_valid_rows_and_keeps_order() -> None:
    parsed = parse(ERRORS)
    results = score(FakeModel(), parsed, threshold=0.5)
    assert [r["line"] for r in results] == list(range(2, 14))
    scored = [r for r in results if r["status"] == "scored"]
    assert len(scored) == 4 and all(r["errors"] == [] for r in scored)
    first = results[0]
    assert first["default_probability"] == 0.5711 and first["decision"] == "review"
    invalid = next(r for r in results if r["line"] == 3)
    assert invalid["default_probability"] is None and invalid["errors"][0].startswith("age: ")


@pytest.mark.parametrize(
    ("value", "escaped"),
    [
        ("=1+1", "'=1+1"),
        ("+55 11", "'+55 11"),
        ("-2", "'-2"),
        ("@SUM(A1)", "'@SUM(A1)"),
        ("\t=1", "'\t=1"),
        ("\r=1", "'\r=1"),
        ("  =1", "'  =1"),
        ("radio/tv", "radio/tv"),
        ("<0", "<0"),
        ("", ""),
        (None, ""),
        (12, "12"),
    ],
)
def test_safe_cell(value: object, escaped: str) -> None:
    assert safe_cell(value) == escaped


def test_output_csv_neutralizes_formulas_from_the_input() -> None:
    evil = ROW.replace("skilled", '"=HYPERLINK(""http://evil"",""x"")"').replace("38", "-38")
    body = csv_of("=cmd|' /C calc'!A0," + ROW, "@x," + evil, header="id," + HEADER)
    parsed = parse(body)
    out = to_csv(parsed, score(FakeModel(), parsed, 0.5))
    rows = list(csv.reader(io.StringIO(out)))
    assert rows[0][:2] == ["line", "id"] and rows[0][-4:] == ["status", "default_probability", "decision", "errors"]
    for row in rows[1:]:
        for cell in row:
            assert not cell.lstrip(" ").startswith(("=", "+", "-", "@", "\t", "\r")), cell
    assert rows[1][1] == "'=cmd|' /C calc'!A0"
    assert rows[2][1] == "'@x" and rows[2][6] == "'-38" and rows[2][2 + FEATURES.index("job")].startswith("'=HYPERLINK")
    assert rows[1][-3:-1] == ["0.5711", "review"] and rows[2][-4] == "invalid"
