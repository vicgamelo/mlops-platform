"""Batch scoring from an uploaded CSV: parse, validate cell by cell, score the valid rows, write a CSV back.

The file is untrusted input. It is read with the stdlib ``csv`` module (no eval, no pandas type
inference), size and row count are capped before anything else, and every cell written back is
escaped so a spreadsheet does not run it as a formula (CSV injection).
"""

from __future__ import annotations

import csv
import difflib
import io
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .data import EXCLUDED, FEATURES, NUMERIC
from .schema import Lang, Problem, Value, check_value, errors


@dataclass(frozen=True)
class Limits:
    max_bytes: int = 50 * 1024
    max_rows: int = 250
    max_listed: int = 300  # problems listed one by one; the summary still counts all of them


ID_COLUMN = "id"

FILE_MESSAGES: dict[str, dict[Lang, str]] = {
    "too_large": {
        "pt": "arquivo grande demais: {size} bytes, o limite é {max} bytes",
        "en": "file too large: {size} bytes, the limit is {max} bytes",
    },
    "empty": {"pt": "arquivo vazio", "en": "empty file"},
    "encoding": {
        "pt": "o arquivo não está em UTF-8: salve de novo como 'CSV UTF-8'",
        "en": "the file is not UTF-8: save it again as 'CSV UTF-8'",
    },
    "binary": {"pt": "isso não parece um CSV (tem bytes nulos)", "en": "this does not look like a CSV (null bytes)"},
    "malformed": {
        "pt": "CSV malformado perto da linha {line}: {detail}",
        "en": "malformed CSV near line {line}: {detail}",
    },
    "no_rows": {"pt": "o arquivo só tem o cabeçalho", "en": "the file only has the header"},
    "too_many_rows": {
        "pt": "linhas demais: {rows}, o limite é {max} por arquivo",
        "en": "too many rows: {rows}, the limit is {max} per file",
    },
    "missing_columns": {
        "pt": "faltam colunas obrigatórias: {columns}",
        "en": "required columns are missing: {columns}",
    },
    "duplicate_column": {
        "pt": "coluna repetida no cabeçalho: {column}",
        "en": "column repeated in the header: {column}",
    },
    "empty_header": {
        "pt": "o cabeçalho tem uma coluna sem nome (posição {pos})",
        "en": "the header has a column with no name (position {pos})",
    },
}
COLUMN_MESSAGES: dict[str, dict[Lang, str]] = {
    "protected_column": {
        "pt": "coluna ignorada: atributo protegido que o modelo não usa",
        "en": "column ignored: protected attribute the model does not use",
    },
    "unknown_column": {
        "pt": "coluna ignorada: não é usada pelo modelo{hint}",
        "en": "column ignored: not used by the model{hint}",
    },
    "wrong_field_count": {
        "pt": "a linha tem {got} campos e o cabeçalho tem {expected}: confira vírgulas e aspas",
        "en": "the row has {got} fields and the header has {expected}: check commas and quotes",
    },
}
HINT: dict[Lang, str] = {"pt": " (era {0!r}?)", "en": " (did you mean {0!r}?)"}


@dataclass
class Row:
    line: int
    id: str | None
    raw: dict[str, str]
    clean: dict[str, Value] | None


@dataclass
class Parsed:
    delimiter: str
    columns: list[str]
    rows: list[Row]
    file_errors: list[dict[str, str]] = field(default_factory=list)
    column_warnings: list[dict[str, str]] = field(default_factory=list)
    problems: list[dict[str, Any]] = field(default_factory=list)

    @property
    def valid(self) -> list[Row]:
        return [r for r in self.rows if r.clean is not None]


def _file_error(parsed: Parsed, code: str, lang: Lang, **kw: object) -> Parsed:
    parsed.file_errors.append({"code": code, "message": FILE_MESSAGES[code][lang].format(**kw)})
    return parsed


def _sniff(first_line: str) -> str:
    return ";" if first_line.count(";") > first_line.count(",") else ","


def parse(body: bytes, lang: Lang = "pt", limits: Limits | None = None) -> Parsed:
    limits = limits or Limits()
    parsed = Parsed(delimiter=",", columns=[], rows=[])
    if len(body) > limits.max_bytes:
        return _file_error(parsed, "too_large", lang, size=len(body), max=limits.max_bytes)
    if b"\x00" in body:
        return _file_error(parsed, "binary", lang)
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        return _file_error(parsed, "encoding", lang)
    if not text.strip():
        return _file_error(parsed, "empty", lang)

    parsed.delimiter = _sniff(text.lstrip().split("\n", 1)[0])
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=parsed.delimiter, strict=True)
    cutoff = limits.max_rows + 1 + 50  # header + some blank lines: past this, stop reading
    stopped = False
    try:
        records = []
        for record in reader:
            records.append((reader.line_num, record))
            if len(records) > cutoff:
                stopped = True
                break
    except csv.Error as exc:
        return _file_error(parsed, "malformed", lang, line=reader.line_num, detail=str(exc))

    records = [(n, r) for n, r in records if any(c.strip() for c in r)]
    if not records:
        return _file_error(parsed, "empty", lang)
    _, header = records[0]
    columns = [h.strip().strip('"').lower() for h in header]
    parsed.columns = columns
    for pos, name in enumerate(columns, start=1):
        if not name:
            return _file_error(parsed, "empty_header", lang, pos=pos)
    dupes = [c for c, n in Counter(columns).items() if n > 1]
    if dupes:
        return _file_error(parsed, "duplicate_column", lang, column=dupes[0])
    missing = [c for c in FEATURES if c not in columns]
    for col in columns:
        if col in FEATURES or col == ID_COLUMN:
            continue
        if col in EXCLUDED:
            code, hint = "protected_column", ""
        else:
            close = difflib.get_close_matches(col, missing or FEATURES, n=1, cutoff=0.7)
            code, hint = "unknown_column", HINT[lang].format(close[0]) if close else ""
        parsed.column_warnings.append(
            {"column": col[:64], "code": code, "message": COLUMN_MESSAGES[code][lang].format(hint=hint)}
        )
    if missing:
        return _file_error(parsed, "missing_columns", lang, columns=", ".join(missing))

    data = records[1:]
    if not data:
        return _file_error(parsed, "no_rows", lang)
    if stopped or len(data) > limits.max_rows:
        shown = f"{limits.max_rows}+" if stopped else str(len(data))
        return _file_error(parsed, "too_many_rows", lang, rows=shown, max=limits.max_rows)

    index = {c: i for i, c in enumerate(columns)}
    for line, record in data:
        cells = {c: (record[i] if i < len(record) else "") for c, i in index.items()}
        row = Row(line, (cells.get(ID_COLUMN) or None), {c: cells[c] for c in FEATURES}, None)
        if row.id is not None:
            row.id = row.id[:64]
        parsed.rows.append(row)
        if len(record) != len(columns):
            msg = COLUMN_MESSAGES["wrong_field_count"][lang].format(got=len(record), expected=len(columns))
            problem = {"row": line, "column": "*", "code": "wrong_field_count", "message": msg}
            parsed.problems.append({**problem, "value": None, "severity": "error"})
            continue
        clean: dict[str, Value] = {}
        found: list[Problem] = []
        for col in FEATURES:
            value, problems = check_value(col, cells[col], lang)
            found.extend(p for p in problems if p.code != "type_coerced")  # CSV cells are always text
            if value is not None:
                clean[col] = value
        parsed.problems.extend({"row": line, **p.as_dict()} for p in found)
        if not errors(found):
            row.clean = clean
    return parsed


def report(parsed: Parsed, limits: Limits | None = None) -> dict[str, Any]:
    limits = limits or Limits()
    errs = [p for p in parsed.problems if p["severity"] == "error"]
    summary = Counter((p["column"], p["code"], p["severity"]) for p in parsed.problems)
    return {
        "ok": not parsed.file_errors and not errs,
        "delimiter": parsed.delimiter,
        "rows": len(parsed.rows),
        "valid_rows": len(parsed.valid),
        "invalid_rows": len(parsed.rows) - len(parsed.valid),
        "file_errors": parsed.file_errors,
        "column_warnings": parsed.column_warnings,
        "problems": parsed.problems[: limits.max_listed],
        "problems_total": len(parsed.problems),
        "errors_total": len(errs),
        "truncated": len(parsed.problems) > limits.max_listed,
        "summary": [
            {"column": c, "code": code, "severity": sev, "count": n}
            for (c, code, sev), n in sorted(summary.items(), key=lambda kv: -kv[1])
        ],
        "limits": {"max_bytes": limits.max_bytes, "max_rows": limits.max_rows},
    }


def score(model: Any, parsed: Parsed, threshold: float) -> list[dict[str, Any]]:
    """One result per row of the file, in order. Invalid rows are kept, with status ``invalid``."""
    valid = parsed.valid
    probs: list[float] = []
    if valid:
        frame = pd.DataFrame([r.clean for r in valid])[FEATURES]
        for col in NUMERIC:
            frame[col] = pd.to_numeric(frame[col])
        probs = [float(p) for p in model.predict_proba(frame)[:, 1]]
    by_line = {r.line: p for r, p in zip(valid, probs, strict=True)}
    errors_by_line: dict[int, list[str]] = {}
    for problem in parsed.problems:
        if problem["severity"] == "error":
            errors_by_line.setdefault(problem["row"], []).append(f"{problem['column']}: {problem['message']}")
    out = []
    for r in parsed.rows:
        p = by_line.get(r.line)
        out.append(
            {
                "line": r.line,
                "id": r.id,
                "status": "scored" if p is not None else "invalid",
                "default_probability": None if p is None else round(p, 4),
                "decision": None if p is None else ("review" if p >= threshold else "approve"),
                "errors": errors_by_line.get(r.line, []),
            }
        )
    return out


FORMULA_START = ("=", "+", "-", "@", "\t", "\r", "\n")


def safe_cell(value: object) -> str:
    """Neutralizes spreadsheet formulas (CSV injection, OWASP): a leading = + - @ (and tab/CR/LF, which
    some spreadsheets strip before deciding) gets a ' in front, so the cell is read as text."""
    text = "" if value is None else str(value)
    stripped = text.lstrip(" ")
    if stripped.startswith(FORMULA_START):
        return "'" + text
    return text


def to_csv(parsed: Parsed, results: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    has_id = ID_COLUMN in parsed.columns
    head = ["line", *([ID_COLUMN] if has_id else []), *FEATURES, "status", "default_probability", "decision", "errors"]
    writer.writerow(head)
    for row, res in zip(parsed.rows, results, strict=True):
        values: list[object] = [row.line]
        if has_id:
            values.append(row.id or "")
        values.extend(row.raw[c] for c in FEATURES)
        values.extend(
            [
                res["status"],
                "" if res["default_probability"] is None else f"{res['default_probability']:.4f}",
                res["decision"] or "",
                " | ".join(res["errors"]),
            ]
        )
        writer.writerow([safe_cell(v) for v in values])
    return buf.getvalue()
