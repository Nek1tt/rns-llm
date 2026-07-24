from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({quote_identifier(table)})")
    ]


def safe_count(connection: sqlite3.Connection, table: str) -> int | None:
    try:
        return int(
            connection.execute(
                f"SELECT COUNT(*) FROM {quote_identifier(table)}"
            ).fetchone()[0]
        )
    except Exception:
        return None


def find_table(tables: list[str], required: tuple[str, ...], excluded: tuple[str, ...] = ()) -> str | None:
    for table in tables:
        upper = table.upper()
        if all(token in upper for token in required) and not any(token in upper for token in excluded):
            return table
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--queries-output", type=Path)
    args = parser.parse_args()

    connection = sqlite3.connect(args.sqlite_path)
    tables = [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]
    schema = {
        table: {"columns": table_columns(connection, table), "rows": safe_count(connection, table)}
        for table in tables
    }
    summaries: dict[str, Any] = {}
    executed_queries: list[dict[str, str]] = []

    string_table = "StringIds" if "StringIds" in tables else None
    kernel_table = find_table(tables, ("KERNEL",), ("ENUM", "VIEW"))
    if kernel_table:
        columns = schema[kernel_table]["columns"]
        duration = '(k."end" - k."start")' if "start" in columns and "end" in columns else "NULL"
        name_column = next(
            (column for column in ("demangledName", "shortName", "name") if column in columns),
            None,
        )
        query: str | None = None
        if name_column and string_table and name_column != "name":
            query = f'''SELECT COALESCE(s.value, CAST(k.{quote_identifier(name_column)} AS TEXT)) AS name,
COUNT(*) AS calls, SUM({duration}) AS total_ns, AVG({duration}) AS avg_ns,
MIN({duration}) AS min_ns, MAX({duration}) AS max_ns
FROM {quote_identifier(kernel_table)} AS k
LEFT JOIN {quote_identifier(string_table)} AS s ON s.id = k.{quote_identifier(name_column)}
GROUP BY name ORDER BY total_ns DESC'''
        elif name_column:
            query = f'''SELECT CAST(k.{quote_identifier(name_column)} AS TEXT) AS name,
COUNT(*) AS calls, SUM({duration}) AS total_ns, AVG({duration}) AS avg_ns,
MIN({duration}) AS min_ns, MAX({duration}) AS max_ns
FROM {quote_identifier(kernel_table)} AS k
GROUP BY name ORDER BY total_ns DESC'''
        if query:
            try:
                executed_queries.append({"name": "kernel_summary", "sql": query})
                names = ["name", "calls", "total_ns", "avg_ns", "min_ns", "max_ns"]
                summaries["kernels"] = [
                    dict(zip(names, row)) for row in connection.execute(query)
                ]
            except Exception as exc:
                summaries["kernel_query_error"] = repr(exc)

    runtime_table = find_table(tables, ("RUNTIME",), ("ENUM", "VIEW"))
    if runtime_table:
        columns = schema[runtime_table]["columns"]
        if "start" in columns and "end" in columns:
            query = f'''SELECT COUNT(*) AS calls,
SUM("end"-"start") AS total_ns, AVG("end"-"start") AS avg_ns,
MIN("end"-"start") AS min_ns, MAX("end"-"start") AS max_ns
FROM {quote_identifier(runtime_table)}'''
            try:
                executed_queries.append({"name": "cuda_runtime_summary", "sql": query})
                row = connection.execute(query).fetchone()
                summaries["runtime"] = dict(
                    zip(("calls", "total_ns", "avg_ns", "min_ns", "max_ns"), row)
                )
            except Exception as exc:
                summaries["runtime_query_error"] = repr(exc)

    memcpy_table = find_table(tables, ("MEMCPY",), ("ENUM", "VIEW"))
    if memcpy_table:
        columns = schema[memcpy_table]["columns"]
        if "start" in columns and "end" in columns:
            bytes_column = next((column for column in ("bytes", "size") if column in columns), None)
            bytes_expr = f"SUM({quote_identifier(bytes_column)})" if bytes_column else "NULL"
            query = f'''SELECT COUNT(*) AS calls, SUM("end"-"start") AS total_ns,
AVG("end"-"start") AS avg_ns, {bytes_expr} AS total_bytes
FROM {quote_identifier(memcpy_table)}'''
            try:
                executed_queries.append({"name": "memcpy_summary", "sql": query})
                row = connection.execute(query).fetchone()
                summaries["memcpy"] = dict(
                    zip(("calls", "total_ns", "avg_ns", "total_bytes"), row)
                )
            except Exception as exc:
                summaries["memcpy_query_error"] = repr(exc)

    nvtx_table = find_table(tables, ("NVTX", "EVENT"), ("ENUM", "VIEW"))
    if nvtx_table:
        summaries["nvtx_rows"] = safe_count(connection, nvtx_table)

    payload: dict[str, Any] = {
        "version": "0.14.2",
        "source_sqlite": str(args.sqlite_path),
        "tables": schema,
        "summaries": summaries,
        "executed_queries": executed_queries,
    }
    if args.metadata and args.metadata.exists():
        payload["workload_metadata"] = json.loads(args.metadata.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str))

    queries_output = args.queries_output or args.output.with_suffix(".sql")
    query_lines = [
        "-- Auto-generated Nsight Systems queries used for this report.",
        f"-- Source database: {args.sqlite_path}",
        "",
    ]
    for item in executed_queries:
        query_lines += [f"-- {item['name']}", item["sql"].rstrip(";") + ";", ""]
    queries_output.write_text("\n".join(query_lines))
    connection.close()
    print(json.dumps({
        "tables": len(tables), "queries": len(executed_queries),
        "output": str(args.output), "queries_output": str(queries_output),
    }, indent=2))


if __name__ == "__main__":
    main()
