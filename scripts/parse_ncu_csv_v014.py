from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def parse_number(value: str) -> int | float | str | None:
    text = value.strip().replace(",", "")
    if not text:
        return None
    if text in {"n/a", "N/A", "nan", "NaN"}:
        return text
    try:
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text)
        return float(text)
    except ValueError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--page", default="raw")
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    headers: list[list[str]] = []
    current_header: list[str] | None = None
    with args.csv_path.open(newline="", errors="replace") as handle:
        for raw_row in csv.reader(handle):
            row = [cell.strip() for cell in raw_row]
            if not row or all(not cell for cell in row):
                continue
            if row[0].startswith("==") or row[0].startswith("Connected"):
                records.append({"record_type": "message", "values": row})
                continue
            if any(
                cell in {"Kernel Name", "Metric Name", "ID", "Process ID", "Section Name"}
                for cell in row
            ):
                current_header = row
                headers.append(row)
                continue
            if current_header and len(row) == len(current_header):
                record: dict[str, Any] = {
                    "record_type": "row",
                    **dict(zip(current_header, row)),
                }
                for key in ("Metric Value", "ID", "Process ID", "Context", "Stream"):
                    if key in record:
                        record[f"{key} Parsed"] = parse_number(str(record[key]))
                records.append(record)
            else:
                records.append({"record_type": "unparsed", "values": row})

    metric_rows = [record for record in records if record.get("record_type") == "row"]
    kernels: dict[str, dict[str, Any]] = {}
    for record in metric_rows:
        kernel = str(record.get("Kernel Name", "unknown"))
        entry = kernels.setdefault(kernel, {"rows": 0, "sections": {}, "metrics": {}})
        entry["rows"] += 1
        section = str(record.get("Section Name", ""))
        if section:
            entry["sections"][section] = entry["sections"].get(section, 0) + 1
        metric = record.get("Metric Name")
        if metric:
            entry["metrics"][str(metric)] = {
                "value": record.get("Metric Value Parsed", record.get("Metric Value")),
                "unit": record.get("Metric Unit"),
            }

    payload: dict[str, Any] = {
        "version": "0.14.2",
        "page": args.page,
        "source_csv": str(args.csv_path),
        "headers": headers,
        "records": records,
        "kernel_summary": kernels,
        "metric_row_count": len(metric_rows),
    }
    if args.metadata and args.metadata.exists():
        payload["workload_metadata"] = json.loads(args.metadata.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps({
        "records": len(records), "metrics": len(metric_rows),
        "kernels": len(kernels), "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
