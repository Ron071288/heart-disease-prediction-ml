from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Iterable

import pandas as pd


LINE_WIDTH = 88


def print_header(title: str) -> None:
    print("=" * LINE_WIDTH)
    print(title.center(LINE_WIDTH))
    print("=" * LINE_WIDTH)
    print()


def print_section(number: int, title: str) -> None:
    print(f"{number}. {title.upper()}")
    print("-" * LINE_WIDTH)


def print_key_values(rows: Iterable[tuple[str, str]]) -> None:
    for label, value in rows:
        print(f"{label:<30}: {value}")
    print()


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_metric_table(table: pd.DataFrame) -> None:
    rows = [
        (
            row["Model"],
            format_percent(row["Accuracy"]),
            format_percent(row["Precision"]),
            format_percent(row["Recall"]),
            format_percent(row["F1-score"]),
        )
        for row in table.to_dict("records")
    ]
    print_simple_table(["Model", "Accuracy", "Precision", "Recall", "F1-score"], rows)
    print()


def print_score_line(metrics: dict) -> None:
    rows = [
        ("Accuracy", format_percent(metrics["accuracy"])),
        ("Precision", format_percent(metrics["precision"])),
        ("Recall", format_percent(metrics["recall"])),
        ("F1-score", format_percent(metrics["f1_score"])),
    ]
    print("Scores:")
    print_simple_table(["Metric", "Value"], rows)
    print()


def print_confusion_matrix(metrics: dict) -> None:
    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]
    print("Confusion Matrix:")
    print_simple_table(
        ["Actual \\ Predicted", "No Disease", "Disease"],
        [
            ("No Disease", str(tn), str(fp)),
            ("Disease", str(fn), str(tp)),
        ],
    )
    print()
    print_simple_table(
        ["Term", "Meaning", "Value"],
        [
            ("TN", "Correct no heart disease", str(tn)),
            ("FP", "Predicted disease wrongly", str(fp)),
            ("FN", "Missed heart disease", str(fn)),
            ("TP", "Correct heart disease", str(tp)),
        ],
    )
    print()


def print_simple_table(headers: list[str], rows: Iterable[tuple[str, ...]]) -> None:
    table_rows = [tuple(str(cell) for cell in row) for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in table_rows))
        for index in range(len(headers))
    ]

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    header_row = "| " + " | ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    ) + " |"

    print(separator)
    print(header_row)
    print(separator)
    for row in table_rows:
        print("| " + " | ".join(
            row[index].ljust(widths[index]) for index in range(len(headers))
        ) + " |")
    print(separator)


def shorten_text(value: str, width: int = 56) -> str:
    return textwrap.shorten(str(value), width=width, placeholder="...")


def print_feature_effects(coefficients: pd.DataFrame, limit: int = 8) -> None:
    print("Positive coefficient = toward disease; negative coefficient = toward no disease.")
    print("The full coefficient list is saved in the artifacts folder.")
    print()
    rows = []
    for row_number, row in enumerate(coefficients.head(limit).itertuples(), start=1):
        direction = (
            "Toward disease"
            if row.Direction == "Toward heart disease"
            else "Toward no disease"
        )
        rows.append(
            (
                str(row_number),
                shorten_text(row.Feature, 56),
                direction,
                f"{row.Coefficient:.4f}",
            )
        )
    print_simple_table(["No.", "Feature", "Direction", "Coef."], rows)
    print()


def print_feature_importances(importances: pd.DataFrame, limit: int = 8) -> None:
    print("Sorted from strongest to weakest predictive influence.")
    print()
    rows = [
        (
            str(row_number),
            shorten_text(row.Feature, 56),
            f"{row.Importance:.4f}",
        )
        for row_number, row in enumerate(importances.head(limit).itertuples(), start=1)
    ]
    print_simple_table(["No.", "Feature", "Importance"], rows)
    print()


def print_output_location(output_dir: str | Path, files: Iterable[str]) -> None:
    output_path = Path(output_dir).resolve()
    print(f"{'Artifacts folder':<30}: {output_path}")
    print("Saved files".ljust(30) + ":")
    for filename in files:
        print(f"  - {filename}")
    print(f"{'Run prototype':<30}: python -m streamlit run app.py")
