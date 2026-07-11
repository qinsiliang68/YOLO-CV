from __future__ import annotations

import re

import pandas as pd


ARM_ORDERS = (
    ("T", "R1", "R2"),
    ("R1", "R2", "T"),
    ("R2", "T", "R1"),
)


def order_triad_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if len(rows) != 3 or rows.triad_id.nunique() != 1 or set(rows.arm.astype(str)) != {"T", "R1", "R2"}:
        raise ValueError("Expected exactly one complete T/R1/R2 triad")
    triad_id = str(rows.triad_id.iloc[0])
    match = re.fullmatch(r"TRIAD_(\d{3})", triad_id)
    if not match:
        raise ValueError(f"Invalid triad id: {triad_id}")
    order = ARM_ORDERS[(int(match.group(1)) - 1) % len(ARM_ORDERS)]
    position = {arm: index for index, arm in enumerate(order)}
    return rows.assign(_arm_order=rows.arm.map(position)).sort_values("_arm_order").drop(columns="_arm_order")
