"""Map each transaction to its laundering typology using HI-Small_Patterns.txt (analysis only, never a feature)."""
import re

import numpy as np
import pandas as pd

KEY = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Paid", "Payment Currency",
       "Payment Format"]
COLUMNS = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Received",
           "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]


def parse_patterns(path):
    rows, typology = [], None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("BEGIN LAUNDERING ATTEMPT"):
            typology = re.match(r"BEGIN LAUNDERING ATTEMPT - ([A-Z-]+)", line).group(1)
        elif line.startswith("END LAUNDERING ATTEMPT"):
            typology = None
        elif line and typology:
            rows.append(line.split(",") + [typology])
    p = pd.DataFrame(rows, columns=COLUMNS + ["typology"])
    return _normalise(p)


def _normalise(df):
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M") if df["Timestamp"].dtype == object \
        else df["Timestamp"]
    for c in ("From Bank", "To Bank"):
        df[c] = df[c].astype(str).str.lstrip("0").replace("", "0")
    df["Amount Paid"] = df["Amount Paid"].astype(float).round(2)
    return df


def row_typology(raw, patterns_path):
    """Return an array (len(raw)) with the typology name, '' for normal rows, 'UNTYPED' for unmatched positives."""
    p = parse_patterns(patterns_path).drop_duplicates(KEY)
    left = _normalise(raw[KEY].assign(Account=raw["Account"].astype(str), **{"Account.1": raw["Account.1"].astype(str)}))
    left["_row"] = np.arange(len(raw))
    merged = left.merge(p[KEY + ["typology"]], on=KEY, how="left").drop_duplicates("_row").sort_values("_row")
    typ = merged["typology"].fillna("").to_numpy(dtype=object)
    positive = raw["Is Laundering"].to_numpy() == 1
    typ[~positive] = ""  # pattern lines are laundering by definition; guard against key collisions
    typ[positive & (typ == "")] = "UNTYPED"
    return typ
