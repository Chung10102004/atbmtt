"""Persistent reservations for a shared API budget, including ambiguous failures."""
import math
import sqlite3
from pathlib import Path


class BudgetLedger:
    def __init__(self, path, limit_usd, input_per_million, output_per_million):
        self.path = Path(path)
        self.limit = float(limit_usd)
        self.input_rate, self.output_rate = float(input_per_million), float(output_per_million)
        if any(not math.isfinite(x) or x <= 0 for x in (self.limit, self.input_rate, self.output_rate)):
            raise ValueError("Budget and token rates must be positive and finite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute("CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, cap REAL, input_rate REAL, output_rate REAL)")
            db.execute("CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, tag TEXT, usd REAL, status TEXT, prompt_tokens INTEGER, completion_tokens INTEGER)")
            db.execute("INSERT OR IGNORE INTO settings VALUES (1, ?, ?, ?)", (self.limit, self.input_rate, self.output_rate))
            saved = db.execute("SELECT cap, input_rate, output_rate FROM settings WHERE id=1").fetchone()
            if saved != (self.limit, self.input_rate, self.output_rate):
                raise ValueError("Existing ledger uses a different budget or token price; refusing to reset it")

    def cost(self, prompt_tokens, completion_tokens):
        if min(prompt_tokens, completion_tokens) < 0:
            raise ValueError("Token counts cannot be negative")
        return (prompt_tokens * self.input_rate + completion_tokens * self.output_rate) / 1e6

    def total(self):
        with sqlite3.connect(self.path, timeout=30) as db:
            return db.execute("SELECT COALESCE(SUM(usd), 0) FROM calls").fetchone()[0]

    def reserve(self, prompt_token_bound, completion_token_bound, tag):
        amount = self.cost(prompt_token_bound, completion_token_bound)
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute("BEGIN IMMEDIATE")
            total = db.execute("SELECT COALESCE(SUM(usd), 0) FROM calls").fetchone()[0]
            if total + amount > self.limit + 1e-12:
                raise RuntimeError(f"API budget reached: ${total:.4f} committed; next reservation ${amount:.4f}, cap ${self.limit:.2f}")
            cursor = db.execute("INSERT INTO calls(tag, usd, status) VALUES (?, ?, 'reserved')", (tag, amount))
            return cursor.lastrowid

    def settle(self, call_id, prompt_tokens, completion_tokens):
        amount = self.cost(prompt_tokens, completion_tokens)
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT usd,status,prompt_tokens,completion_tokens FROM calls WHERE id=?", (call_id,)).fetchone()
            if row is not None and row[1] == "settled" and row[2:] == (int(prompt_tokens), int(completion_tokens)):
                return
            if row is None or row[1] != "reserved":
                raise ValueError("Call reservation does not exist or is already settled")
            db.execute("UPDATE calls SET usd=?,status='settled',prompt_tokens=?,completion_tokens=? WHERE id=?",
                       (amount, int(prompt_tokens), int(completion_tokens), call_id))
        if amount > row[0] + 1e-12:
            raise RuntimeError("Provider usage exceeded the reserved bound; stop before further API calls")

    def summary(self):
        with sqlite3.connect(self.path, timeout=30) as db:
            rows = db.execute("SELECT status,COUNT(*),SUM(usd),COALESCE(SUM(prompt_tokens),0),COALESCE(SUM(completion_tokens),0) FROM calls GROUP BY status").fetchall()
        return {"limit_usd": self.limit, "committed_usd": self.total(),
                "input_usd_per_million": self.input_rate, "output_usd_per_million": self.output_rate,
                "calls_by_status": {r[0]: {"calls": r[1], "usd": r[2], "prompt_tokens": r[3], "completion_tokens": r[4]} for r in rows}}
