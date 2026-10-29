import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class BudgetTests(unittest.TestCase):
    def ledger(self):
        from llm_budget import BudgetLedger
        p = ROOT / "data/test_artifacts" / uuid.uuid4().hex / "budget.sqlite"
        p.parent.mkdir(parents=True)
        return BudgetLedger(p, limit_usd=.01, input_per_million=.1, output_per_million=.5)

    def test_reservations_prevent_concurrent_or_unresolved_calls_exceeding_cap(self):
        ledger = self.ledger()
        ledger.reserve(10000, 10000, "call1")  # $0.006
        with self.assertRaises(RuntimeError):
            ledger.reserve(10000, 10000, "call2")
        self.assertAlmostEqual(ledger.total(), .006)

    def test_actual_usage_releases_unused_reservation(self):
        ledger = self.ledger()
        call = ledger.reserve(10000, 10000, "call1")
        ledger.settle(call, 1000, 1000)  # $0.0006
        ledger.reserve(10000, 10000, "call2")
        self.assertAlmostEqual(ledger.total(), .0066)

    def test_reopening_does_not_reset_spending_or_increase_cap(self):
        from llm_budget import BudgetLedger
        ledger = self.ledger()
        ledger.reserve(10000, 10000, "call1")
        with self.assertRaises(ValueError):
            BudgetLedger(ledger.path, limit_usd=2., input_per_million=.1, output_per_million=.5)


if __name__ == "__main__":
    unittest.main()
