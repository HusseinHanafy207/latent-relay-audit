"""Sender-transcript audit: explanations and arithmetic, not only RESULT labels."""

from __future__ import annotations

import inspect

from src.plan4_task import load_plan4_jsonl
from src.paths import DATA_DIR
from src.sender_audit import audit_sender_text
from src.stage_plan4_audit import PRESERVED, main


FAM001_WRITTEN = """
Carrier P: vehicle capacity 14 boxes; productive hours 5; loading rate 5 boxes per hour; refrigeration unit working; driver unavailable.
- productive hours: 5 (given)
- effective capacity: min(14, floor(5 * 5)) = min(14, 25) = 14
- eligibility: effective capacity (14) is less than shipment size (12) → eligible? No, because effective capacity must be at least shipment size. Refrigeration is present, but driver is unavailable → ineligible

Carrier Q: vehicle capacity 20 boxes; productive hours 5; loading rate 4 boxes per hour; refrigeration unit working; driver available.
- productive hours: 5 (given)
- effective capacity: min(20, floor(5 * 4)) = min(20, 20) = 20
- eligibility: effective capacity (20) is at least shipment size (12), refrigeration is present, driver is available → eligible

Carrier R: vehicle capacity 20 boxes; productive hours 1; loading rate 1 boxes per hour; refrigeration unit working; driver available.
- productive hours: 1 (given)
- effective capacity: min(20, floor(1 * 1)) = min(20, 1) = 1
- eligibility: effective capacity (1) is less than shipment size (12), refrigeration is present, driver is available → ineligible

Carrier S: vehicle capacity 14 boxes; productive hours 4; loading rate 5 boxes per hour; refrigeration unit working; driver available.
- productive hours: 4 (given)
- effective capacity: min(14, floor(4 * 5)) = min(14, 20) = 14
- eligibility: effective capacity (14) is at least shipment size (12), refrigeration is present, driver is available → eligible

RESULTS
P: q=14 | ineligible | effective capacity is less than shipment size
Q: q=20 | eligible | effective capacity is at least shipment size, refrigeration is present, driver is available
R: q=1 | ineligible | effective capacity is less than shipment size
S: q=14 | eligible | effective capacity is at least shipment size, refrigeration is present, driver is available
"""

FAM000_WRITTEN = """
**Carrier R:**
- productive hours = max(0, 5 - 1 - 0) = 4
- effective capacity = min(12, floor(4 * 4)) = min(12, 16) = 12
- eligibility: effective capacity (12) is at least shipment size (10), but refrigeration is not present when required. Ineligible.

**Carrier P:**
- productive hours = max(0, 5 - 1 - 1) = 3
- effective capacity = min(10, floor(4 * 3)) = min(10, 12) = 10
- eligibility: effective capacity (10) is at least shipment size (10), refrigeration is present, and driver is available. Eligible.

**Carrier S:**
- productive hours = max(0, 6 - 1 - 1) = 4
- effective capacity = min(18, floor(4 * 4)) = min(18, 16) = 16
- eligibility: effective capacity (16) is at least shipment size (10), refrigeration is present, and driver is available. Eligible.

**Carrier Q:**
- productive hours = max(0, 6 - 0 - 1) = 5
- effective capacity = min(14, floor(5 * 5)) = min(14, 25) = 14
- eligibility: effective capacity (14) is at least shipment size (10), refrigeration is present, but driver is unavailable. Ineligible.

**RESULTS**
P: q=10 | Eligible | effective capacity is at least shipment size, refrigeration is present, and driver is available
Q: q=14 | Ineligible | driver is unavailable
R: q=12 | Ineligible | refrigeration is not present when required
S: q=16 | Eligible | effective capacity is at least shipment size, refrigeration is present, and driver is available
"""


def _ex(example_id: str):
    return next(ex for ex in load_plan4_jsonl(DATA_DIR / "plan4_dev.jsonl") if ex.example_id == example_id)


def test_false_capacity_reason_is_incorrect_even_when_labels_match():
    ex = _ex("plan4_dev-fam-001-base")
    rec = audit_sender_text(FAM001_WRITTEN, ex)
    assert rec["all_q_correct"]
    assert rec["all_status_correct"]
    assert rec["verdict"] == "incorrect"
    codes = {issue["code"] for issue in rec["issues"]}
    assert "false_capacity_comparison" in codes
    assert "false_result_reason" in codes
    assert "false_eligibility_reason" in codes
    assert rec["used_b_outcome"] is False


def test_consistent_working_and_reasons_pass():
    ex = _ex("plan4_dev-fam-000-base")
    rec = audit_sender_text(FAM000_WRITTEN, ex)
    assert rec["verdict"] == "consistent"
    assert rec["n_incorrect"] == 0
    assert rec["all_q_correct"]
    assert rec["all_status_correct"]


def test_wrong_copied_capacity_is_incorrect():
    ex = _ex("plan4_dev-fam-012-a_flip")
    written = """
Carrier P: vehicle capacity 5 boxes; refrigeration unit working; driver available.
- effective capacity: 5 boxes (given vehicle capacity)
- eligibility: effective capacity (5) < shipment size (8) → ineligible
Carrier Q: vehicle capacity 8 boxes; refrigeration unit working; driver available.
- effective capacity: 8 boxes (given vehicle capacity)
- eligibility: effective capacity (8) ≥ shipment size (8), refrigeration is present, driver is available → eligible
Carrier R: vehicle capacity 11 boxes; refrigeration unit working; driver available.
- effective capacity: 11 boxes (given vehicle capacity)
- eligibility: effective capacity (11) ≥ shipment size (8), refrigeration is present, driver is available → eligible
Carrier S: vehicle capacity 11 boxes; refrigeration unit working; driver available.
- effective capacity: 11 boxes (given vehicle capacity)
- eligibility: effective capacity (11) ≥ shipment size (8), refrigeration is present, driver is available → eligible
RESULTS
P: q=5 | ineligible | effective capacity is less than shipment size
Q: q=8 | eligible | effective capacity is at least shipment size, refrigeration is present, driver is available
R: q=11 | eligible | effective capacity is at least shipment size, refrigeration is present, driver is available
S: q=11 | eligible | effective capacity is at least shipment size, refrigeration is present, driver is available
"""
    rec = audit_sender_text(written, ex)
    assert rec["verdict"] == "incorrect"
    codes = {issue["code"] for issue in rec["issues"]}
    assert "wrong_q" in codes
    assert "wrong_copied_fact" in codes


def test_subtraction_hours_are_checked_by_the_result_not_the_first_number():
    ex = _ex("plan4_dev-fam-014-base")
    written = """
**Carrier Q:**
- Productive hours: 4 - 1 - 0 = 3
- Effective capacity: min(10, floor(5 * 3)) = min(10, 15) = 10
- Eligibility: effective capacity (10) is at least shipment size (10), refrigeration is present, and driver is available → eligible
**Carrier P:**
- Productive hours: 4 - 1 - 0 = 3
- Effective capacity: min(12, floor(5 * 3)) = min(12, 15) = 12
- Eligibility: effective capacity (12) is at least shipment size (10), refrigeration is present, and driver is available → eligible
**Carrier S:**
- Productive hours: 4 - 3 - 1 = 0
- Effective capacity: min(10, floor(3 * 0)) = min(10, 0) = 0
- Eligibility: effective capacity (0) is not at least shipment size (10), refrigeration is present, and driver is available → ineligible
**Carrier R:**
- Productive hours: 4 - 1 - 0 = 3
- Effective capacity: min(10, floor(5 * 3)) = min(10, 15) = 10
- Eligibility: effective capacity (10) is at least shipment size (10), refrigeration is present, but driver is unavailable → ineligible
**RESULTS**
P: q=12 | eligible | effective capacity is at least shipment size, refrigeration is present, and driver is available
Q: q=10 | eligible | effective capacity is at least shipment size, refrigeration is present, and driver is available
R: q=10 | ineligible | driver is unavailable
S: q=0 | ineligible | effective capacity is not at least shipment size
"""
    rec = audit_sender_text(written, ex)
    assert rec["verdict"] == "consistent"


def test_rejected_at_least_question_is_not_an_asserted_comparison():
    ex = _ex("plan4_dev-fam-012-base")
    written = """
Carrier P: vehicle capacity 5 boxes; refrigeration unit working; driver available.
- effective capacity: 5 boxes (given vehicle capacity)
- eligibility: Check if effective capacity (5) is at least shipment size (8)? No. Since effective capacity is less than shipment size, carrier is ineligible.
Carrier R: vehicle capacity 8 boxes; refrigeration unit working; driver available.
- effective capacity: 8 boxes (given vehicle capacity)
- eligibility: Effective capacity (8) is at least shipment size (8). Refrigeration is present. Driver is available.
Carrier S: vehicle capacity 11 boxes; refrigeration unit working; driver available.
- effective capacity: 11 boxes (given vehicle capacity)
- eligibility: Effective capacity (11) is at least shipment size (8). Refrigeration is present. Driver is available.
Carrier Q: vehicle capacity 8 boxes; refrigeration unit working; driver unavailable.
- effective capacity: 8 boxes (given vehicle capacity)
- eligibility: Effective capacity (8) is at least shipment size (8). Refrigeration is present. Driver is unavailable.
RESULTS
P: q=5 | ineligible | effective capacity is less than shipment size
Q: q=8 | ineligible | driver unavailable
R: q=8 | eligible | all conditions met
S: q=11 | eligible | all conditions met
"""
    rec = audit_sender_text(written, ex)
    assert rec["verdict"] == "consistent"


def test_audit_source_does_not_read_b_or_gold_from_receiver():
    source = inspect.getsource(__import__("src.sender_audit", fromlist=["audit_sender_text"]))
    assert "plan_all_status_correct" not in source
    assert "pred_carrier" not in source
    assert "b_facts" not in source
    assert "used_b_outcome" in source


def test_audit_keeps_hold_out_and_does_not_regenerate(monkeypatch):
    source = inspect.getsource(__import__("src.stage_plan4_audit", fromlist=["main"]))
    assert "plan4_dev.jsonl" in source
    assert "generate_text_keep_cache" not in source
    assert any("evaluations.jsonl" in item for item in PRESERVED)
    assert "results_stage_plan4_hold" in PRESERVED
    monkeypatch.setattr("sys.argv", ["stage_plan4_audit", "--split", "plan4_hold"])
    assert main() == 2
