"""Independently prepared generic instruction prefixes, reused per template."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from src.generic_select import (
    N_EXPECTED_TEMPLATES,
    causal_prefix_from_ids,
    group_generic_templates,
)
from src.plan4_task import a_state_key, load_plan4_jsonl
from src.prompts_plan4 import ELIGIBILITY_RULE, FRAMING_INSTRUCTION, b_reason_kv_instruct_prompt
from src.stage_plan4_generic import CONDS, FROZEN_PLACEMENT, INTERPRETATION_GUIDE, PRESERVED, SPEC, main
from src.stage_plan4_instr import EXTRA_SEP
from src.stage_plan4_kvswap import LAYOUT
from src.two_agent import TwoAgentInventory


REPO = Path(__file__).resolve().parents[1]


def test_causal_prefix_stops_before_private_facts():
    pieces = ["pre ", ELIGIBILITY_RULE, " later Shipment: 10 boxes"]
    ids = list(range(len(pieces)))

    def decode(seq):
        return "".join(pieces[int(i)] for i in seq)

    rec = causal_prefix_from_ids(ids, {"start": 1, "end": 2}, decode=decode)
    assert rec["n"] == 2
    assert rec["instruction_ids"] == [1]
    assert rec["contains_rule"]
    assert rec["private_facts"] is False
    try:
        causal_prefix_from_ids(ids, {"start": 1, "end": 3}, decode=decode)
        assert False, "expected shipment facts to fail"
    except RuntimeError as exc:
        assert "shipment" in str(exc).lower() or "carrier" in str(exc).lower()


def test_saved_dev_has_three_reusable_templates():
    cons = json.loads((REPO / "results_stage_plan4_instr" / "construction.json").read_text(encoding="utf-8"))
    arts = {}
    for line in (REPO / "results_stage_plan4_instr" / "a_artifacts.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        arts[rec["example_id"]] = rec
    examples = {a_state_key(ex): ex for ex in load_plan4_jsonl(REPO / "data" / "plan4_dev.jsonl")}
    records = []
    for key, pack in cons["by_a_state"].items():
        prefix = causal_prefix_from_ids(arts[key]["full_ids"], pack["instruction"])
        prefix["a_state_key"] = key
        prefix["difficulty"] = examples[key].difficulty
        records.append(prefix)
        assert pack["instruction_before_facts"]
    templates = group_generic_templates(records)
    assert len(templates) == N_EXPECTED_TEMPLATES == 3
    assert set(templates) == {"L1", "L2", "L3"}
    for name, rec in templates.items():
        assert rec["n"] == 20
        assert rec["instruction_end"] - rec["instruction_start"] == 64
        assert rec["difficulties"] == [name]


def test_three_conditions_reuse_r_and_keep_hold_out(monkeypatch):
    assert LAYOUT == "sep"
    assert EXTRA_SEP is True
    assert CONDS == ("instr_kv", "instr_fresh", "instr_generic")
    assert SPEC["instr_generic"]["instruction"] == "generic"
    assert "independently" in FROZEN_PLACEMENT or "independent" in FROZEN_PLACEMENT or "once per" in FROZEN_PLACEMENT
    assert "reuse" in INTERPRETATION_GUIDE or "reused" in INTERPRETATION_GUIDE
    source = inspect.getsource(__import__("src.stage_plan4_generic", fromlist=["main"]))
    assert "plan4_dev.jsonl" in source
    assert "plan4_hold.jsonl" in source
    assert "do_not_load" in source
    assert "generate_text_keep_cache" not in source
    assert "results_stage_plan4_ifresh" in PRESERVED
    assert "results_stage_plan4_hold" in PRESERVED
    assert "concat_gathered_block" in source
    assert "insert_after_framing" not in source
    assert FRAMING_INSTRUCTION
    mix_src = inspect.getsource(TwoAgentInventory.concat_gathered_block)
    assert "rerotate_cache_keys" in mix_src
    from src.plan4_task import N_PLAN4_FAMILIES, PLAN4_DEV_SEED, generate_split

    ex = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)[0]
    assert FRAMING_INSTRUCTION in b_reason_kv_instruct_prompt(ex)
    from src.stage_plan4_generic import _paths

    data, out_root, _kaggle = _paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_generic" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_generic", "--split", "plan4_hold"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_generic", "--split", "plan4_fresh"])
    assert main() == 2
