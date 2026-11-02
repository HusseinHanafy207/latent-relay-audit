"""Receiver query-adapter pilot (twelve carriers).

Question: can B learn to use the unchanged transferred result states, without
rebuilding them or receiving attention from a reference model?

This applies Attuner's query-side repair (query-projection adapters that read a
frozen cache, no reference at inference) to the selected-result setting. The
earlier fresh-key intervention (63 -> 94 labels with transferred values kept)
motivates trying a change to how B reads the cache; it does not show that a
query adapter can reproduce that improvement.

Phases:
- smoke:  implementation check on training sheets only. A zero-initialised
          adapter must reproduce the unadapted model; only adapter parameters
          receive gradients; the loss is finite; received caches are unchanged;
          the teacher's plan is absent from the plan-pass input. The adapter and
          optimizer are reset afterwards and nothing is saved.
- train:  the registered pilot on the 96 training base sheets, 3 epochs, fixed
          hyperparameters; checkpoints at the end of each epoch only.
- select: adapted tk_tv on the 12 dev base sheets for each checkpoint, plus
          unadapted tk_tv; highest labels wins, ties go to the earlier epoch.
- gates:  the registered development gates with the selected checkpoint:
          improvement over unadapted KV, worse without results, movement
          toward the partner's labels under mismatched results, and A-change
          pairs. Untouched (eval) families are not touched here.
- freeze, eval_write, eval_check, eval: only after every gate passed. The
          selected checkpoint is hashed and frozen first; eval A sheets are then
          written with no adapter attached, checked, and every condition (plus
          A's full cache as a reference) runs once on the untouched families.

The nine timing12 cases are never trained on or used for selection.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cache_utils import (
    cache_slice_delta,
    cache_to_device,
    clone_cache_blob,
    concat_cache_blobs,
    n_kv_positions,
)
from src.manifest import load_pinned
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_qadapt_task import (
    QADAPT_SPLITS,
    changed_carrier,
    ensure_split,
    file_sha256,
    mismatch_partner,
    split_path,
)
from src.plan4_task import a_state_key
from src.plan4_timing_task import gold_eligibility_statuses_n, load_timing12_jsonl
from src.precision import peak_gpu_memory_gb, reset_peak_memory
from src.prompts_plan4 import ordinary_sender_prompt, prompt_contains_a_private, prompt_contains_b_private
from src.prompts_plan4_timing import encode_prompt_for, written_result_handoff_n
from src.qadapt import (
    ALPHA,
    RANK,
    QueryAdapter,
    argmax_agreement,
    attention_implementation,
    current_attention,
    grad_report,
    kl_sum,
    logit_delta,
    response_logits,
)
from src.span_select import caches_equal, gather_cache_positions
from src.stage_plan0 import B_ANSWER_MAX_NEW, _load_messages
from src.stage_plan4_kvmix12 import _append, _prepare_runner, _twelve_choice, _twelve_labels
from src.stage_plan4_qadapt_prep import _setup as _prep_setup
from src.stage_plan4_qadapt_prep import _state as _prep_state
from src.stage_plan4_qadapt_prep import _a_flip_sender, _a_record, _support_ids, _support_vs_ref
from src.stage_plan4_query12 import BOOT_SEED, N_BOOT, _cluster_delta, _copy_first
from src.stage_plan4_recon12 import _run_kv_n
from src.stage_plan4_timing import (
    A_WRITTEN_MAX_NEW_12,
    B_PLAN_MAX_NEW_12,
    _prompts_for,
    _run_kv_b_timed,
    _run_text_b_timed,
    _write_artifacts12,
    prompt_contains_solver_labels_n,
)
from src.stage_select import _load_rows

STUDENT = "tk_tv"
TEACHER = "fk_fv"
TRAIN_ATTN = "sdpa"
GEN_ATTN = "eager"
N_SMOKE = 4
SMOKE_STEPS = 3
SHEETS_PER_STEP = 4
LR = 2e-4
BETAS = (0.9, 0.999)
CLIP = 1.0
INIT_SCALE = 1024.0
MIN_ARGMAX_AGREE_TEACHER = 0.95
MIN_ARGMAX_AGREE_KERNELS = 0.98
PLAN_LINE_MIN_CHARS = 20
PRESERVED = (
    "results_stage_plan4_qadapt_prep",
    "results_stage_plan4_publicprefix12",
    "results_stage_plan4_timing",
    "results_stage_plan4_recon12",
    "results_stage_plan4_query12",
    "results_stage_plan4_kvmix12",
    "results_stage_plan4_refattn12",
    "results_stage_plan4_refgen12",
    "results_stage_plan4_refval12",
    "results_stage_plan4_refgenval12",
    "results_stage_plan4_nbound",
    "results_stage_plan4_confirm",
    "results_stage_plan4_hold",
)
DO_NOT_LOAD = (
    "data/plan4_dev.jsonl",
    "data/plan_dev.jsonl",
    "data/plan4_fresh.jsonl",
    "data/plan4_hold.jsonl",
    "data/test.jsonl",
)
PREP_INPUTS = {
    "a_artifacts_train.jsonl": "prep_a_artifacts_train.jsonl",
    "a_artifacts_dev.jsonl": "prep_a_artifacts_dev.jsonl",
    "teacher_train.jsonl": "prep_teacher_train.jsonl",
    "check_summary.json": "prep_check_summary.json",
    "timing12_a_artifacts_12.jsonl": "prep_timing12_a_artifacts_12.jsonl",
}
EPOCHS = 3
ORDER_SEED_BASE = 2026
GATE_MIN_GAIN = 12
GATE_MIN_R_DEPENDENCE = 12
GATE_MAX_EXTRA_MALFORMED = 1
CHECKPOINT = "adapter_epoch{}.pt"
TRAIN_STATE = "train_state.pt"
PERSIST_NAMES = (
    "manifest.json",
    "smoke_check.json",
    "train_log.jsonl",
    "train_summary.json",
    TRAIN_STATE,
    *(CHECKPOINT.format(e) for e in range(1, EPOCHS + 1)),
    "select_evaluations.jsonl",
    "selection.json",
    "dev_construction.json",
    "dev_evaluations.jsonl",
    "dev_summary.json",
    "frozen_candidate.json",
    "a_artifacts_eval.jsonl",
    "eval_construction.json",
    "eval_evaluations.jsonl",
    "eval_summary.json",
) + tuple(PREP_INPUTS.values())
FROZEN = "frozen_candidate.json"
EVAL_ARTIFACTS = "a_artifacts_eval.jsonl"
EVAL_PHASES = ("freeze", "eval_write", "eval_check", "eval")
KV_MIB_PER_POSITION = 0.140625
DEV_CONDS = {
    "adapted_correct_r": {"cache": "correct_r", "adapted": True, "variants": ("base", "a_flip")},
    "unadapted_correct_r": {"cache": "correct_r", "adapted": False, "variants": ("base", "a_flip")},
    "adapted_no_r": {"cache": "no_r", "adapted": True, "variants": ("base",)},
    "unadapted_no_r": {"cache": "no_r", "adapted": False, "variants": ("base",)},
    "adapted_mismatched_r": {"cache": "mismatched_r", "adapted": True, "variants": ("base",)},
    "unadapted_mismatched_r": {"cache": "mismatched_r", "adapted": False, "variants": ("base",)},
    "fk_fv": {"cache": "fk_fv", "adapted": False, "variants": ("base", "a_flip")},
    "text_result": {"cache": None, "adapted": False, "variants": ("base", "a_flip")},
}
EVAL_CONDS = {
    **DEV_CONDS,
    "full_kv": {"cache": "full_kv", "adapted": False, "variants": ("base", "a_flip")},
}
EVAL_PAIRED = (
    ("adapted_vs_unadapted", "adapted_correct_r", "unadapted_correct_r"),
    ("adapted_vs_reconstruction", "adapted_correct_r", "fk_fv"),
    ("adapted_vs_full_kv", "adapted_correct_r", "full_kv"),
    ("adapted_vs_text", "adapted_correct_r", "text_result"),
    ("adapted_r_vs_no_r", "adapted_correct_r", "adapted_no_r"),
)
COST_NOTE_QADAPT = (
    "Costs are measured, not claimed as an advantage. The adapted receiver reads the same "
    "compact transferred cache as unadapted KV; its extra cost is the adapter weights, the "
    "unmerged LoRA matmuls during prefill and decode, and the one-time teacher and training "
    "runs. The sender must run without the adapter. Reconstruction (fk_fv) needs the support "
    "and results recomputed on B's side; full KV ships A's whole cache; text sends only the "
    "result lines. A reliability gain does not by itself establish a cost advantage."
)
GATE_NOTE = (
    "Development gates (registered before training). Improvement alone could come "
    "from learned task shortcuts, so the adapter must also depend on the message: "
    "worse without results, labels moving toward the partner's under mismatched "
    "results, and A-change pairs handled at least as well as unadapted KV."
)
SMOKE_NOTE = (
    "Implementation check on training sheets only. The adapter trained here is "
    "discarded (reset afterwards, never saved). Its losses are not pilot results."
)
KERNEL_NOTE = (
    "Training forwards (teacher and student) use memory-efficient attention because "
    "eager attention keeps every layer's attention matrix for the backward pass, "
    "which does not fit on a T4. Generation and evaluation stay on eager attention, "
    "as in every earlier run; agreement between the two kernels is reported."
)


def _paths():
    out_root = RESULTS_DIR / "stage_plan4_qadapt" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_qadapt")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return out_root, kaggle_dest


def _persist(out_root: Path, extra_dest: Path | None = None) -> None:
    dests = [out_root.parent / "latest"]
    if extra_dest is not None:
        dests.append(extra_dest)
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        for name in PERSIST_NAMES:
            src, out = out_root / name, dest / name
            if not src.is_file() or src.resolve() == out.resolve():
                continue
            if out.is_file() and out.stat().st_size == src.stat().st_size and out.stat().st_mtime >= src.stat().st_mtime:
                continue
            out.write_bytes(src.read_bytes())


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _copy_prep(out_root: Path) -> dict:
    dirs = [REPO_ROOT / "results_stage_plan4_qadapt_prep", Path("/kaggle/working/results_stage_plan4_qadapt_prep")]
    return {
        name: _copy_first(out_root, local, [d / name for d in dirs], what=f"prep {name}")
        for name, local in PREP_INPUTS.items()
    }


def registered_hash_mismatches(protocol: dict) -> list[str]:
    data = (protocol or {}).get("data") or {}
    return [name for name in QADAPT_SPLITS if name in data and data[name].get("sha256") != file_sha256(split_path(name))]


# ---------------------------------------------------------------- targets


def plan_leak_check(plan_rendered: str, plan_text: str, *, min_chars: int = PLAN_LINE_MIN_CHARS) -> dict:
    """The teacher's plan must not appear in the plan-pass input."""
    lines = [ln.strip() for ln in (plan_text or "").splitlines() if len(ln.strip()) >= min_chars]
    leaked = [ln[:60] for ln in lines if ln in plan_rendered]
    whole = bool(plan_text.strip()) and plan_text.strip() in plan_rendered
    return {"whole_plan_in_input": whole, "plan_lines_in_input": leaked, "ok": not whole and not leaked}


def _sequence(runner, prompt: str, text: str, *, truncated: bool) -> dict:
    tok = runner.wrapper.tokenizer
    rendered, input_ids, _mask = runner._encode(prompt)
    body = [int(t) for t in tok(text, add_special_tokens=False)["input_ids"]]
    eos = tok.eos_token_id
    response = body + ([] if truncated or eos is None else [int(eos)])
    return {
        "prompt": prompt,
        "rendered": rendered,
        "prompt_ids": [int(t) for t in input_ids.view(-1).tolist()],
        "response_ids": response,
        "text": text,
        "ends_with_eos": not truncated,
        "roundtrip_equal": tok.decode(body, skip_special_tokens=True).strip() == (text or "").strip(),
    }


def sheet_targets(runner, ex, teacher_row: dict) -> dict:
    """Plan and answer sequences built from the saved teacher texts (re-tokenized)."""
    prompts = _prompts_for(ex)
    plan_prompt = prompts["kv_plan"]
    plan_text = teacher_row["plan_text"] or ""
    answer_prompt = prompts["kv_answer"](ex, plan_text)
    plan = _sequence(runner, plan_prompt, plan_text, truncated=bool(teacher_row.get("plan_truncated")))
    answer = _sequence(runner, answer_prompt, teacher_row.get("raw_text") or "", truncated=bool(teacher_row.get("truncated")))
    checks = {
        "plan_prompt_has_no_a_private": not prompt_contains_a_private(plan_prompt, ex),
        "answer_prompt_has_no_a_private": not prompt_contains_a_private(answer_prompt, ex),
        "plan_not_in_plan_input": plan_leak_check(plan["rendered"], plan_text),
        "answer_prompt_carries_teacher_plan": bool(plan_text.strip()) and plan_text.strip() in answer["rendered"],
        "answer_prompt_equals_teacher_generation": answer_prompt == teacher_row.get("prompt"),
        "plan_roundtrip_equal": plan["roundtrip_equal"],
        "answer_roundtrip_equal": answer["roundtrip_equal"],
        "n_plan_response": len(plan["response_ids"]),
        "n_answer_response": len(answer["response_ids"]),
    }
    checks["ok"] = bool(
        checks["plan_prompt_has_no_a_private"]
        and checks["answer_prompt_has_no_a_private"]
        and checks["plan_not_in_plan_input"]["ok"]
        and checks["answer_prompt_carries_teacher_plan"]
        and checks["answer_prompt_equals_teacher_generation"]
        and checks["n_plan_response"] > 0
        and checks["n_answer_response"] > 0
    )
    return {"plan": plan, "answer": answer, "checks": checks}


# ---------------------------------------------------------------- forwards


def _adapter(runner) -> QueryAdapter:
    """One adapter per loaded model, reset to its initial state for each phase."""
    adapter = getattr(runner, "_query_adapter", None)
    if adapter is None:
        adapter = QueryAdapter(runner.wrapper.model, rank=RANK, alpha=ALPHA)
        runner._query_adapter = adapter
    adapter.reset()
    adapter.switch["on"] = False
    return adapter


def _cache_maker(runner, bundle):
    return lambda: cache_to_device(clone_cache_blob(bundle.cache_cpu), runner.device)


def _logits(runner, bundle, seq: dict) -> torch.Tensor:
    return response_logits(
        runner.wrapper.model,
        _cache_maker(runner, bundle),
        prompt_ids=seq["prompt_ids"],
        response_ids=seq["response_ids"],
        pos_cursor=int(bundle.pos_cursor_cpu.view(-1)[0].item()),
        past_mask=bundle.past_mask_cpu,
        device=runner.device,
    )


def teacher_logits(runner, adapter, state, seq: dict, *, impl: str = TRAIN_ATTN) -> torch.Tensor:
    model = runner.wrapper.model
    with attention_implementation(model, impl), adapter.off(), torch.no_grad():
        return _logits(runner, state["kvmix"][TEACHER], seq).detach()


def student_logits(runner, adapter, state, seq: dict, *, on: bool, impl: str = TRAIN_ATTN, grad: bool = False):
    model = runner.wrapper.model
    switch = adapter.on() if on else adapter.off()
    with attention_implementation(model, impl), switch, torch.set_grad_enabled(grad):
        return _logits(runner, state["kvmix"][STUDENT], seq)


def sheet_backward(runner, adapter, state, targets: dict, scaler, *, weight: float = 1.0) -> dict:
    """Token-mean KL over the plan and answer response tokens; each part backpropagates on its own."""
    total = len(targets["plan"]["response_ids"]) + len(targets["answer"]["response_ids"])
    out = {"n_tokens": total}
    kl_total = 0.0
    for part in ("plan", "answer"):
        seq = targets[part]
        t = teacher_logits(runner, adapter, state, seq)
        s = student_logits(runner, adapter, state, seq, on=True, grad=True)
        kl = kl_sum(t, s)
        loss = kl * (weight / total)
        scaler.scale(loss).backward()
        kl_val = float(kl.detach())
        kl_total += kl_val
        out[part] = {
            "kl_sum": kl_val,
            "n_tokens": len(seq["response_ids"]),
            "finite": bool(torch.isfinite(kl.detach())),
            "teacher_argmax_agree": argmax_agreement(t, seq["response_ids"]),
            "student_argmax_agree": argmax_agreement(s.detach(), seq["response_ids"]),
        }
        del t, s, kl, loss
    out["loss"] = kl_total / total
    out["finite"] = all(out[p]["finite"] for p in ("plan", "answer"))
    return out


def make_optimizer(adapter):
    return torch.optim.AdamW(adapter.parameters(), lr=LR, betas=BETAS, weight_decay=0.0)


def make_scaler(device):
    enabled = getattr(device, "type", str(device)) == "cuda"
    try:
        return torch.amp.GradScaler("cuda", init_scale=INIT_SCALE, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(init_scale=INIT_SCALE, enabled=enabled)


def optimizer_step(adapter, opt, scaler, *, inspect=None) -> dict:
    scaler.unscale_(opt)
    report = inspect() if inspect is not None else None
    norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), CLIP)
    before = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
    scaler.step(opt)
    scaler.update()
    after = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
    opt.zero_grad(set_to_none=True)
    return {
        "grad_norm": float(norm),
        "grad_norm_finite": bool(torch.isfinite(torch.as_tensor(norm))),
        "skipped": after < before,
        "scale_before": before,
        "scale_after": after,
        "grads": report,
    }


# ---------------------------------------------------------------- smoke


def _snap(state) -> dict:
    return {c: clone_cache_blob(state["kvmix"][c].cache_cpu) for c in (STUDENT, TEACHER)} | {
        "a_full": clone_cache_blob(state["bundle"].cache_cpu)
    }


def _held(state, snap) -> dict:
    return {
        STUDENT: bool(caches_equal(snap[STUDENT], state["kvmix"][STUDENT].cache_cpu)),
        TEACHER: bool(caches_equal(snap[TEACHER], state["kvmix"][TEACHER].cache_cpu)),
        "a_full": bool(caches_equal(snap["a_full"], state["bundle"].cache_cpu)),
    }


def _generation_pair(runner, adapter, ex, state, args) -> dict:
    rows = {}
    for name, ctx in (("adapter_off", adapter.off), ("adapter_on_zero", adapter.on)):
        with ctx():
            rec = _run_kv_n(
                runner,
                ex,
                runner.clone_sender(state["kvmix"][STUDENT]),
                plan_max_new=args.plan_max_new_12,
                answer_max_new=args.max_new_tokens,
            )
        rows[name] = rec
    off, on = rows["adapter_off"], rows["adapter_on_zero"]
    return {
        "plan_text_equal": off.get("plan_text") == on.get("plan_text"),
        "answer_text_equal": off.get("raw_text") == on.get("raw_text"),
        "labels": {k: v.get("plan_n_status_correct") for k, v in rows.items()},
        "texts": {k: {"plan_text": v.get("plan_text"), "raw_text": v.get("raw_text")} for k, v in rows.items()},
    }


def smoke_passed(report: dict) -> dict:
    gates = {
        "param_count_matches_registered": report["adapter"]["n_params"] == report["adapter"]["registered_params"],
        "only_q_proj_wrapped": report["adapter"]["n_modules"] == report["adapter"]["n_layers"],
        "targets_ok": all(s["checks"]["ok"] for s in report["sheets"]),
        "teacher_targets_align": all(
            s["teacher_alignment"][p] >= MIN_ARGMAX_AGREE_TEACHER for s in report["sheets"] for p in ("plan", "answer")
        ),
        "zero_init_logits_identical": all(v["equal"] for v in report["zero_init"]["logits"].values()),
        "zero_init_generation_identical": all(
            g["plan_text_equal"] and g["answer_text_equal"] for g in report["zero_init"]["generation"].values()
        ),
        "kernel_agreement": all(v["argmax_agree"] >= MIN_ARGMAX_AGREE_KERNELS for v in report["kernels"].values()),
        "attention_restored_to_eager": report["attention_after"] == GEN_ATTN,
        "loss_finite": all(m["finite"] for m in report["training"]["micro_steps"]),
        "no_base_gradients": all(
            st["grads"]["n_base_params_with_grad_or_trainable"] == 0 for st in report["training"]["steps"]
        ),
        "first_step_grads": report["training"]["first_step_grads_ok"],
        "later_step_grads": report["training"]["later_step_grads_ok"],
        "some_step_applied": any(not st["skipped"] for st in report["training"]["steps"]),
        "adapter_changes_outputs_after_training": not report["after_training"]["logits_on_vs_off"]["equal"],
        "received_caches_unchanged": all(all(v.values()) for v in report["after_training"]["caches_held"].values()),
        "construction_unaffected_by_adapter": report["after_training"]["rebuilt_state_equal"],
        "reset_restores_init": report["reset"]["matches_init"] and report["reset"]["b_all_zero"],
        "reset_logits_identical": report["reset"]["logits_on_vs_off"]["equal"],
        "reset_optimizer_empty": report["reset"]["optimizer_state_empty"],
    }
    return {"gates": gates, "all_passed": all(gates.values())}


def _run_smoke(args, out_root, kaggle_dest, splits, runner, inputs, protocol) -> int:
    model = runner.wrapper.model
    train_base = [ex for ex in splits["train"] if ex.variant == "base"]
    artifacts = _load_messages(inputs["a_artifacts_train.jsonl"])
    frozen = _load_messages(inputs["timing12_a_artifacts_12.jsonl"])
    timing12 = load_timing12_jsonl(DATA_DIR / "plan4_timing12.jsonl")
    teacher_rows = {
        r["example_id"]: r for r in _load_rows(inputs["teacher_train.jsonl"]) if r["condition"] == TEACHER
    }
    saved_student = {
        r["example_id"]: r for r in _load_rows(inputs["teacher_train.jsonl"]) if r["condition"] == STUDENT
    }
    sheets = train_base[:N_SMOKE]
    if any(ex.example_id not in teacher_rows for ex in sheets):
        print("Teacher targets are missing for the smoke sheets. Run the prep teacher phase first.")
        return 2

    setup = _prep_setup(runner, train_base, artifacts, frozen, timing12)
    adapter = _adapter(runner)
    init = adapter.snapshot()
    registered = int(((protocol or {}).get("adapter") or {}).get("trainable_params") or -1)
    report: dict = {
        "note": SMOKE_NOTE,
        "kernel_note": KERNEL_NOTE,
        "sheets_used": [ex.example_id for ex in sheets],
        "adapter": {
            "rank": RANK,
            "alpha": ALPHA,
            "n_modules": len(adapter.modules),
            "n_layers": len(model.model.layers),
            "n_params": adapter.n_params(),
            "registered_params": registered,
            "n_bytes": adapter.n_bytes(),
            "attention_at_load": current_attention(model),
        },
    }
    print(json.dumps(report["adapter"], indent=1), flush=True)

    states, snaps, targets, sheet_rows = [], [], [], []
    for ex in sheets:
        state, built, mix, build_s = _prep_state(runner, ex, artifacts, setup)
        if not (mix["ok"] and built["support_held"]):
            raise RuntimeError(f"Construction failed for {ex.example_id}.")
        tg = sheet_targets(runner, ex, teacher_rows[ex.example_id])
        states.append(state)
        snaps.append(_snap(state))
        targets.append(tg)
        align = {}
        for part in ("plan", "answer"):
            t = teacher_logits(runner, adapter, state, tg[part])
            align[part] = argmax_agreement(t, tg[part]["response_ids"])
            del t
        sheet_rows.append(
            {
                "example_id": ex.example_id,
                "build_s": build_s,
                "n_kv": mix["n_kv"],
                "n_result": mix["n_result"],
                "checks": tg["checks"],
                "teacher_alignment": align,
            }
        )
        print(f"  {ex.example_id} targets_ok={tg['checks']['ok']} teacher_align={align}", flush=True)
    report["sheets"] = sheet_rows

    seq0, state0 = targets[0]["plan"], states[0]
    zero_logits = {}
    for impl in (TRAIN_ATTN, GEN_ATTN):
        off = student_logits(runner, adapter, state0, seq0, on=False, impl=impl)
        on = student_logits(runner, adapter, state0, seq0, on=True, impl=impl)
        zero_logits[impl] = logit_delta(off, on)
        del off, on
    kernels = {}
    for name, fn in (
        ("student_plan", lambda impl: student_logits(runner, adapter, state0, seq0, on=False, impl=impl)),
        ("teacher_plan", lambda impl: teacher_logits(runner, adapter, state0, seq0, impl=impl)),
    ):
        a, b = fn(TRAIN_ATTN), fn(GEN_ATTN)
        kernels[name] = logit_delta(a, b)
        del a, b
    generation = {}
    for ex, state in list(zip(sheets, states))[:2]:
        g = _generation_pair(runner, adapter, ex, state, args)
        saved = saved_student.get(ex.example_id) or {}
        g["matches_saved_prep_student"] = {
            "plan_text": g["texts"]["adapter_off"]["plan_text"] == saved.get("plan_text"),
            "raw_text": g["texts"]["adapter_off"]["raw_text"] == saved.get("raw_text"),
        }
        generation[ex.example_id] = g
        print(f"  zero-init generation {ex.example_id}: plan_equal={g['plan_text_equal']} answer_equal={g['answer_text_equal']}", flush=True)
    report["zero_init"] = {"logits": zero_logits, "generation": generation}
    report["kernels"] = kernels
    print("  zero-init logits", zero_logits, "\n  kernels", kernels, flush=True)

    opt = make_optimizer(adapter)
    scaler = make_scaler(runner.device)
    micro, steps = [], []
    reset_peak_memory()
    t_train = time.perf_counter()
    for step in range(SMOKE_STEPS):
        for ex, state, tg in zip(sheets, states, targets):
            t0 = time.perf_counter()
            row = sheet_backward(runner, adapter, state, tg, scaler, weight=1.0 / SHEETS_PER_STEP)
            row.update({"example_id": ex.example_id, "step": step, "seconds": time.perf_counter() - t0})
            micro.append(row)
            print(f"  step {step} {ex.example_id} loss={row['loss']:.4f} finite={row['finite']} {row['seconds']:.1f}s", flush=True)
        st = optimizer_step(adapter, opt, scaler, inspect=lambda: grad_report(model, adapter))
        st["step"] = step
        st["b_all_zero_after"] = adapter.b_all_zero()
        steps.append(st)
        print(f"  optimizer step {step}: skipped={st['skipped']} norm={st['grad_norm']:.4g} grads={st['grads']['kinds']}", flush=True)
    train_s = time.perf_counter() - t_train
    n_layers = len(adapter.modules)
    first_applied = next((i for i, s in enumerate(steps) if not s["skipped"]), None)
    first = steps[first_applied if first_applied is not None else 0]["grads"]["kinds"]
    first_ok = first_applied is not None and (
        first["lora_B"]["with_grad"] == n_layers
        and first["lora_B"]["finite"] == n_layers
        and first["lora_B"]["nonzero"] == n_layers
        and first["lora_A"]["with_grad"] == n_layers
        and first["lora_A"]["nonzero"] == 0
    )
    later = steps[first_applied + 1]["grads"]["kinds"] if first_applied is not None and first_applied + 1 < len(steps) else None
    later_ok = bool(
        later
        and later["lora_A"]["nonzero"] == n_layers
        and later["lora_A"]["finite"] == n_layers
        and later["lora_B"]["finite"] == n_layers
    )
    mean_micro = sum(m["seconds"] for m in micro) / len(micro)
    n_train = int(((protocol or {}).get("data") or {}).get("train", {}).get("families") or 96)
    epochs = int(((protocol or {}).get("training") or {}).get("epochs") or 3)
    report["training"] = {
        "micro_steps": micro,
        "steps": steps,
        "first_step_grads_ok": first_ok,
        "later_step_grads_ok": later_ok,
        "loss_by_step": {s: [m["loss"] for m in micro if m["step"] == s] for s in range(SMOKE_STEPS)},
        "train_s": train_s,
        "mean_micro_step_s": mean_micro,
        "projected_training_s": mean_micro * n_train * epochs,
        "peak_gpu_gb": peak_gpu_memory_gb(),
        "attention_during_training": TRAIN_ATTN,
    }

    trained_on = student_logits(runner, adapter, state0, seq0, on=True)
    trained_off = student_logits(runner, adapter, state0, seq0, on=False)
    held = {ex.example_id: _held(state, snap) for ex, state, snap in zip(sheets, states, snaps)}
    rebuilt, _b, _m, _s = _prep_state(runner, sheets[0], artifacts, setup)
    rebuilt_equal = all(
        caches_equal(rebuilt["kvmix"][c].cache_cpu, snaps[0][c]) for c in (STUDENT, TEACHER)
    ) and caches_equal(rebuilt["bundle"].cache_cpu, snaps[0]["a_full"])
    report["after_training"] = {
        "logits_on_vs_off": logit_delta(trained_on, trained_off),
        "caches_held": held,
        "rebuilt_state_equal": bool(rebuilt_equal),
    }
    del trained_on, trained_off, rebuilt

    adapter.reset()
    opt = make_optimizer(adapter)
    scaler = make_scaler(runner.device)
    on = student_logits(runner, adapter, state0, seq0, on=True)
    off = student_logits(runner, adapter, state0, seq0, on=False)
    report["reset"] = {
        "matches_init": adapter.matches(init),
        "b_all_zero": adapter.b_all_zero(),
        "logits_on_vs_off": logit_delta(on, off),
        "optimizer_state_empty": len(opt.state) == 0,
        "grads_cleared": all(p.grad is None for p in adapter.parameters()),
    }
    report["attention_after"] = current_attention(model)
    report.update(smoke_passed(report))
    _write_json(out_root / "smoke_check.json", report)
    _persist(out_root, kaggle_dest)
    print(json.dumps({"gates": report["gates"], "all_passed": report["all_passed"]}, indent=2))
    return 0 if report["all_passed"] else 1


# ---------------------------------------------------------------- train


def epoch_order(example_ids: list[str], epoch: int) -> list[str]:
    """Registered order seed: 2026 + epoch, epochs numbered from 1."""
    order = list(example_ids)
    random.Random(ORDER_SEED_BASE + int(epoch)).shuffle(order)
    return order


def _require_smoke(out_root: Path) -> bool:
    path = out_root / "smoke_check.json"
    return path.is_file() and bool(json.loads(path.read_text(encoding="utf-8")).get("all_passed"))


def _keep(state) -> dict:
    """Only what B needs; A's full cache is dropped to save memory."""
    return {
        "kvmix": {c: state["kvmix"][c] for c in (STUDENT, TEACHER)},
        "result_start": int(state["slots_extra"]["result_start"]),
        "result_text": (state.get("instr") or {}).get("result_text") or "",
    }


def _train_states(runner, sheets, artifacts, setup, teacher_rows) -> tuple[dict, dict]:
    states, targets = {}, {}
    t0 = time.perf_counter()
    for i, ex in enumerate(sheets):
        state, built, mix, _s = _prep_state(runner, ex, artifacts, setup)
        if not (mix["ok"] and built["support_held"]):
            raise RuntimeError(f"Construction failed for {ex.example_id}.")
        tg = sheet_targets(runner, ex, teacher_rows[ex.example_id])
        if not tg["checks"]["ok"]:
            raise RuntimeError(f"Teacher targets failed their checks for {ex.example_id}: {tg['checks']}")
        states[ex.example_id] = _keep(state)
        targets[ex.example_id] = tg
        del state
        if (i + 1) % 16 == 0:
            print(f"  built {i + 1}/{len(sheets)} training states ({time.perf_counter() - t0:.0f}s)", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return states, targets


def _train_summary(log: list[dict], extra: dict) -> dict:
    by_epoch = {}
    for e in sorted({r["epoch"] for r in log if r["kind"] == "micro"}):
        micro = [r for r in log if r["kind"] == "micro" and r["epoch"] == e]
        steps = [r for r in log if r["kind"] == "step" and r["epoch"] == e]
        by_epoch[e] = {
            "n_micro": len(micro),
            "n_steps": len(steps),
            "mean_loss": sum(r["loss"] for r in micro) / len(micro),
            "skipped_steps": sum(bool(r["skipped"]) for r in steps),
            "mean_grad_norm": sum(r["grad_norm"] for r in steps) / max(1, len(steps)),
            "train_s": sum(r["seconds"] for r in micro),
        }
    return {"by_epoch": by_epoch, **extra}


def _run_train(args, out_root, kaggle_dest, splits, runner, inputs, protocol) -> int:
    if not _require_smoke(out_root):
        print("The implementation check (smoke) must pass before training.")
        return 2
    model = runner.wrapper.model
    sheets = [ex for ex in splits["train"] if ex.variant == "base"]
    by_id = {ex.example_id: ex for ex in sheets}
    adapter = _adapter(runner)
    opt = make_optimizer(adapter)
    scaler = make_scaler(runner.device)
    done = 0
    state_path = out_root / TRAIN_STATE
    if state_path.is_file():
        saved = torch.load(state_path, map_location="cpu")
        adapter.load(saved["adapter"])
        opt.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        done = int(saved["epoch"])
        print(f"Resuming after epoch {done}.")
    log_path = out_root / "train_log.jsonl"
    log = [r for r in _load_rows(log_path) if int(r["epoch"]) <= done]
    log_path.write_text("".join(json.dumps(r) + "\n" for r in log), encoding="utf-8")
    if done >= EPOCHS:
        print("Training already finished.")
        return 0

    artifacts = _load_messages(inputs["a_artifacts_train.jsonl"])
    frozen = _load_messages(inputs["timing12_a_artifacts_12.jsonl"])
    timing12 = load_timing12_jsonl(DATA_DIR / "plan4_timing12.jsonl")
    teacher_rows = {r["example_id"]: r for r in _load_rows(inputs["teacher_train.jsonl"]) if r["condition"] == TEACHER}
    if any(ex.example_id not in teacher_rows for ex in sheets):
        print("Teacher targets are missing for some training sheets.")
        return 2
    setup = _prep_setup(runner, sheets, artifacts, frozen, timing12)
    t_build = time.perf_counter()
    states, targets = _train_states(runner, sheets, artifacts, setup, teacher_rows)
    build_s = time.perf_counter() - t_build
    snaps = {k: {c: clone_cache_blob(s["kvmix"][c].cache_cpu) for c in (STUDENT, TEACHER)} for k, s in list(states.items())[:4]}
    reset_peak_memory()
    ids = [ex.example_id for ex in sheets]
    for epoch in range(done + 1, EPOCHS + 1):
        order = epoch_order(ids, epoch)
        t_epoch = time.perf_counter()
        for step in range(len(order) // SHEETS_PER_STEP):
            batch = order[step * SHEETS_PER_STEP : (step + 1) * SHEETS_PER_STEP]
            for key in batch:
                t0 = time.perf_counter()
                row = sheet_backward(runner, adapter, states[key], targets[key], scaler, weight=1.0 / SHEETS_PER_STEP)
                if not row["finite"]:
                    raise RuntimeError(f"Non-finite loss on {key} (epoch {epoch}, step {step}).")
                rec = {
                    "kind": "micro",
                    "epoch": epoch,
                    "step": step,
                    "example_id": key,
                    "difficulty": by_id[key].difficulty,
                    "loss": row["loss"],
                    "n_tokens": row["n_tokens"],
                    "student_argmax_agree_plan": row["plan"]["student_argmax_agree"],
                    "seconds": time.perf_counter() - t0,
                }
                log.append(rec)
                _append(log_path, rec)
            st = optimizer_step(adapter, opt, scaler, inspect=lambda: grad_report(model, adapter))
            if st["grads"]["n_base_params_with_grad_or_trainable"]:
                raise RuntimeError("Base-model parameters received gradients.")
            rec = {
                "kind": "step",
                "epoch": epoch,
                "step": step,
                "skipped": st["skipped"],
                "grad_norm": st["grad_norm"],
                "scale_after": st["scale_after"],
            }
            log.append(rec)
            _append(log_path, rec)
            if step % 6 == 5:
                recent = [r["loss"] for r in log if r["kind"] == "micro" and r["epoch"] == epoch][-24:]
                print(f"  epoch {epoch} step {step + 1}/24 mean loss (last 24 sheets) {sum(recent) / len(recent):.4f}", flush=True)
        torch.save(adapter.snapshot(), out_root / CHECKPOINT.format(epoch))
        torch.save(
            {"epoch": epoch, "adapter": adapter.snapshot(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict()},
            state_path,
        )
        print(f"Epoch {epoch} saved ({time.perf_counter() - t_epoch:.0f}s).", flush=True)
        _persist(out_root, kaggle_dest)
    held = {k: all(caches_equal(v[c], states[k]["kvmix"][c].cache_cpu) for c in (STUDENT, TEACHER)) for k, v in snaps.items()}
    summary = _train_summary(
        log,
        {
            "n_sheets": len(sheets),
            "epochs": EPOCHS,
            "sheets_per_step": SHEETS_PER_STEP,
            "lr": LR,
            "trainable_params": adapter.n_params(),
            "adapter_bytes": adapter.n_bytes(),
            "state_build_s": build_s,
            "peak_gpu_gb": peak_gpu_memory_gb(),
            "received_caches_unchanged": held,
            "teacher_target_generation_s": None,
            "note": "Training loss is on the training sheets only; it is not a pilot result.",
        },
    )
    prep_teacher = [r for r in _load_rows(inputs["teacher_train.jsonl"]) if r["condition"] == TEACHER]
    summary["teacher_target_generation_s"] = sum(
        float(r.get("plan_time_s") or 0) + float(r.get("receiver_latency_s") or 0) for r in prep_teacher
    )
    _write_json(out_root / "train_summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({k: v for k, v in summary.items() if k != "received_caches_unchanged"}, indent=2))
    return 0 if all(held.values()) else 1


# ---------------------------------------------------------------- dev


def select_epoch(labels_by_epoch: dict) -> int:
    """Highest dev-base labels correct; ties go to the earlier epoch."""
    labels = {int(e): int(v) for e, v in labels_by_epoch.items()}
    best = max(labels.values())
    return min(e for e, v in labels.items() if v == best)


def no_r_bundle(runner, bundle, result_start: int):
    rs = int(result_start)
    ids = [int(t) for t in bundle.full_ids_cpu.view(-1).tolist()]
    blob = gather_cache_positions(bundle.cache_cpu, list(range(rs)))
    return runner._bundle_from_packed(blob, ids[:rs], pos_cursor=rs, source="no_r")


def mismatched_bundle(runner, own, partner, result_start: int, partner_start: int):
    """This sheet's own support, then the partner's transferred R at the same packed positions."""
    rs = int(result_start)
    if int(partner_start) != rs:
        raise RuntimeError("Mismatched R needs the same result start on both sheets.")
    own_ids = [int(t) for t in own.full_ids_cpu.view(-1).tolist()]
    p_ids = [int(t) for t in partner.full_ids_cpu.view(-1).tolist()]
    n_p = n_kv_positions(partner.cache_cpu)
    blob = concat_cache_blobs(
        [
            gather_cache_positions(own.cache_cpu, list(range(rs))),
            gather_cache_positions(partner.cache_cpu, list(range(rs, n_p))),
        ]
    )
    ids = own_ids[:rs] + p_ids[rs:]
    return runner._bundle_from_packed(blob, ids, pos_cursor=len(ids), source="mismatched_r")


def dev_cache_check(own, partner, no_r, mism, result_start: int) -> dict:
    rs = int(result_start)
    n_own, n_p = n_kv_positions(own.cache_cpu), n_kv_positions(partner.cache_cpu)
    support = list(range(rs))
    row = {
        "own_cursor_is_length": int(own.pos_cursor_cpu.view(-1)[0].item()) == n_own,
        "no_r_is_own_support": n_kv_positions(no_r.cache_cpu) == rs
        and bool(cache_slice_delta(no_r.cache_cpu, own.cache_cpu, support)["equal"]),
        "mismatched_support_is_own": bool(cache_slice_delta(mism.cache_cpu, own.cache_cpu, support)["equal"]),
        "mismatched_r_is_partner_r": n_kv_positions(mism.cache_cpu) == n_p
        and bool(cache_slice_delta(mism.cache_cpu, partner.cache_cpu, list(range(rs, n_p)))["equal"]),
        "mismatched_ids": [int(t) for t in mism.full_ids_cpu.view(-1).tolist()][rs:]
        == [int(t) for t in partner.full_ids_cpu.view(-1).tolist()][rs:],
        "masks_all_ones": all(bool((b.past_mask_cpu > 0).all()) for b in (no_r, mism)),
        "cursors": {
            "no_r": int(no_r.pos_cursor_cpu.view(-1)[0].item()) == rs,
            "mismatched_r": int(mism.pos_cursor_cpu.view(-1)[0].item()) == n_p,
        },
        "n_result_own": n_own - rs,
        "n_result_partner": n_p - rs,
    }
    row["ok"] = bool(
        row["own_cursor_is_length"]
        and row["no_r_is_own_support"]
        and row["mismatched_support_is_own"]
        and row["mismatched_r_is_partner_r"]
        and row["mismatched_ids"]
        and row["masks_all_ones"]
        and all(row["cursors"].values())
    )
    return row


def _dev_setup(runner, splits, inputs):
    dev = list(splits["dev"])
    artifacts = _load_messages(inputs["a_artifacts_dev.jsonl"])
    frozen = _load_messages(inputs["timing12_a_artifacts_12.jsonl"])
    timing12 = load_timing12_jsonl(DATA_DIR / "plan4_timing12.jsonl")
    if any(a_state_key(ex) not in artifacts for ex in dev):
        raise RuntimeError("Dev A sheets are missing. Do not regenerate them here.")
    setup = _prep_setup(runner, dev, artifacts, frozen, timing12)
    states = {}
    for ex in dev:
        state, built, mix, _s = _prep_state(runner, ex, artifacts, setup)
        if not (mix["ok"] and built["support_held"]):
            raise RuntimeError(f"Construction failed for {ex.example_id}.")
        states[ex.example_id] = _keep(state)
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return dev, states


def _generate(runner, adapter, ex, cond: str, bundle, message: str, args) -> dict:
    spec = EVAL_CONDS[cond]
    if spec["cache"] is None:
        with adapter.off():
            rec = _run_text_b_timed(
                runner, ex, message=message, plan_max_new=args.plan_max_new_12, answer_max_new=args.max_new_tokens
            )
    else:
        switch = adapter.on() if spec["adapted"] else adapter.off()
        with switch:
            rec = _run_kv_b_timed(
                runner,
                ex,
                runner.clone_sender(bundle),
                plan_max_new=args.plan_max_new_12,
                answer_max_new=args.max_new_tokens,
            )
    rec.update(
        {
            "condition": cond,
            "cache": spec["cache"],
            "adapted": spec["adapted"],
            "a_state_key": a_state_key(ex),
            "attention": current_attention(runner.wrapper.model),
        }
    )
    return rec


def _brief(rec: dict) -> str:
    return (
        f"labels={rec.get('plan_n_status_correct')}/12 I->E={rec.get('plan_ineligible_marked_eligible')} "
        f"E->I={rec.get('plan_eligible_marked_ineligible')} choice={rec.get('correct')} parse={rec.get('parse_ok')}"
    )


def _run_select(args, out_root, kaggle_dest, splits, runner, inputs, protocol) -> int:
    ckpts = [out_root / CHECKPOINT.format(e) for e in range(1, EPOCHS + 1)]
    if not all(p.is_file() for p in ckpts):
        print("Run --phase train first (all three epoch checkpoints are needed).")
        return 2
    adapter = _adapter(runner)
    dev, states = _dev_setup(runner, splits, inputs)
    bases = [ex for ex in dev if ex.variant == "base"]
    path = out_root / "select_evaluations.jsonl"
    rows = _load_rows(path)
    done = {(r["example_id"], r["condition"], r.get("epoch")) for r in rows}
    jobs = [("unadapted_correct_r", None)] + [("adapted_correct_r", e) for e in range(1, EPOCHS + 1)]
    for cond, epoch in jobs:
        if epoch is not None:
            adapter.load(torch.load(ckpts[epoch - 1], map_location="cpu"))
        for ex in bases:
            if (ex.example_id, cond, epoch) in done:
                continue
            st = states[ex.example_id]
            rec = _generate(runner, adapter, ex, cond, st["kvmix"][STUDENT], "", args)
            rec["epoch"] = epoch
            rows.append(rec)
            _append(path, rec)
            print(f"  select {cond} epoch={epoch} {ex.example_id}: {_brief(rec)}", flush=True)
        _persist(out_root, kaggle_dest)
    labels = {
        e: sum(int(r.get("plan_n_status_correct") or 0) for r in rows if r["condition"] == "adapted_correct_r" and r.get("epoch") == e)
        for e in range(1, EPOCHS + 1)
    }
    unadapted = [r for r in rows if r["condition"] == "unadapted_correct_r"]
    selected = select_epoch(labels)
    out = {
        "rule": "highest dev-base labels correct (of 144); ties go to the earlier epoch",
        "labels_by_epoch": labels,
        "selected_epoch": selected,
        "unadapted_labels": sum(int(r.get("plan_n_status_correct") or 0) for r in unadapted),
        "by_epoch": {
            e: _twelve_labels([r for r in rows if r["condition"] == "adapted_correct_r" and r.get("epoch") == e])
            for e in range(1, EPOCHS + 1)
        },
        "unadapted": _twelve_labels(unadapted),
        "note": "Selection uses dev base sheets only. Nothing on eval or timing12 is computed.",
    }
    _write_json(out_root / "selection.json", out)
    _persist(out_root, kaggle_dest)
    print(json.dumps(out, indent=2))
    return 0


def _pred(row: dict, carrier: str):
    return (row.get("plan_pred_status") or {}).get(carrier)


def mismatch_following(rows: list[dict], cond: str, partners: dict, gold: dict) -> dict:
    """On carriers whose status differs from the partner's: does the label follow the partner?"""
    by_id = {r["example_id"]: r for r in rows if r["condition"] == cond}
    out = {"n_carriers": 0, "follows_partner": 0, "follows_own": 0, "missing": 0}
    for ex_id, p_id in partners.items():
        row = by_id.get(ex_id)
        if row is None:
            continue
        for c, own in gold[ex_id].items():
            other = gold[p_id][c]
            if own == other:
                continue
            out["n_carriers"] += 1
            pred = _pred(row, c)
            out["follows_partner"] += pred == other
            out["follows_own"] += pred == own
            out["missing"] += pred is None
    return out


def a_change_pairs(rows: list[dict], cond: str, pairs: dict, gold: dict) -> dict:
    """Pairs whose changed carrier is labelled correctly in both base and a_flip."""
    by_id = {r["example_id"]: r for r in rows if r["condition"] == cond}
    out = {"n_pairs": 0, "both_correct": 0, "base_correct": 0, "flip_correct": 0}
    for base_id, (flip_id, c) in pairs.items():
        b, f = by_id.get(base_id), by_id.get(flip_id)
        if b is None or f is None:
            continue
        out["n_pairs"] += 1
        bc, fc = _pred(b, c) == gold[base_id][c], _pred(f, c) == gold[flip_id][c]
        out["base_correct"] += bc
        out["flip_correct"] += fc
        out["both_correct"] += bc and fc
    return out


def _malformed(rows: list[dict]) -> int:
    return sum((not r.get("parse_ok")) + bool(r.get("plan_truncated")) for r in rows)


def dev_gates(rows: list[dict], partners: dict, pairs: dict, gold: dict) -> dict:
    def sub(cond, variant=None):
        return [r for r in rows if r["condition"] == cond and (variant is None or r.get("variant") == variant)]

    def labels(rs):
        return sum(int(r.get("plan_n_status_correct") or 0) for r in rs)

    def per_family(rs):
        return {r["family_id"]: int(r.get("plan_n_status_correct") or 0) for r in rs}

    a_base, u_base = sub("adapted_correct_r", "base"), sub("unadapted_correct_r", "base")
    gain = _cluster_delta(per_family(u_base), per_family(a_base))
    no_r = sub("adapted_no_r", "base")
    follow = {c: mismatch_following(rows, c, partners, gold) for c in ("adapted_mismatched_r", "unadapted_mismatched_r")}
    pair = {c: a_change_pairs(rows, c, pairs, gold) for c in ("adapted_correct_r", "unadapted_correct_r")}
    mal = {c: _malformed(sub(c)) for c in ("adapted_correct_r", "unadapted_correct_r")}
    ci = gain.get("ci95_per_case") or [0, 0]
    gates = {
        "beats_unadapted_kv": {
            "adapted": labels(a_base),
            "unadapted": labels(u_base),
            "gain": labels(a_base) - labels(u_base),
            "ci95_per_family": ci,
            "pass": labels(a_base) - labels(u_base) >= GATE_MIN_GAIN and ci[0] > 0,
        },
        "worse_without_results": {
            "correct_r": labels(a_base),
            "no_r": labels(no_r),
            "pass": len(no_r) == len(a_base) and labels(a_base) - labels(no_r) >= GATE_MIN_R_DEPENDENCE,
        },
        "follows_mismatched_results": {
            **follow["adapted_mismatched_r"],
            "pass": follow["adapted_mismatched_r"]["n_carriers"] > 0
            and follow["adapted_mismatched_r"]["follows_partner"] > follow["adapted_mismatched_r"]["follows_own"],
        },
        "a_change_pairs": {
            "adapted": pair["adapted_correct_r"],
            "unadapted": pair["unadapted_correct_r"],
            "pass": pair["adapted_correct_r"]["n_pairs"] > 0
            and pair["adapted_correct_r"]["both_correct"] >= pair["unadapted_correct_r"]["both_correct"],
        },
        "malformed_output": {
            "adapted": mal["adapted_correct_r"],
            "unadapted": mal["unadapted_correct_r"],
            "pass": mal["adapted_correct_r"] <= mal["unadapted_correct_r"] + GATE_MAX_EXTRA_MALFORMED,
        },
    }
    return {
        "gates": gates,
        "advance": all(g["pass"] for g in gates.values()),
        "context": {"unadapted_mismatched_following": follow["unadapted_mismatched_r"], "gain_detail": gain},
    }


def _dev_summary(rows, partners, pairs, gold, selection) -> dict:
    conds = list(DEV_CONDS)
    out = {
        "selected_epoch": selection.get("selected_epoch"),
        "labels": {
            v: {c: _twelve_labels([r for r in rows if r["condition"] == c and r.get("variant") == v]) for c in conds}
            for v in ("base", "a_flip")
        },
        "final_choice": {
            v: {c: _twelve_choice([r for r in rows if r["condition"] == c and r.get("variant") == v]) for c in conds}
            for v in ("base", "a_flip")
        },
        "costs": {
            c: {
                k: (sum(float(r.get(k) or 0) for r in rows if r["condition"] == c) / max(1, sum(r["condition"] == c for r in rows)))
                for k in ("plan_prefill_s", "plan_decode_s", "answer_prefill_s", "answer_decode_s", "b_prefill_s", "b_decode_s")
            }
            for c in conds
        },
        "note": GATE_NOTE,
        "scope": "Development families (already used for checkpoint selection). Not confirmation.",
    }
    out.update(dev_gates(rows, partners, pairs, gold))
    return out


def _run_gates(args, out_root, kaggle_dest, splits, runner, inputs, protocol) -> int:
    spath = out_root / "selection.json"
    if not spath.is_file():
        print("Run --phase select first.")
        return 2
    selection = json.loads(spath.read_text(encoding="utf-8"))
    epoch = int(selection["selected_epoch"])
    adapter = _adapter(runner)
    adapter.load(torch.load(out_root / CHECKPOINT.format(epoch), map_location="cpu"))
    dev, states = _dev_setup(runner, splits, inputs)
    by_id = {ex.example_id: ex for ex in dev}
    partners = mismatch_partner(dev)
    bases = {ex.family_id: ex for ex in dev if ex.variant == "base"}
    pairs = {
        bases[ex.family_id].example_id: (ex.example_id, changed_carrier(bases[ex.family_id], ex))
        for ex in dev
        if ex.variant == "a_flip"
    }
    gold = {ex.example_id: gold_eligibility_statuses_n(ex) for ex in dev}

    caches, checks = {}, {}
    for ex in dev:
        st = states[ex.example_id]
        own = st["kvmix"][STUDENT]
        caches[ex.example_id] = {"correct_r": own, "fk_fv": st["kvmix"][TEACHER]}
        if ex.variant == "base":
            p = states[partners[ex.example_id]]
            no_r = no_r_bundle(runner, own, st["result_start"])
            mism = mismatched_bundle(runner, own, p["kvmix"][STUDENT], st["result_start"], p["result_start"])
            caches[ex.example_id].update({"no_r": no_r, "mismatched_r": mism})
            checks[ex.example_id] = dev_cache_check(own, p["kvmix"][STUDENT], no_r, mism, st["result_start"])
            checks[ex.example_id]["partner"] = partners[ex.example_id]
            checks[ex.example_id]["carriers_differing"] = sum(
                gold[ex.example_id][c] != gold[partners[ex.example_id]][c] for c in gold[ex.example_id]
            )
    construction = {
        "selected_epoch": epoch,
        "partners": partners,
        "a_change_pairs": {k: {"flip": v[0], "changed_carrier": v[1]} for k, v in pairs.items()},
        "cases": checks,
        "all_passed": all(c["ok"] for c in checks.values()),
    }
    _write_json(out_root / "dev_construction.json", construction)
    _persist(out_root, kaggle_dest)
    if not construction["all_passed"]:
        print("Dev cache construction failed. No answers generated.")
        return 1

    path = out_root / "dev_evaluations.jsonl"
    rows = _load_rows(path)
    done = {(r["example_id"], r["condition"]) for r in rows}
    reuse = {
        ("adapted_correct_r", epoch),
        ("unadapted_correct_r", None),
    }
    for r in _load_rows(out_root / "select_evaluations.jsonl"):
        if (r["condition"], r.get("epoch")) in reuse and (r["example_id"], r["condition"]) not in done:
            r = dict(r, reused_from="selection")
            rows.append(r)
            done.add((r["example_id"], r["condition"]))
            _append(path, r)
    for ex in dev:
        snaps = {k: clone_cache_blob(b.cache_cpu) for k, b in caches[ex.example_id].items()}
        for cond, spec in DEV_CONDS.items():
            if ex.variant not in spec["variants"] or (ex.example_id, cond) in done:
                continue
            bundle = None if spec["cache"] is None else caches[ex.example_id][spec["cache"]]
            rec = _generate(runner, adapter, ex, cond, bundle, states[ex.example_id]["result_text"], args)
            rec["epoch"] = epoch if spec["adapted"] else None
            rows.append(rec)
            done.add((ex.example_id, cond))
            _append(path, rec)
            print(f"  {ex.example_id} {cond}: {_brief(rec)}", flush=True)
        for k, snap in snaps.items():
            if not caches_equal(snap, caches[ex.example_id][k].cache_cpu):
                raise RuntimeError(f"{k} cache changed during generation for {ex.example_id}.")
        _persist(out_root, kaggle_dest)
    summary = _dev_summary(rows, partners, pairs, gold, selection)
    summary["construction_passed"] = construction["all_passed"]
    summary["n_rows"] = len(rows)
    _write_json(out_root / "dev_summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({"gates": summary["gates"], "advance": summary["advance"]}, indent=2))
    return 0


# ---------------------------------------------------------------- eval (untouched families)


def _run_freeze(args, out_root, kaggle_dest, splits, runner, inputs, protocol) -> int:
    dpath, spath = out_root / "dev_summary.json", out_root / "selection.json"
    if not (dpath.is_file() and spath.is_file()):
        print("Selection and dev gates must be complete before freezing.")
        return 2
    dev = json.loads(dpath.read_text(encoding="utf-8"))
    selection = json.loads(spath.read_text(encoding="utf-8"))
    if not (dev.get("advance") and dev.get("construction_passed")):
        print("The development gates did not pass. Untouched families stay untouched.")
        return 2
    epoch = int(selection["selected_epoch"])
    ckpt = out_root / CHECKPOINT.format(epoch)
    if not ckpt.is_file():
        print(f"{ckpt.name} is missing. Upload the results folder from the training run; do not retrain.")
        return 2
    frozen = {
        "selected_epoch": epoch,
        "checkpoint": ckpt.name,
        "sha256": file_sha256(ckpt),
        "dev_labels_by_epoch": selection["labels_by_epoch"],
        "dev_gates": {k: g["pass"] for k, g in dev["gates"].items()},
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "Frozen before any eval A sheet is written. Every eval row must use this checkpoint.",
    }
    fpath = out_root / FROZEN
    if fpath.is_file():
        old = json.loads(fpath.read_text(encoding="utf-8"))
        if old["sha256"] != frozen["sha256"] or int(old["selected_epoch"]) != epoch:
            print("A different candidate was frozen earlier. Refusing to change it.")
            return 2
        print(f"Candidate already frozen: epoch {epoch}, sha256 {old['sha256'][:16]}...")
        return 0
    _write_json(fpath, frozen)
    _persist(out_root, kaggle_dest)
    print(json.dumps(frozen, indent=2))
    return 0


def _frozen(out_root: Path) -> dict | None:
    fpath = out_root / FROZEN
    if not fpath.is_file():
        return None
    frozen = json.loads(fpath.read_text(encoding="utf-8"))
    ckpt = out_root / frozen["checkpoint"]
    if not ckpt.is_file() or file_sha256(ckpt) != frozen["sha256"]:
        raise RuntimeError("Frozen checkpoint is missing or its hash changed.")
    return frozen


def _run_eval_write(args, out_root, kaggle_dest, splits, runner, inputs, protocol) -> int:
    if _frozen(out_root) is None:
        print("Freeze the candidate first (--phase freeze).")
        return 2
    if getattr(runner, "_query_adapter", None) is not None:
        raise RuntimeError("A's sheets must be written without any adapter attached.")
    path = out_root / EVAL_ARTIFACTS
    artifacts = _load_messages(path)
    todo = [ex for ex in splits["eval"] if a_state_key(ex) not in artifacts]
    print(f"A write eval: {len(artifacts)} written, {len(todo)} to do", flush=True)
    for i, ex in enumerate(todo):
        key = a_state_key(ex)
        ordinary = ordinary_sender_prompt(encode_prompt_for(ex), written_result_handoff_n(ex.carriers))
        if prompt_contains_b_private(ordinary, ex) or prompt_contains_solver_labels_n(ordinary, ex):
            raise RuntimeError(f"A prompt leaked private fields for {key}.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        text, n_new, truncated, gen_s, bundle = runner.generate_text_keep_cache(
            ordinary, max_new_tokens=A_WRITTEN_MAX_NEW_12, source="written"
        )
        artifacts[key] = _a_record(ex, text, n_new, truncated, gen_s, bundle, ordinary)
        _write_artifacts12(path, artifacts)
        _persist(out_root, kaggle_dest)
        print(
            f"  [{i + 1}/{len(todo)}] {key} trunc={truncated} n_new={n_new} "
            f"status={artifacts[key]['written_score']['n_status_correct']}/12 s={gen_s:.1f}",
            flush=True,
        )
        del bundle
    return 0


def _eval_inputs(out_root, splits, inputs):
    examples = list(splits["eval"])
    artifacts = _load_messages(out_root / EVAL_ARTIFACTS)
    missing = [a_state_key(ex) for ex in examples if a_state_key(ex) not in artifacts]
    if missing:
        raise RuntimeError(f"{len(missing)} eval A sheets are missing; run --phase eval_write.")
    frozen_a = _load_messages(inputs["timing12_a_artifacts_12.jsonl"])
    timing12 = load_timing12_jsonl(DATA_DIR / "plan4_timing12.jsonl")
    return examples, artifacts, frozen_a, timing12


def _eval_maps(examples):
    bases = {ex.family_id: ex for ex in examples if ex.variant == "base"}
    partners = mismatch_partner(examples)
    pairs = {
        bases[ex.family_id].example_id: (ex.example_id, changed_carrier(bases[ex.family_id], ex))
        for ex in examples
        if ex.variant == "a_flip"
    }
    gold = {ex.example_id: gold_eligibility_statuses_n(ex) for ex in examples}
    return bases, partners, pairs, gold


def _run_eval_check(args, out_root, kaggle_dest, splits, runner, inputs, protocol) -> int:
    if _frozen(out_root) is None:
        print("Freeze the candidate first.")
        return 2
    examples, artifacts, frozen_a, timing12 = _eval_inputs(out_root, splits, inputs)
    setup = _prep_setup(runner, examples, artifacts, frozen_a, timing12)
    bases, partners, _pairs, gold = _eval_maps(examples)
    kept, rows, template_ref = {}, {}, {}
    for ex in examples:
        state, built, mix, build_s = _prep_state(runner, ex, artifacts, setup)
        rs = int(state["slots_extra"]["result_start"])
        student = state["kvmix"][STUDENT]
        ref = template_ref.setdefault(
            built["template_id"],
            {"a_state_key": a_state_key(ex), "result_start": rs, "ids": _support_ids(student, rs), "cache": student.cache_cpu},
        )
        rec = artifacts[a_state_key(ex)]
        rows[ex.example_id] = {
            "variant": ex.variant,
            "template_id": built["template_id"],
            "build_s": build_s,
            "support_held": bool(built["support_held"]),
            "mixes_ok": bool(mix["ok"]),
            "result_start": rs,
            "n_result": mix["n_result"],
            "support_ids_equal_template": _support_vs_ref(student, rs, ref)["ids_equal"],
            "a_written_status_correct": rec["written_score"]["n_status_correct"],
            "a_written_truncated": bool(rec["written_truncated"]),
            "a_flip_sender": _a_flip_sender(ex, bases[ex.family_id], rec),
        }
        if ex.variant == "base":
            kept[ex.example_id] = _keep(state)
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for ex_id, st in kept.items():
        p = kept[partners[ex_id]]
        own = st["kvmix"][STUDENT]
        no_r = no_r_bundle(runner, own, st["result_start"])
        mism = mismatched_bundle(runner, own, p["kvmix"][STUDENT], st["result_start"], p["result_start"])
        row = dev_cache_check(own, p["kvmix"][STUDENT], no_r, mism, st["result_start"])
        rows[ex_id]["dependence_caches"] = row
        rows[ex_id]["partner"] = partners[ex_id]
        rows[ex_id]["carriers_differing"] = sum(gold[ex_id][c] != gold[partners[ex_id]][c] for c in gold[ex_id])
    for r in rows.values():
        r["ok"] = bool(
            r["support_held"]
            and r["mixes_ok"]
            and r["n_result"] > 0
            and r["support_ids_equal_template"]
            and (r["variant"] != "base" or r["dependence_caches"]["ok"])
        )
    flips = [r["a_flip_sender"] for r in rows.values() if r["a_flip_sender"]]
    out = {
        "n": len(rows),
        "all_passed": all(r["ok"] for r in rows.values()) and len(rows) == len(examples) and setup[5]["ok"],
        "templates_match_frozen": setup[5]["ok"],
        "a_written": {
            "status_correct": sum(r["a_written_status_correct"] for r in rows.values()),
            "n_status": 12 * len(rows),
            "truncated": sum(r["a_written_truncated"] for r in rows.values()),
        },
        "a_flip_changed_carrier_written_correctly": {"n": len(flips), "correct": sum(f["a_correct"] for f in flips)},
        "partners": partners,
        "rows": rows,
    }
    _write_json(out_root / "eval_construction.json", out)
    _persist(out_root, kaggle_dest)
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "partners")}, indent=2))
    return 0 if out["all_passed"] else 1


def _family_counts(rows: list[dict], cond: str, variant: str, field: str) -> dict:
    return {
        r["family_id"]: int(r.get(field) or 0)
        for r in rows
        if r["condition"] == cond and r.get("variant") == variant
    }


def _per_token(rows: list[dict], cond: str) -> dict:
    sub = [r for r in rows if r["condition"] == cond]
    toks = sum(int(r.get("n_plan_tokens") or 0) + int(r.get("n_new_tokens") or 0) for r in sub)
    dec = sum(float(r.get("b_decode_s") or 0) for r in sub)
    pre = [float(r.get("b_prefill_s") or 0) for r in sub]
    kv = [int(r.get("n_kv_positions") or 0) for r in sub]
    return {
        "n_rows": len(sub),
        "decode_s_per_token": dec / toks if toks else None,
        "mean_prefill_s": sum(pre) / len(pre) if pre else None,
        "mean_decode_s": dec / len(sub) if sub else None,
        "mean_kv_positions": sum(kv) / len(kv) if kv else None,
        "mean_resident_mib": (sum(kv) / len(kv)) * KV_MIB_PER_POSITION if kv else None,
    }


def eval_summary(rows: list[dict], partners: dict, pairs: dict, gold: dict, *, train: dict | None = None, frozen: dict | None = None) -> dict:
    conds = list(EVAL_CONDS)
    comparisons = {}
    for name, left, right in EVAL_PAIRED:
        comparisons[name] = {
            v: {
                "labels": _cluster_delta(
                    _family_counts(rows, right, v, "plan_n_status_correct"),
                    _family_counts(rows, left, v, "plan_n_status_correct"),
                ),
                "all_twelve": _cluster_delta(
                    _family_counts(rows, right, v, "plan_all_status_correct"),
                    _family_counts(rows, left, v, "plan_all_status_correct"),
                ),
            }
            for v in ("base", "a_flip")
            if v in EVAL_CONDS[left]["variants"] and v in EVAL_CONDS[right]["variants"]
        }
    adapted, unadapted = _per_token(rows, "adapted_correct_r"), _per_token(rows, "unadapted_correct_r")
    overhead = None
    if adapted["decode_s_per_token"] and unadapted["decode_s_per_token"]:
        overhead = {
            "decode_per_token_ratio": adapted["decode_s_per_token"] / unadapted["decode_s_per_token"],
            "prefill_ratio": (adapted["mean_prefill_s"] or 0) / (unadapted["mean_prefill_s"] or 1),
        }
    out = {
        "candidate": frozen,
        "n_rows": len(rows),
        "primary": "eligibility labels (12 per sheet) and all-twelve cases, adapted transferred R vs unadapted",
        "labels": {
            v: {c: _twelve_labels([r for r in rows if r["condition"] == c and r.get("variant") == v]) for c in conds}
            for v in ("base", "a_flip")
        },
        "final_choice": {
            v: {c: _twelve_choice([r for r in rows if r["condition"] == c and r.get("variant") == v]) for c in conds}
            for v in ("base", "a_flip")
        },
        "paired_by_family": comparisons,
        "costs": {
            "receiver": {c: _per_token(rows, c) for c in conds},
            "adapter_overhead": overhead,
            "adapter": None if train is None else {"params": train.get("trainable_params"), "bytes": train.get("adapter_bytes")},
            "one_time": None
            if train is None
            else {
                "teacher_target_generation_s": train.get("teacher_target_generation_s"),
                "training_state_build_s": train.get("state_build_s"),
                "training_s": sum(float(b.get("train_s") or 0) for b in (train.get("by_epoch") or {}).values()),
                "training_peak_gpu_gb": train.get("peak_gpu_gb"),
            },
            "note": COST_NOTE_QADAPT,
        },
        "stats_note": (
            "Deltas are adapted minus the named comparison. "
            f"Paired bootstrap over families ({N_BOOT} draws, seed {BOOT_SEED}); each sheet's twelve labels "
            "stay together. Labels are not independent examples."
        ),
        "scope": "Untouched families, run once with the frozen candidate. Reported whatever they show.",
    }
    checks = dev_gates(rows, partners, pairs, gold)
    out["message_dependence"] = {
        "checks": checks["gates"],
        "all_hold": checks["advance"],
        "unadapted_mismatched_following": checks["context"]["unadapted_mismatched_following"],
        "note": "The registered development gates recomputed on eval. Reported only; nothing is chosen from them.",
    }
    return out


def _run_eval(args, out_root, kaggle_dest, splits, runner, inputs, protocol) -> int:
    frozen = _frozen(out_root)
    if frozen is None:
        print("Freeze the candidate first.")
        return 2
    cpath = out_root / "eval_construction.json"
    check = json.loads(cpath.read_text(encoding="utf-8")) if cpath.is_file() else None
    if check is None or not check.get("all_passed"):
        print("Run --phase eval_check first; it must pass before any eval answers.")
        return 2
    examples, artifacts, frozen_a, timing12 = _eval_inputs(out_root, splits, inputs)
    setup = _prep_setup(runner, examples, artifacts, frozen_a, timing12)
    adapter = _adapter(runner)
    adapter.load(torch.load(out_root / frozen["checkpoint"], map_location="cpu"))
    _bases, partners, pairs, gold = _eval_maps(examples)
    path = out_root / "eval_evaluations.jsonl"
    rows = _load_rows(path)
    done = {(r["example_id"], r["condition"]) for r in rows}
    todo = [ex for ex in examples if any(ex.variant in s["variants"] and (ex.example_id, c) not in done for c, s in EVAL_CONDS.items())]
    if args.limit is not None:
        todo = todo[: max(0, args.limit)]
    kept = {}
    need = {partners[ex.example_id] for ex in todo if ex.variant == "base"}
    for ex in examples:
        if ex.example_id in need:
            state, built, mix, _s = _prep_state(runner, ex, artifacts, setup)
            if not (mix["ok"] and built["support_held"]):
                raise RuntimeError(f"Construction failed for partner {ex.example_id}.")
            kept[ex.example_id] = _keep(state)
            del state
    print(f"Eval: {len(done)} rows done, {len(todo)} sheets to run", flush=True)
    for i, ex in enumerate(todo):
        state, built, mix, _s = _prep_state(runner, ex, artifacts, setup)
        if not (mix["ok"] and built["support_held"]):
            raise RuntimeError(f"Construction failed for {ex.example_id}.")
        st = _keep(state)
        caches = {"correct_r": st["kvmix"][STUDENT], "fk_fv": st["kvmix"][TEACHER], "full_kv": state["bundle"]}
        if ex.variant == "base":
            p = kept[partners[ex.example_id]]
            caches["no_r"] = no_r_bundle(runner, caches["correct_r"], st["result_start"])
            caches["mismatched_r"] = mismatched_bundle(
                runner, caches["correct_r"], p["kvmix"][STUDENT], st["result_start"], p["result_start"]
            )
            if not dev_cache_check(caches["correct_r"], p["kvmix"][STUDENT], caches["no_r"], caches["mismatched_r"], st["result_start"])["ok"]:
                raise RuntimeError(f"Dependence caches failed for {ex.example_id}.")
        snaps = {k: clone_cache_blob(b.cache_cpu) for k, b in caches.items()}
        reset_peak_memory()
        for cond, spec in EVAL_CONDS.items():
            if ex.variant not in spec["variants"] or (ex.example_id, cond) in done:
                continue
            bundle = None if spec["cache"] is None else caches[spec["cache"]]
            rec = _generate(runner, adapter, ex, cond, bundle, st["result_text"], args)
            rec.update({"epoch": frozen["selected_epoch"] if spec["adapted"] else None, "candidate_sha256": frozen["sha256"]})
            rows.append(rec)
            done.add((ex.example_id, cond))
            _append(path, rec)
            print(f"  [{i + 1}/{len(todo)}] {ex.example_id} {cond}: {_brief(rec)}", flush=True)
        for k, snap in snaps.items():
            if not caches_equal(snap, caches[k].cache_cpu):
                raise RuntimeError(f"{k} cache changed during generation for {ex.example_id}.")
        _persist(out_root, kaggle_dest)
        del state, caches, snaps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if any(r.get("candidate_sha256") != frozen["sha256"] for r in rows):
        raise RuntimeError("Eval rows from a different candidate are present.")
    tpath = out_root / "train_summary.json"
    train = json.loads(tpath.read_text(encoding="utf-8")) if tpath.is_file() else None
    summary = eval_summary(rows, partners, pairs, gold, train=train, frozen=frozen)
    summary["construction"] = {k: check[k] for k in ("all_passed", "a_written", "a_flip_changed_carrier_written_correctly")}
    _write_json(out_root / "eval_summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({"labels": summary["labels"]["base"], "message_dependence": summary["message_dependence"]["all_hold"]}, indent=1))
    return 0


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Receiver query-adapter pilot.")
    parser.add_argument("--phase", choices=("smoke", "train", "select", "gates", "pilot") + EVAL_PHASES, default="smoke")
    parser.add_argument("--limit", type=int, default=None, help="eval phase only: run at most this many sheets")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--plan_max_new_12", type=int, default=B_PLAN_MAX_NEW_12)
    args = parser.parse_args()
    if args.max_new_tokens != 48 or args.plan_max_new_12 != 384:
        print("Decoding allowances are frozen at 384 plan / 48 answer tokens.")
        return 2
    protocol = pinned.get("study_plan4_qadapt") or {}
    mism = registered_hash_mismatches(protocol)
    if mism:
        print(f"Split files differ from the registered hashes: {mism}. Do not regenerate.")
        return 2

    out_root, kaggle_dest = _paths()
    if "stage_plan4_qadapt" not in str(out_root) or "qadapt_prep" in str(out_root):
        print("Refusing to write outside results_stage_plan4_qadapt.")
        return 2
    out_root.mkdir(parents=True, exist_ok=True)
    for folder in [d for d in (kaggle_dest, REPO_ROOT / "results_stage_plan4_qadapt") if d is not None]:
        for name in PERSIST_NAMES:
            src, dest = folder / name, out_root / name
            if src.is_file() and not dest.is_file():
                dest.write_bytes(src.read_bytes())

    inputs = _copy_prep(out_root)
    check = json.loads(Path(inputs["check_summary.json"]).read_text(encoding="utf-8"))
    if not check.get("all_passed"):
        print("The prep construction check did not pass. Fix it before any adapter work.")
        return 2
    names = ("train", "dev") + (("eval",) if args.phase in EVAL_PHASES else ())
    splits = {name: ensure_split(name) for name in names}

    policy, runner = _prepare_runner(
        args,
        out_root,
        extra={
            "stage": "plan4-qadapt",
            "phase": args.phase,
            "conditions": [STUDENT, TEACHER],
            "frozen_placement": (protocol or {}).get("frozen_construction"),
            "do_not_overwrite": list(PRESERVED),
            "do_not_load": list(DO_NOT_LOAD),
            "student": STUDENT,
            "teacher": TEACHER,
            "train_attention": TRAIN_ATTN,
            "generation_attention": GEN_ATTN,
            "kernel_note": KERNEL_NOTE,
            "protocol": protocol,
        },
    )
    _persist(out_root, kaggle_dest)
    print(f"device={policy.device_name} phase={args.phase}")
    run = (args, out_root, kaggle_dest, splits, runner, inputs, protocol)
    if args.phase == "smoke":
        return _run_smoke(*run)
    if args.phase == "train":
        return _run_train(*run)
    if args.phase == "select":
        return _run_select(*run)
    if args.phase == "gates":
        return _run_gates(*run)
    if args.phase in EVAL_PHASES:
        return {"freeze": _run_freeze, "eval_write": _run_eval_write, "eval_check": _run_eval_check, "eval": _run_eval}[
            args.phase
        ](*run)
    for fn in (_run_train, _run_select, _run_gates):
        rc = fn(*run)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
