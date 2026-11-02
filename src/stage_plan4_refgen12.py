"""Reference attention under free generation on the twelve-carrier transferred cache.

Same nine cases, frozen A artifacts, generic support, compact layout, prompts,
positions, greedy decoding and output allowances (384 plan / 48 answer tokens)
as the K/V swap and teacher-forced replay runs.

Ordinary branches (tk_tv, fk_tv, fk_fv) use the existing generation path.

Lockstep branches decode a two-row batch with greedy steps:
- tk_tv_refattn: rows [fk_fv, tk_tv]. Every attention layer gives the tk_tv row
  the fk_fv row's attention weights on tk_tv's own values, during B's prompt
  processing and every decoding step. The next token is the argmax of the
  tk_tv row and is fed to both rows, so the reference follows the
  intervention's actual history. The reference never chooses tokens and the
  saved reference plan is never fed. Each row keeps its own new cache entries.
- tk_tv_selfattn: rows [tk_tv, tk_tv], the same procedure replaying its own
  attention (numerics control).

The answer (choice) pass uses each branch's own generated plan, with the same
procedure as its plan pass.

Checks on the incremental path, per case and pass: each lockstep row is
compared with an ordinary forward on that same generated history
(self-replay row vs ordinary tk_tv; reference row vs ordinary fk_fv).
Oracle diagnostic, not a method. No adapter is trained here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.attn_replay import dist_stats, label_decision, replay_attention, within_tolerance
from src.cache_utils import cache_to_device, n_kv_positions
from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import LEVELS, a_state_key
from src.plan4_timing_task import (
    N_CARRIERS_12,
    N_TIMING_FAMILIES,
    assert_timing12_invariants,
    ensure_timing12_jsonl,
    gold_eligibility_statuses_n,
)
from src.precision import apply_precision_policy, peak_gpu_memory_gb, reset_peak_memory
from src.prompts_plan4 import FRAMING_INSTRUCTION, prompt_contains_a_private
from src.span_select import caches_equal
from src.stage_plan0 import B_ANSWER_MAX_NEW, _load_messages
from src.stage_plan4_boundary import PAYLOAD_NOTE
from src.stage_plan4_kvmix12 import (
    _label_crosstab,
    _reproduction,
    _setup,
    _state_for,
    _twelve_choice,
    _twelve_label_map,
    _twelve_labels,
    _twelve_row,
)
from src.stage_plan4_kvswap import LAYOUT
from src.stage_plan4_query12 import _cluster_delta, _copy_first, _recon_refs
from src.stage_plan4_refattn12 import _batch_logits, label_tokens, response_labels
from src.stage_plan4_timing import B_PLAN_LINE_N, B_PLAN_MAX_NEW_12, _prompts_for, _row_n
from src.stage_select import _load_rows
from src.two_agent import TwoAgentInventory

REFERENCE = "fk_fv"
ORDINARY = ("tk_tv", "fk_tv", "fk_fv")
LOCKSTEP = {
    "tk_tv_refattn": {"rows": ("fk_fv", "tk_tv"), "mode": "replace"},
    "tk_tv_selfattn": {"rows": ("tk_tv", "tk_tv"), "mode": "replace"},
}
BRANCHES = ORDINARY + tuple(LOCKSTEP)
DONOR, TARGET = 0, 1
BRANCH_SPEC = {
    "tk_tv": "transferred K + transferred V, ordinary generation",
    "fk_tv": "fresh K + transferred V, ordinary generation",
    "fk_fv": "fresh K + fresh V, ordinary generation (reference)",
    "tk_tv_refattn": "transferred K + V; reference attention on its own values; picks its own tokens",
    "tk_tv_selfattn": "transferred K + V; its own attention replayed; picks its own tokens (control)",
}
TOL_MAX_MEAN_KL = 0.01
TOL_MIN_AGREE = 0.98
PAIRED = (
    ("tk_tv_refattn_vs_tk_tv", "tk_tv", "tk_tv_refattn"),
    ("tk_tv_refattn_vs_tk_tv_selfattn", "tk_tv_selfattn", "tk_tv_refattn"),
    ("tk_tv_refattn_vs_fk_tv", "fk_tv", "tk_tv_refattn"),
    ("tk_tv_refattn_vs_fk_fv", "fk_fv", "tk_tv_refattn"),
    ("fk_tv_vs_tk_tv", "tk_tv", "fk_tv"),
    ("tk_tv_selfattn_vs_tk_tv", "tk_tv", "tk_tv_selfattn"),
)
PRESERVED = (
    "results_stage_plan4_refattn12",
    "results_stage_plan4_kvmix12",
    "results_stage_plan4_query12",
    "results_stage_plan4_recon12",
    "results_stage_plan4_timing",
    "results_stage_plan4_confirm",
    "results_stage_plan4_hold",
)
INTERPRETATION_GUIDE = (
    "Free generation: the intervention chooses its own tokens and the reference row "
    "follows that history. Label figures here are free-generation accuracies; keep them "
    "separate from the teacher-forced figures of the replay run (where fresh keys gave "
    "96/108) and compare free generation with free generation (fresh keys gave 94/108). "
    "If the recovery does not survive, inspect the first step where the reference row "
    "would have chosen a different token, and whether that precedes the first label "
    "error, before training anything. Oracle diagnostic: it needs the reference cache "
    "and a second forward at every step, so it is neither efficient nor deployable."
)
PERSIST_NAMES = (
    "evaluations.jsonl",
    "case_checks.jsonl",
    "summary.json",
    "manifest.json",
    "a_artifacts_12.jsonl",
    "recon12_check_summary.json",
    "kvmix12_evaluations.jsonl",
    "refattn12_summary.json",
)


def _paths():
    timing12 = DATA_DIR / "plan4_timing12.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_refgen12" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_refgen12")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return timing12, out_root, kaggle_dest


def _persist(out_root: Path, extra_dest: Path | None = None) -> None:
    dests = [out_root.parent / "latest"]
    if extra_dest is not None:
        dests.append(extra_dest)
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        for name in PERSIST_NAMES:
            src = out_root / name
            if src.is_file() and src.resolve() != (dest / name).resolve():
                (dest / name).write_bytes(src.read_bytes())


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _copy_inputs(out_root: Path) -> dict:
    recon = [REPO_ROOT / "results_stage_plan4_recon12", Path("/kaggle/working/results_stage_plan4_recon12")]
    kvmix = [REPO_ROOT / "results_stage_plan4_kvmix12", Path("/kaggle/working/results_stage_plan4_kvmix12")]
    refattn = [REPO_ROOT / "results_stage_plan4_refattn12", Path("/kaggle/working/results_stage_plan4_refattn12")]
    inputs = {
        "artifacts": _copy_first(
            out_root,
            "a_artifacts_12.jsonl",
            [d / "a_artifacts_12.jsonl" for d in kvmix + recon + refattn],
            what="frozen 12-carrier A artifacts",
        ),
        "recon_check": _copy_first(
            out_root,
            "recon12_check_summary.json",
            [d / "check_summary.json" for d in recon] + [d / "recon12_check_summary.json" for d in refattn],
            what="recon12 check",
        ),
        "kvmix_evals": _copy_first(
            out_root,
            "kvmix12_evaluations.jsonl",
            [d / "evaluations.jsonl" for d in kvmix] + [d / "kvmix12_evaluations.jsonl" for d in refattn],
            what="K/V swap twelve-line evaluations (ordinary free generation)",
        ),
    }
    try:
        inputs["refattn_summary"] = _copy_first(
            out_root, "refattn12_summary.json", [d / "summary.json" for d in refattn], what="teacher-forced replay summary"
        )
    except FileNotFoundError:
        inputs["refattn_summary"] = None
    return inputs


def _prepare_runner(args, out_root: Path, extra: dict):
    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")
    write_manifest(
        out_root / "manifest.json",
        policy=policy,
        extra={
            "stage": "plan4-refgen12",
            "do_not_overwrite": list(PRESERVED),
            "branches": list(BRANCHES),
            "lockstep": {k: {"rows": list(v["rows"]), "mode": v["mode"]} for k, v in LOCKSTEP.items()},
            "reference": REFERENCE,
            "plan_max_new_12": args.plan_max_new_12,
            "answer_max_new": args.max_new_tokens,
            "tolerance": {"max_mean_kl": TOL_MAX_MEAN_KL, "min_argmax_agree": TOL_MIN_AGREE, "labels_equal": True},
            **extra,
        },
    )
    runner = TwoAgentInventory(
        args.model_name,
        args.device,
        latent_steps=0,
        kv_budget=32,
        sink_size=4,
        pca_rank=2,
        max_new_tokens=args.max_new_tokens,
    )
    revision = resolve_model_revision(args.model_name, model=runner.wrapper.model)
    if revision.get("mismatch"):
        print(f"Warning: pinned snapshot {revision['pinned']} != loaded {revision['loaded']}")
    if revision.get("sha"):
        pin_model_revision(revision["sha"])
    return policy, runner


def _stacked_past(bundles) -> tuple[dict, torch.Tensor, int, int]:
    first = bundles[0]
    ids0 = first.full_ids_cpu.view(-1).tolist()
    cursor = int(first.pos_cursor_cpu.view(-1)[0].item())
    n_past = n_kv_positions(first.cache_cpu)
    mask0 = first.past_mask_cpu.view(-1).tolist()
    for b in bundles[1:]:
        if b.full_ids_cpu.view(-1).tolist() != ids0 or int(b.pos_cursor_cpu.view(-1)[0].item()) != cursor:
            raise RuntimeError("Lockstep rows must share token IDs and cursor.")
        if n_kv_positions(b.cache_cpu) != n_past or b.past_mask_cpu.view(-1).tolist() != mask0:
            raise RuntimeError("Lockstep rows must share cache length and mask.")
    layers = []
    for i in range(len(first.cache_cpu["layers"])):
        k = torch.cat([b.cache_cpu["layers"][i][0] for b in bundles], dim=0)
        v = torch.cat([b.cache_cpu["layers"][i][1] for b in bundles], dim=0)
        layers.append((k, v))
    past_mask = first.past_mask_cpu.view(1, -1).long()
    if past_mask.size(1) != n_past:
        past_mask = torch.ones((1, n_past), dtype=torch.long)
    return {"layers": layers}, past_mask.expand(len(bundles), -1).contiguous(), cursor, n_past


def lockstep_generate(runner, bundles, prompt_ids, *, max_new_tokens: int, mode: str, chooser: int = TARGET) -> dict:
    """Greedy decoding of a matched batch; every row receives the chooser row's token.

    Mirrors the wrapper's generate_text_batch: the prompt is processed after the
    saved cache with the full mask, decode positions continue from the cursor,
    decoding stops at the tokenizer EOS (counted, not cached), the text is decoded
    with skip_special_tokens and stripped, and truncation is n_new >= max.
    Bundles are not mutated.
    """
    model = runner.wrapper.model
    tokenizer = runner.wrapper.tokenizer
    device = runner.device
    blob, past_mask, cursor, n_past = _stacked_past(bundles)
    n_b = len(bundles)
    past = cache_to_device(blob, device)
    n_prompt = len(prompt_ids)
    ids = torch.tensor(list(prompt_ids), dtype=torch.long, device=device).view(1, -1).expand(n_b, -1).contiguous()
    full_mask = torch.cat([past_mask.to(device), torch.ones((n_b, n_prompt), dtype=torch.long, device=device)], dim=-1)
    pos = (cursor + torch.arange(n_prompt, device=device, dtype=torch.long)).view(1, -1).expand(n_b, -1)
    eos = tokenizer.eos_token_id
    gen: list[int] = []
    step_logits: list[torch.Tensor] = []
    n_forwards = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode(), replay_attention(model, mode, donor=DONOR, target=TARGET) as ctl:
        out = model(
            input_ids=ids,
            attention_mask=full_mask,
            position_ids=pos,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        n_forwards += 1
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        for step in range(int(max_new_tokens)):
            step_logits.append(logits.detach().to("cpu", copy=True))
            tok = int(torch.argmax(logits[chooser]).item())
            gen.append(tok)
            if (eos is not None and tok == int(eos)) or step == int(max_new_tokens) - 1:
                break
            full_mask = torch.cat([full_mask, torch.ones((n_b, 1), dtype=torch.long, device=device)], dim=-1)
            nxt = torch.full((n_b, 1), tok, dtype=torch.long, device=device)
            step_pos = torch.full((n_b, 1), cursor + n_prompt + step, dtype=torch.long, device=device)
            out = model(
                input_ids=nxt,
                attention_mask=full_mask,
                position_ids=step_pos,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            n_forwards += 1
            past = out.past_key_values
            logits = out.logits[:, -1, :]
        info = {
            "mode": ctl.mode,
            "n_calls": ctl.n_calls,
            "n_replaced": ctl.n_replaced,
            "n_layers": len(ctl.layers_seen),
            "n_forwards": n_forwards,
        }
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.time() - t0
    n_layers = len(blob["layers"])
    info["every_forward_every_layer"] = info["n_calls"] == n_layers * n_forwards and info["n_layers"] == n_layers
    if mode == "replace":
        info["replaced_every_call"] = info["n_replaced"] == info["n_calls"]
    del out, past, ids, full_mask, pos, blob
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rows_logits = torch.stack(step_logits, dim=1)
    return {
        "ids": gen,
        "text": tokenizer.decode(gen, skip_special_tokens=True).strip(),
        "n_new": len(gen),
        "truncated": len(gen) >= int(max_new_tokens),
        "latency_s": latency,
        "n_past": n_past,
        "pos_cursor": cursor,
        "logits": [rows_logits[r] for r in range(n_b)],
        "info": info,
    }


def gen_char_offsets(tokenizer, ids) -> list[tuple[int, int]]:
    """Character span of each generated token in the unstripped decoded text."""
    offsets = []
    prev = 0
    for i in range(len(ids)):
        cur = max(prev, len(tokenizer.decode(ids[: i + 1], skip_special_tokens=True)))
        offsets.append((prev, cur))
        prev = cur
    return offsets


def history_labels(tokenizer, ids, carriers) -> dict:
    raw = tokenizer.decode(ids, skip_special_tokens=True)
    offsets = gen_char_offsets(tokenizer, ids)
    labels = response_labels(raw, carriers, offsets, ids)
    try:
        pair = label_tokens(labels, tokenizer)
    except RuntimeError:
        pair = None
    return {"raw": raw, "offsets": offsets, "labels": labels, "pair": pair}


def _decisions(logits, labels, pair) -> list:
    out = []
    for lab in labels:
        t = lab["token_index"]
        if t is None:
            out.append(None)
        elif pair is not None:
            out.append(label_decision(logits[t], *pair)["decision"])
        else:
            out.append(int(logits[t].argmax().item()))
    return out


def _chunked_dist(ref_logits, logits, chunk: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    kls, agrees = [], []
    for s in range(0, ref_logits.size(0), chunk):
        st = dist_stats(ref_logits[s : s + chunk], logits[s : s + chunk])
        kls.append(st["kl"])
        agrees.append(st["agree"])
    return torch.cat(kls), torch.cat(agrees)


def path_tolerance(ref_logits, logits, hist: dict) -> dict:
    """Ordinary-forward logits vs lockstep-row logits on the same generated history."""
    kl, agree = _chunked_dist(ref_logits, logits)
    a = _decisions(ref_logits, hist["labels"], hist["pair"])
    b = _decisions(logits, hist["labels"], hist["pair"])
    st = {
        "n_steps": int(kl.numel()),
        "mean_kl": float(kl.mean().item()),
        "max_kl": float(kl.max().item()),
        "argmax_agree": float(agree.float().mean().item()),
        "n_labels": len(hist["labels"]),
        "labels_equal": a == b,
        "label_method": "binary" if hist["pair"] is not None else "argmax",
    }
    st["ok"] = within_tolerance(st, max_mean_kl=TOL_MAX_MEAN_KL, min_agree=TOL_MIN_AGREE, labels_equal=a == b)
    return st


def _line_at(text: str, char_pos: int) -> tuple[int, str]:
    start = text.rfind("\n", 0, char_pos) + 1
    end = text.find("\n", char_pos)
    return text.count("\n", 0, char_pos), text[start : end if end >= 0 else len(text)]


def _line_carrier(line: str) -> str | None:
    m = B_PLAN_LINE_N.search(line)
    return m.group("carrier").upper() if m else None


def first_text_difference(a: str, b: str) -> dict | None:
    a, b = a or "", b or ""
    if a == b:
        return None
    n = min(len(a), len(b))
    i = next((j for j in range(n) if a[j] != b[j]), n)
    line_a = _line_at(a, i)
    return {
        "char": i,
        "line_index": line_a[0],
        "line_left": line_a[1],
        "line_right": _line_at(b, i)[1],
        "carrier": _line_carrier(line_a[1]) or _line_carrier(_line_at(b, i)[1]),
    }


def divergence(tokenizer, gen_ids, donor_logits, chooser_logits, hist: dict, gold: dict) -> dict:
    """Where the reference row, on the intervention's own history, would choose differently."""
    chosen = torch.tensor(gen_ids, dtype=torch.long)
    donor_arg = donor_logits.argmax(dim=-1)
    dis = (donor_arg != chosen).nonzero().view(-1).tolist()
    raw, offsets = hist["raw"], hist["offsets"]
    label_at = {lab["token_index"]: lab for lab in hist["labels"] if lab["token_index"] is not None}
    out = {"n_steps": len(gen_ids), "n_disagree_steps": len(dis), "first_disagree_step": dis[0] if dis else None}
    if dis:
        t = dis[0]
        lpd = torch.log_softmax(donor_logits[t].float(), dim=-1)
        lpc = torch.log_softmax(chooser_logits[t].float(), dim=-1)
        c_tok, r_tok = int(chosen[t]), int(donor_arg[t])
        char = offsets[t][0]
        line_index, line = _line_at(raw, char)
        out.update(
            {
                "chosen_token": tokenizer.decode([c_tok]),
                "reference_token": tokenizer.decode([r_tok]),
                "reference_logp_gap": float(lpd[r_tok] - lpd[c_tok]),
                "intervention_logp_gap": float(lpc[c_tok] - lpc[r_tok]),
                "char": char,
                "line_index": line_index,
                "line": line,
                "line_carrier": _line_carrier(line),
                "at_label_position": t in label_at,
                "text_before": raw[max(0, char - 80) : char],
            }
        )
    per = []
    for lab in sorted(hist["labels"], key=lambda x: x["token_index"] if x["token_index"] is not None else 1 << 30):
        t = lab["token_index"]
        if t is None:
            continue
        if hist["pair"] is not None:
            ref_label = label_decision(donor_logits[t], *hist["pair"])["decision"]
        else:
            ref_label = lab["ref_text_label"] if int(donor_arg[t]) == int(chosen[t]) else "other"
        g = gold[lab["carrier"]]
        per.append(
            {
                "carrier": lab["carrier"],
                "step": t,
                "gold": g,
                "generated": lab["ref_text_label"],
                "reference_on_history": ref_label,
                "generated_correct": lab["ref_text_label"] == g,
                "reference_correct": ref_label == g,
                "agree": ref_label == lab["ref_text_label"],
            }
        )
    first_err = next((p for p in per if not p["generated_correct"]), None)
    out.update(
        {
            "labels": per,
            "n_labels_located": len(per),
            "labels_reference_agrees": sum(p["agree"] for p in per),
            "labels_generated_correct": sum(p["generated_correct"] for p in per),
            "labels_reference_correct_on_history": sum(p["reference_correct"] for p in per),
            "first_label_error_step": None if first_err is None else first_err["step"],
            "first_label_error_carrier": None if first_err is None else first_err["carrier"],
            "disagreement_precedes_first_label_error": (
                None if first_err is None or not dis else bool(dis[0] < first_err["step"])
            ),
        }
    )
    return out


def _lockstep_row(runner, ex, caches, branch, args) -> tuple[dict, dict]:
    spec = LOCKSTEP[branch]
    bundles = [caches[name] for name in spec["rows"]]
    tokenizer = runner.wrapper.tokenizer
    prompts = _prompts_for(ex)
    plan_prompt = prompts["kv_plan"]
    if prompt_contains_a_private(plan_prompt, ex):
        raise RuntimeError("KV B leaked A's raw sheet.")
    _r, plan_ids_t, _a = runner._encode(plan_prompt)
    plan_prompt_ids = [int(t) for t in plan_ids_t[0].tolist()]
    plan = lockstep_generate(runner, bundles, plan_prompt_ids, max_new_tokens=args.plan_max_new_12, mode=spec["mode"])
    prompt = prompts["kv_answer"](ex, plan["text"])
    if prompt_contains_a_private(prompt, ex):
        raise RuntimeError("KV answer pass leaked A's raw sheet.")
    _r, ans_ids_t, _a = runner._encode(prompt)
    ans_prompt_ids = [int(t) for t in ans_ids_t[0].tolist()]
    ans = lockstep_generate(runner, bundles, ans_prompt_ids, max_new_tokens=args.max_new_tokens, mode=spec["mode"])

    gold = gold_eligibility_statuses_n(ex)
    checks = {}
    extra_diag = {}
    for pass_name, res, p_ids in (("plan", plan, plan_prompt_ids), ("answer", ans, ans_prompt_ids)):
        hist = history_labels(tokenizer, res["ids"], ex.carriers)
        checked = [(r, name) for r, name in enumerate(spec["rows"]) if r == DONOR or spec["rows"][DONOR] == name]
        ordinary = {}
        for name in {name for _r, name in checked}:
            logits, _ = _batch_logits(runner, [caches[name]], p_ids, res["ids"], mode=None)
            ordinary[name] = logits[0]
        for r, name in checked:
            role = "donor" if r == DONOR else "target"
            checks[f"{pass_name}_{role}_vs_ordinary_{name}"] = path_tolerance(ordinary[name], res["logits"][r].float(), hist)
        checks[f"{pass_name}_instrumentation"] = {
            **res["info"],
            "ok": bool(res["info"]["every_forward_every_layer"] and res["info"].get("replaced_every_call", True)),
        }
        if pass_name == "plan":
            extra_diag["divergence"] = divergence(
                tokenizer, res["ids"], res["logits"][DONOR].float(), res["logits"][TARGET].float(), hist, gold
            )
            extra_diag["plan_labels_located"] = len(hist["labels"])
            extra_diag["plan_label_method"] = "binary" if hist["pair"] is not None else "argmax"
        del ordinary
    extra = {
        "channel": "kv",
        "message_text": "",
        "message_bytes": int(caches["tk_tv"].full_relay_bytes),
        "n_kv_positions": plan["n_past"],
        "plan_time_s": plan["latency_s"],
        "n_plan_tokens": plan["n_new"],
        "plan_truncated": plan["truncated"],
        "plan_text": plan["text"],
        "plan_max_new": args.plan_max_new_12,
        "truncated": ans["truncated"],
        "n_new_tokens": ans["n_new"],
        "receiver_latency_s": ans["latency_s"],
        "max_new_tokens": args.max_new_tokens,
        "peak_gpu_gb": peak_gpu_memory_gb(),
    }
    rec = _row_n(ex, prompt=prompt, text=ans["text"], extra=extra)
    rec.update(
        {
            "condition": branch,
            "decoding": "lockstep",
            "lockstep_rows": list(spec["rows"]),
            "replay_mode": spec["mode"],
            "token_chooser": "target row (intervention)",
            "plan_ids": plan["ids"],
            "answer_ids": ans["ids"],
            "pos_cursor": plan["pos_cursor"],
            "layout": LAYOUT,
            "framing": "instruct",
            "framing_instruction": FRAMING_INSTRUCTION,
            "payload_note": PAYLOAD_NOTE,
            **extra_diag,
        }
    )
    del plan, ans
    return rec, checks


def _case(runner, ex, artifacts, setup, recon_refs, args) -> dict:
    state, built, mix = _state_for(runner, ex, artifacts, setup, recon_refs)
    if not (mix["ok"] and built["matches_recon12"]):
        raise RuntimeError(f"Mixed-cache construction failed for {ex.example_id}.")
    caches = state["kvmix"]
    snapshots = {c: runner.clone_sender(caches[c]) for c in ORDINARY}
    reset_peak_memory()
    rows = {}
    for cond in ORDINARY:
        rec = _twelve_row(runner, ex, state, cond, args)
        rec["decoding"] = "ordinary"
        rows[cond] = rec
    checks = {}
    for branch in LOCKSTEP:
        rec, ch = _lockstep_row(runner, ex, caches, branch, args)
        rows[branch] = rec
        for k, v in ch.items():
            checks[f"{branch}:{k}"] = v
    checks["caches_unchanged"] = all(caches_equal(snapshots[c].cache_cpu, caches[c].cache_cpu) for c in ORDINARY)
    ok = all(v["ok"] if isinstance(v, dict) else bool(v) for v in checks.values())
    self_rec, tk_rec = rows["tk_tv_selfattn"], rows["tk_tv"]
    ref_rec, int_rec = rows["fk_fv"], rows["tk_tv_refattn"]
    comparisons = {
        "selfattn_plan_identical_to_ordinary_tk_tv": self_rec["plan_text"] == tk_rec["plan_text"],
        "selfattn_answer_identical_to_ordinary_tk_tv": self_rec["raw_text"] == tk_rec["raw_text"],
        "selfattn_labels_equal_ordinary_tk_tv": self_rec["plan_pred_status"] == tk_rec["plan_pred_status"],
        "selfattn_vs_tk_tv_first_text_difference": first_text_difference(tk_rec["plan_text"], self_rec["plan_text"]),
        "refattn_plan_identical_to_ordinary_fk_fv": int_rec["plan_text"] == ref_rec["plan_text"],
        "refattn_labels_equal_ordinary_fk_fv": int_rec["plan_pred_status"] == ref_rec["plan_pred_status"],
        "refattn_vs_fk_fv_first_text_difference": first_text_difference(ref_rec["plan_text"], int_rec["plan_text"]),
    }
    for rec in rows.values():
        rec["instrumentation_ok"] = ok
    check_row = {
        "example_id": ex.example_id,
        "a_state_key": a_state_key(ex),
        "ok": ok,
        "checks": checks,
        "comparisons": comparisons,
        "divergence": int_rec.get("divergence"),
    }
    del state, caches, snapshots
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"rows": [rows[b] for b in BRANCHES], "check": check_row}


def _run(args, out_root, kaggle_dest, examples, runner, artifacts, recon_refs, inputs) -> int:
    setup = _setup(runner, examples, artifacts)
    eval_path = out_root / "evaluations.jsonl"
    check_path = out_root / "case_checks.jsonl"
    checks = _load_rows(check_path)
    done = {c["example_id"] for c in checks}
    saved = _load_rows(eval_path)
    rows = [r for r in saved if r["example_id"] in done]
    if len(rows) != len(saved):
        eval_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    todo = [ex for ex in examples if ex.example_id not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"Refgen12: {len(done)}/{len(examples)} cases complete; running {len(todo)}", flush=True)
    rc = 0
    for i, ex in enumerate(todo):
        res = _case(runner, ex, artifacts, setup, recon_refs, args)
        for row in res["rows"]:
            rows.append(row)
            _append(eval_path, row)
        checks.append(res["check"])
        _append(check_path, res["check"])
        c = res["check"]
        print(f"  [{i + 1}/{len(todo)}] {ex.example_id} instrumentation_ok={c['ok']}", flush=True)
        for k, v in c["checks"].items():
            if isinstance(v, dict) and "mean_kl" in v:
                print(f"    {k}: ok={v['ok']} meanKL={v['mean_kl']:.5f} agree={v['argmax_agree']:.4f} labels_equal={v['labels_equal']}")
            elif isinstance(v, dict):
                print(f"    {k}: ok={v['ok']}")
            else:
                print(f"    {k}: {v}")
        for row in res["rows"]:
            print(
                f"    {row['condition']}: labels={row.get('plan_n_status_correct')}/12 "
                f"I->E={row.get('plan_ineligible_marked_eligible')} E->I={row.get('plan_eligible_marked_ineligible')} "
                f"parse={row.get('parse_ok')} plan_trunc={row.get('plan_truncated')} choice={row.get('correct')}",
                flush=True,
            )
        d = c.get("divergence") or {}
        print(
            f"    divergence: first_step={d.get('first_disagree_step')} n_disagree={d.get('n_disagree_steps')} "
            f"line={d.get('line')!r} ref_agrees_labels={d.get('labels_reference_agrees')}/{d.get('n_labels_located')}",
            flush=True,
        )
        _persist(out_root, kaggle_dest)
        if not c["ok"] and not args.force:
            print("Incremental-path check failed for this case; stopping. Inspect case_checks.jsonl.")
            rc = 1
            break
    summary = _summarize(rows, checks, kvmix_rows=_load_rows(inputs["kvmix_evals"]), refattn_summary=_refattn(inputs))
    _write_json(out_root / "summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps(summary["labels"], indent=2))
    return rc


def _refattn(inputs) -> dict | None:
    path = inputs.get("refattn_summary")
    if path is None or not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _case_metric(rows: list[dict], field: str) -> dict:
    return {r["example_id"]: int(r.get(field) or 0) for r in rows}


def _divergence_block(checks: list[dict]) -> dict:
    divs = [c["divergence"] for c in checks if c.get("divergence")]
    if not divs:
        return {"n_cases": 0}
    firsts = [d["first_disagree_step"] for d in divs if d["first_disagree_step"] is not None]
    precede = [d["disagreement_precedes_first_label_error"] for d in divs]
    return {
        "n_cases": len(divs),
        "cases_never_disagree": sum(d["first_disagree_step"] is None for d in divs),
        "first_disagree_steps": firsts,
        "first_disagree_at_label_position": sum(bool(d.get("at_label_position")) for d in divs),
        "total_disagree_steps": sum(d["n_disagree_steps"] for d in divs),
        "total_plan_steps": sum(d["n_steps"] for d in divs),
        "labels_located": sum(d["n_labels_located"] for d in divs),
        "labels_reference_agrees_on_history": sum(d["labels_reference_agrees"] for d in divs),
        "labels_reference_correct_on_history": sum(d["labels_reference_correct_on_history"] for d in divs),
        "labels_generated_correct": sum(d["labels_generated_correct"] for d in divs),
        "cases_with_label_error": sum(d["first_label_error_step"] is not None for d in divs),
        "cases_disagreement_precedes_first_label_error": sum(p is True for p in precede),
        "per_case": {
            c["example_id"]: {
                k: c["divergence"].get(k)
                for k in (
                    "first_disagree_step",
                    "n_disagree_steps",
                    "chosen_token",
                    "reference_token",
                    "reference_logp_gap",
                    "intervention_logp_gap",
                    "line_index",
                    "line",
                    "line_carrier",
                    "at_label_position",
                    "first_label_error_step",
                    "first_label_error_carrier",
                    "disagreement_precedes_first_label_error",
                    "labels_reference_agrees",
                    "n_labels_located",
                )
            }
            for c in checks
            if c.get("divergence")
        },
    }


def _teacher_forced_context(refattn_summary: dict | None, kvmix_rows: list[dict]) -> dict:
    tf = {}
    if refattn_summary:
        for b, block in (refattn_summary.get("branches") or {}).items():
            tf[b] = {k: block.get(k) for k in ("labels_correct", "n_labels", "ineligible_marked_eligible", "eligible_marked_ineligible")}
    by: dict[str, list[dict]] = defaultdict(list)
    for r in kvmix_rows:
        by[r["condition"]].append(r)
    free = {c: _twelve_labels(by[c]) for c in ("tk_tv", "fk_tv", "tk_fv", "fk_fv") if by[c]}
    return {
        "teacher_forced_on_reference_plan_refattn12": tf,
        "free_generation_kvmix12": free,
        "note": (
            "Teacher-forced figures (refattn12) score label decisions on the reference's saved "
            "plan; free-generation figures (kvmix12 and this run) score each branch's own text. "
            "Fresh keys: 96/108 teacher-forced vs 94/108 free generation. Do not mix them."
        ),
    }


def _summarize(rows: list[dict], checks: list[dict], *, kvmix_rows: list[dict], refattn_summary: dict | None) -> dict:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    maps = {b: _twelve_label_map(by[b]) for b in BRANCHES}
    paired, cross = {}, {}
    for name, left, right in PAIRED:
        paired[name] = {
            metric: _cluster_delta(_case_metric(by[left], field), _case_metric(by[right], field))
            for metric, field in (
                ("labels_correct", "plan_n_status_correct"),
                ("ineligible_marked_eligible", "plan_ineligible_marked_eligible"),
                ("eligible_marked_ineligible", "plan_eligible_marked_ineligible"),
            )
        }
        cross[name] = _label_crosstab(maps[left], maps[right])
    reproduction = {
        cond: _reproduction(
            by[cond],
            [r for r in kvmix_rows if r.get("condition") == cond],
            key_fields=("example_id",),
            text_fields=("plan_text", "raw_text"),
        )
        for cond in ORDINARY
    }
    comp = [c.get("comparisons") or {} for c in checks]
    return {
        "n_rows": len(rows),
        "n_cases": len({r["example_id"] for r in rows}),
        "n_expected_cases": N_TIMING_FAMILIES,
        "reference": REFERENCE,
        "branch_spec": BRANCH_SPEC,
        "instrumentation_ok": bool(checks) and all(bool(c.get("ok")) for c in checks),
        "n_cases_instrumentation_ok": sum(bool(c.get("ok")) for c in checks),
        "labels": {b: _twelve_labels(by[b]) for b in BRANCHES},
        "final_choice": {b: _twelve_choice(by[b]) for b in BRANCHES},
        "by_difficulty": {
            lvl: {b: _twelve_labels([r for r in by[b] if r.get("difficulty") == lvl]) for b in BRANCHES} for lvl in LEVELS
        },
        "paired_by_case": paired,
        "label_crosstab": cross,
        "reproduction_vs_kvmix12": reproduction,
        "self_replay_free_generation": {
            "n_cases": len(comp),
            "plan_identical_to_ordinary_tk_tv": sum(bool(c.get("selfattn_plan_identical_to_ordinary_tk_tv")) for c in comp),
            "answer_identical_to_ordinary_tk_tv": sum(bool(c.get("selfattn_answer_identical_to_ordinary_tk_tv")) for c in comp),
            "labels_equal_ordinary_tk_tv": sum(bool(c.get("selfattn_labels_equal_ordinary_tk_tv")) for c in comp),
        },
        "intervention_vs_reference_text": {
            "n_cases": len(comp),
            "plan_identical_to_ordinary_fk_fv": sum(bool(c.get("refattn_plan_identical_to_ordinary_fk_fv")) for c in comp),
            "labels_equal_ordinary_fk_fv": sum(bool(c.get("refattn_labels_equal_ordinary_fk_fv")) for c in comp),
        },
        "divergence": _divergence_block(checks),
        "context": _teacher_forced_context(refattn_summary, kvmix_rows),
        "measurement_note": (
            "Free generation. Labels are parsed from each branch's own twelve-line plan; the "
            "final choice comes from each branch's own answer pass on its own plan. Lockstep "
            "branches choose tokens from the intervention row; the reference row follows that "
            "history and never supplies tokens. Final-choice accuracy is reported separately."
        ),
        "stats_note": (
            "Bootstrap resamples the nine source cases (2000 draws); labels are not independent. "
            "Deltas are right minus left; cases_increased / cases_decreased give direction only "
            "(for error counts an increase is worse)."
        ),
        "tolerance": {"max_mean_kl": TOL_MAX_MEAN_KL, "min_argmax_agree": TOL_MIN_AGREE, "labels_equal": True},
        "interpretation_guide": INTERPRETATION_GUIDE,
    }


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Reference attention under free generation (twelve carriers).")
    parser.add_argument("--split", default="plan4_timing12")
    parser.add_argument("--phase", choices=("run", "rescore"), default="run")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--plan_max_new_12", type=int, default=B_PLAN_MAX_NEW_12)
    parser.add_argument("--limit", type=int, default=None, help="Run at most this many remaining cases.")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--force", action="store_true", help="Continue after a failed incremental-path check.")
    args = parser.parse_args()
    if args.split != "plan4_timing12":
        print("Refgen12 uses only the frozen timing-12 sheets. Do not load plan4_dev or plan4_hold.")
        return 2
    if args.plan_max_new_12 != B_PLAN_MAX_NEW_12 or args.max_new_tokens != B_ANSWER_MAX_NEW:
        print("Keep the output allowances of the earlier runs (384 plan / 48 answer tokens).")
        return 2

    timing12_path, out_root, kaggle_dest = _paths()
    if "stage_plan4_refgen12" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_refgen12.")
        return 2
    if args.fresh and out_root.exists():
        out_root = out_root.parent / time.strftime("%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    if kaggle_dest is not None:
        for name in PERSIST_NAMES:
            src, dest = kaggle_dest / name, out_root / name
            if src.is_file() and not dest.is_file():
                dest.write_bytes(src.read_bytes())

    examples = ensure_timing12_jsonl(timing12_path)
    for ex in examples:
        assert_timing12_invariants(ex)
    if len(examples) != N_TIMING_FAMILIES:
        print(f"Expected {N_TIMING_FAMILIES} twelve-carrier sheets, got {len(examples)}.")
        return 2

    if args.phase == "rescore":
        shared = REPO_ROOT / "results_stage_plan4_refgen12"
        if not (out_root / "evaluations.jsonl").is_file() and (shared / "evaluations.jsonl").is_file():
            out_root = shared
        inputs = {
            "kvmix_evals": out_root / "kvmix12_evaluations.jsonl",
            "refattn_summary": out_root / "refattn12_summary.json",
        }
        summary = _summarize(
            _load_rows(out_root / "evaluations.jsonl"),
            _load_rows(out_root / "case_checks.jsonl"),
            kvmix_rows=_load_rows(inputs["kvmix_evals"]),
            refattn_summary=_refattn(inputs),
        )
        _write_json(out_root / "summary.json", summary)
        _persist(out_root, kaggle_dest)
        print(json.dumps(summary["labels"], indent=2))
        return 0

    inputs = _copy_inputs(out_root)
    artifacts = _load_messages(inputs["artifacts"])
    if any(a_state_key(ex) not in artifacts for ex in examples):
        print("Frozen 12-carrier artifacts are missing. Do not regenerate A.")
        return 2
    recon_refs = _recon_refs(inputs["recon_check"])

    policy, runner = _prepare_runner(args, out_root, extra={"phase": args.phase, "n_carriers": N_CARRIERS_12})
    _persist(out_root, kaggle_dest)
    print(f"device={policy.device_name} phase={args.phase} n={len(examples)}")
    return _run(args, out_root, kaggle_dest, examples, runner, artifacts, recon_refs, inputs)


if __name__ == "__main__":
    raise SystemExit(main())
