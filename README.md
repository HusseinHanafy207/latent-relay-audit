# latent-relay-audit

When agents share completed intermediate results, what must a selected KV message retain to preserve task-relevant constraints, beyond merely keeping generation stable?

Working title: **Communicating Evidence and Intermediate Results in Multi-Agent LLM Systems**

## Setup

Two frozen agents share one Qwen3-4B model on a Kaggle Tesla T4 (FP16). Agent A has operational facts; Agent B has prices and a budget. Neither sees the other’s private sheet. B must book the cheapest eligible carrier among P, Q, R, and S.

A writes the calculations, then sends a message. The compared messages are text versus KV states from that same written work. Gold labels come from a deterministic solver. Each family has a base, an A-only change, and a B-only change; those are related variants, not independent items.

Pinned hardware, model revision, and splits are in `configs/pinned.json`. Do not mix GPU types.

## Main result

A’s result lines (capacity and eligibility for each carrier) as **text** transfer those constraints. The **KV states of those same lines** do not, unless extra cache positions are kept just to make generation usable—and even then eligibility transfer stays unreliable.

Five frozen conditions, same prompts and selection rule:

| condition | development (`plan4_dev`) | held-out (`plan4_hold`) | payload |
|---|---|---|---|
| Full KV | 82/90 correct; eligibility 90/90 | 76/90; 88/90 | ~162 MiB |
| Result lines as text | 79/90; 90/90 | 74/90; 87/90 | ~310 B |
| Result KV + first 4 positions | 42/90; 25/90 | 39/90; 24/90 | ~11.3 MiB |
| First 4 positions only | 23/90; 4/90 | 23/90; 6/90 | 0.56 MiB |
| Result KV only | 0/90; generation collapsed | 0/90; collapsed | ~10.8 MiB |

The held-out split (`data/plan4_hold.jsonl`, seed 2070) is new families, disjoint from `plan4_dev` and `plan4_fresh`. `plan4_fresh` was already used and is not an unused confirmation set.

What this means:

- The first four cache positions are the prompt opening. They restore parseable generation (90/90 on both splits) but carry little of A’s results.
- Adding the result states to those four positions helps final answers (development: 25 fixes, 6 hurts versus four-only; held-out: 21 fixes, 5 hurts) but B still mislabels eligibility on most examples (25/90 and 24/90 all-four correct). Many “correct” finals still contain a wrong eligibility label elsewhere.
- Result text of the same lines recovers eligibility nearly as well as full KV. The small final-answer gap between full KV and result text does not establish a reliable channel advantage.
- A’s own eligibility labels were correct on all unique sender sheets in both evaluations. Token-to-cache alignment passed. The remaining failure is using selected result KV without the surrounding cache.
- Recorded processing is about 33–35 s including A’s written work. The KV variants reduce payload, not wall-clock time. Network transfer was not measured.

Do not resume `results_stage_plan4_stable` or `results_stage_plan4_hold`. The procedure is frozen.

## Earlier findings (background)

These are diagnostic, not the current claim.

**Collaboration is real on this task.** Sending A’s result lines beats no message and a mismatched message (77/90 vs 17/90 and 26/90 on development). Extra 40-step latent rollouts did not help A compute those lines.

**Whole-cache compression failed here.** Dummy-token and question-conditioned headwise budgets (32/64/256) stayed far below text and full KV. The probe selected chat markers, not result lines. OBF did not recover accuracy.

**Simple inventory lookup (completed earlier).** On a one-record held-out set, full KV was 76/80 and headwise-32 was 45/80 when B depended on the relay; that penalty disappeared when B could read the inventory. On two-hop lookup, targeted text was smaller, faster, and at least as accurate as KV.

## Reproduction

```bash
python -m pytest tests -q
```

Frozen files: `data/test.jsonl`, `data/hop_test.jsonl`, `data/plan_dev.jsonl`, `data/plan4_dev.jsonl`, `data/plan4_fresh.jsonl`, `data/plan4_hold.jsonl`. Do not rewrite them.

- Model: Qwen3-4B, revision `1cfa9a7208912126459214e8b04321603b3df60c`.
- Upstream LatentMAS commit `36f43183e84d2fbf3b84c9bf4e3e06b7c442a523`. `transformers==4.57.3`.
- Decoding: greedy, temperature 0.
- Cache sizes in result JSON (`mean_relay_mb`) are mebibytes (1024²).

Stage runners live in `src/stage_*.py`. Fetch upstream with `scripts/fetch_upstream.py`. Completed T4 logs are under `results_stage_*`.

## Limitations

Synthetic tasks, one model, one GPU class. n=90 is 30 families with three variants each. Name matching is unusually strong on the inventory format. Full-KV misses after correct eligibility show that some errors are B’s later choice, not lost communication. This is a pilot about this selected-KV procedure, not a verdict against latent communication in general.
