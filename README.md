# latent-relay-audit

How reliably do compressed latent messages transfer private evidence between LLM agents, and how do they compare with targeted text handoffs?

Working title: **Evaluating Evidence Preservation in Compressed Latent Agent Communication**

## Overview

Agents in LatentMAS talk by passing transformer KV caches instead of writing messages. Compression (eviction, and Orthogonal BackFill after eviction) makes those caches smaller by dropping prompt-cache slots while keeping later latent states. Shared evidence can hide those losses: if B already sees the document, B may still answer after eviction. That does not mean compression is harmless when B depends on the relay.

This project measures that case on synthetic package–locker inventories. Agent A reads an inventory. Agent B must name a locker and never receives the full inventory as text. We reuse A’s KV cache, compress it, or replace it with a short text handoff, and score B’s answer.

Two question-timing settings matter:

- **Question during encoding.** A sees the inventory and B’s question before latent reasoning. Later cache entries can absorb information that is useful for that specific question.
- **Question withheld during encoding.** A reads the inventory only. The question becomes available later, for selection or for B. This is the stricter test of whether a compressed cache still holds facts A did not yet know would be asked.

The plot that carries the claim is B’s accuracy **without** its own copy of the inventory, as the message shrinks. A shared-inventory panel, where B also sees the document, is only there to interpret that plot.

## Main findings

All main numbers are from a **Kaggle Tesla T4**, Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`, FP16 inference, FP32 for alignment and OBF. Cache sizes below are mebibytes (bytes / 1024²), matching the `mean_relay_mb` fields in the result JSON. Text payloads are UTF-8 bytes of the message sent to B, not the whole receiver prompt. Do not mix sizes across stages: a shorter A prompt (inventory without the question) produces a smaller full cache.

### One-record lookup, 80 held-out examples (Stage C)

Each item is one inventory line of the form “Package Luma is in locker 7.” A saw the question while encoding. B either also saw the inventory (shared) or saw only the question plus filler (sender-only). Inferential unit: the example, with paired bootstrap over 80.

**Sender-only (B depends on the relay)**

| method | correct | cache |
|---|---|---|
| Full KV | 76/80 | 56.0 MiB |
| Headwise eviction, budget 32 | 45/80 | 10.69 MiB |
| OBF after eviction, budget 32 | 47/80 | 10.69 MiB |
| Headwise, budget 64 | 67/80 | 15.19 MiB |
| OBF, budget 64 | 66/80 | 15.19 MiB |

The eviction penalty versus full relay is 38.8 points at budget 32 (95% CI 27.5–50.0) and 11.2 points at budget 64 (95% CI 1.3–21.3). OBF’s paired recovery at budget 32 is +2 examples; the CI includes 0. This is not a ranking of OBF.

**Shared inventory (B can read the document).** Full, headwise-32, and OBF-32 are all 78/80. Compression that looks acceptable here is the setting that hid the sender-only loss.

**Controls.** Direct text (B reads the inventory) is 79/80. Relaying the unmatched inventory’s full cache moved B to the donor locker 80/80 times. A predeclared no-filler check on the same 80 examples still showed a large budget-32 penalty (44/80 for both headwise and OBF, versus 79/80 full). Filler is not the cause.

### Two-hop lookup, 40 held-out examples

Each inventory has 20 lines: package→shipment records and shipment→locker records. A encoded the inventory **without** the question. B never saw the full inventory. No filler. Six methods per example.

| method | correct | payload | process time |
|---|---|---|---|
| Iterative retrieved text | 40/40 | 84.5 B | 0.58 s |
| Sender-generated text | 40/40 | 242 B | 5.13 s |
| Oracle supporting text (diagnostic) | 40/40 | 84.5 B | 0.59 s |
| Full KV | 38/40 | 52.64 MiB | 3.34 s |
| Probed KV, budget 64 | 34/40 | 15.19 MiB | 3.50 s |
| Probed KV, budget 32 | 32/40 | 10.69 MiB | 3.50 s |

The retriever found both gold records on all 40 examples and sent the same two sentences as the oracle. KV process time includes A’s ~2.7 s inventory encode and 40 latent steps; text methods do not pay that rollout. Iterative text fixed the two full-relay misses and lost none. Probe-32 lost 8 paired examples versus iterative text. n=40 is still small: a few extra correct answers are not a ranking by themselves. On this structured two-hop task, targeted text was smaller, faster, and at least as accurate as the latent channel.

A no-filler development check on the same 10 hop examples (not held-out) was healthy before this run: oracle and iterative text 10/10, full KV 10/10, no-message 3/10.

### Question timing and cheap selectors (development and validation)

These are smaller sets. The 20-example **validate** split was used for more than one mechanism check; it is not a second independent confirmation. The original 80 Stage C items in `data/test.jsonl` stay frozen.

When A **already knew** the question (20 validate examples, budget 32): a correct-question probe, a generic probe, and recency (last 32 prompt slots) were all 20/20 under the shared answer extractor, versus 15/20 for ordinary headwise ranking. Recency did not keep the gold evidence line. Later prompt tokens and the 40 latent states can carry enough context for this format once A has processed the asked question.

When A **did not know** the question (same 20 validate examples): full KV and the correct-question probe stayed 20/20; recency and the generic probe dropped to 7/20. Question-name matching (keep the inventory line named in B’s question) was also 20/20 and kept that line on every head. The probe recovered accuracy without keeping the complete evidence line. That does **not** mean it transmitted no evidence: other cached positions, including latents, can still hold useful context. On this format, name matching already captured the accuracy gain, so the probe is not a stronger method here.

On the original 20 **development** examples, forcing the relevant inventory line inside budget 32 recovered full-relay accuracy; forcing an unrelated line did not. Ordinary headwise ranking kept a complete evidence line on almost no heads.

## Experiment setup

- Frozen model: Qwen3-4B, revision above. Upstream LatentMAS commit `36f43183e84d2fbf3b84c9bf4e3e06b7c442a523`. `transformers==4.57.3`.
- Decoding: greedy, temperature 0, `max_new_tokens` 48 for B’s locker line. Sender-generated hop messages use 128 new tokens and record truncation.
- One-record task: 20 inventory lines, four locker choices. Development n=20 (`data/dev.jsonl`); held-out n=80 (`data/test.jsonl`).
- Two-hop task: 10 package–shipment pairs (20 lines), shuffled. Development n=10 (`data/hop_dev.jsonl`); held-out n=40 (`data/hop_test.jsonl`).
- Scoring: exact letter and locker. The shared extractor uses the last `Answer: <letter>) locker <n>` line. Earlier first-match scores are stored as `correct_legacy` where that comparison exists.
- Baselines are not interchangeable. Uniform random among four choices has 25% expected accuracy. On the 20 development one-record items, gold letter C appears 8 times, so an always-C rule scores 40%. Majority-class and uniform chance are different numbers.
- Result JSON labels some cache sizes `mean_relay_mb`; those values are mebibytes (1024²), not decimal megabytes.

## Reproduction

Pinned hardware, model, and splits are in `configs/pinned.json`. Frozen one-record test items are `data/test.jsonl`. Frozen two-hop test items are `data/hop_test.jsonl`. Do not regenerate those files.

CPU tests:

```bash
python -m pytest tests -q
```

Completed GPU result folders (T4 logs): `results_stage_c` (80 one-record), `results_stage_hop_test` (40 two-hop), plus development and mechanism checks under `results_stage_*`. Stage runners live in `src/stage_*.py`. Upstream code is fetched with `scripts/fetch_upstream.py`.

## Limitations and planned work

The inventories are synthetic and easy to parse. Name matching is unusually strong when the question contains the package name. n=80 and n=40 are enough for a pilot, not a publication-scale ranking. One model, one GPU class. Full-relay misses on the two-hop set show that some errors are not “information lost through compression.” Peak GPU memory is reset once per example; do not compare per-method peaks.

**Planned, not run.** Next is prose-based evidence transfer: documents where the supporting fact is not a single templated line. If those feasibility checks pass, a later step is intermediate-computation transfer, where B needs a result A computed rather than a retrieved record. Those are labeled plans. They are not the current next command.
