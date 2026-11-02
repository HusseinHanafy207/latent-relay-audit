# Compact KV-Cache Communication Between LLM Agents

This project investigates:

**What context must a compact KV-cache message retain or reconstruct so that another agent can reliably use the sender’s intermediate results?**

## Setup

Agent A receives private operational information and calculates carrier capacities and eligibility. Agent B receives prices and a budget, then chooses the cheapest eligible carrier.

The experiments compare full KV transfer, selected result-line KV states, reconstructed states, and text containing the same results. Result positions are identified structurally; the focus is reliable reuse after selection.

Experiments use Qwen3-4B on a Tesla T4 with FP16 inference. A’s written output stays fixed within each comparison. Later experiments adapt only B’s query projections, following [Attuner](https://arxiv.org/abs/2609.36722).

Evaluation checks both **eligibility labels** and **final choices**, because a correct final answer can hide incorrect labels.

## Experiments and findings

- **Interface checks:** Prefix reuse passed, but position and framing fixes left a large reliability gap. The strongest compact condition recovered all four labels in 35/90 development cases, versus 90/90 for full KV and text.
- **Supporting context:** General instruction states improved four-carrier recovery to 87/90. However, the supported handoff failed to recover all twelve labels in any of nine twelve-carrier diagnostic cases.
- **Reconstruction and component tests:** Recomputing the selected result states improved twelve-carrier label accuracy from 63/108 to 106/108. Key/value swaps and attention interventions examined possible causes. Recovery under borrowed reference attention persisted after zeroing result values, so it did not establish successful reading of those values.
- **Receiver adaptation:** A query adapter trained on 96 sheets substantially improved reuse without reconstructing result states at inference. Removing or substituting A’s results confirmed continued dependence on the message.

On 24 new evaluation families—48 sheets, each with twelve labels:

- **Unadapted compact KV:** 356/576 labels; 0/48 completely correct sheets.
- **Adapted compact KV:** 570/576 labels; 43/48 completely correct sheets.
- **Full KV:** 572/576 labels; 45/48 completely correct sheets.
- **Result text:** 571/576 labels; 44/48 completely correct sheets.

Final choices were 33/48 for adapted KV, 34/48 for full KV, and 39/48 for text. Recovery within the predefined five-percentage-point margin remains statistically unestablished.

## Cost

After merging the adapter and connecting actual GPU-side message preparation to B, the warm same-GPU benchmark measured:

- **Adapted compact KV:** 9.501 seconds; 31.41 MiB including shared support.
- **Full KV:** 11.273 seconds; 297.33 MiB.
- **Result text:** 9.492 seconds; approximately 506 bytes.

Compact KV used about **89% less cache memory** and **15.7% less handoff time** than full KV. Its timing difference from text was inconclusive; text messages were much smaller in bytes.

Measurements used three repetitions per sheet and family-clustered comparisons. They exclude shared sender generation, training, and one-time setup. Network transfer was not measured.

## Running the experiments

Stage-specific Kaggle notebooks are in `notebooks/`, with runners in `src/stage_*.py`. Use the pinned model revision and settings in `configs/pinned.json`, and run each stage’s construction checks before evaluation.

```bash
python -m pytest tests -q
```

Keep recorded sender outputs and dataset splits unchanged. Final integrated benchmark exports are under `results_stage_plan4_qmerge/`.

## Scope

This is a controlled study using one model, one GPU class, and synthetic shipment tasks. Earlier diagnostics reused development examples; the adapter was frozen before its new-family evaluation, whose sheets were later reused for engineering checks.

The results demonstrate a reliability repair and savings relative to full KV. They do not establish that compact KV is preferable to text.