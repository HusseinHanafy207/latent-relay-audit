# latent-relay-audit

Can a compressed KV-cache message still carry a fact that the next agent cannot read in text?

Working title: **Evaluating Evidence Preservation in Compressed Latent Agent Communication**

## What this is

Agents in LatentMAS talk by passing transformer KV caches instead of writing messages. Compression (eviction, and OBF after eviction) makes those caches smaller. That is fine when both agents already see the same evidence. It may not be fine when only the first agent saw the document.

This project checks that on a small synthetic task: package–locker inventories. Agent A sees the inventory and the question. Agent B either also sees the inventory, or only sees the question plus a dummy document of similar length. We reuse A’s cache, compress it in a few ways, and measure whether B still answers correctly.

The interesting plot is B’s accuracy **without** its own copy of the inventory, as we keep fewer cache slots. The shared-inventory condition is only there to help interpret that plot.

## So far

The pipeline runs on a **Kaggle Tesla T4** with Qwen3-4B in float16. Alignment and OBF math stay in float32. Peak memory is about 10.8 GB.

On 20 development inventories, B scored 20/20 from the inventory as text and 19/20 from A’s uncompressed cache without seeing that inventory. Sending the other inventory’s cache moved B to the donor answer 17/20 times. Full relay is about 56 MB; compressed relay is 10.7 MB.

Filler widened the eviction penalty (45 points with filler, 35 without). Compression still hurts without filler. OBF’s +2 paired fixes on development are not a ranking. Gold labels on these 20 examples are uneven (C appears 8 times), so 25% is the wrong no-information baseline.

The 80 held-out examples have been run. Those numbers stay frozen. Forcing the relevant inventory line on development recovered full-relay accuracy at budget 32; forcing an unrelated line did not.

On 20 development examples, several simple selectors recovered full-relay-level accuracy at about one-fifth of the relay size, without using answer labels. On 20 fresh examples that recovery held when A already knew the question: correct probing, generic probing, and recency were all 20/20 under the shared extractor.

When A encoded the inventory **without** the question, full relay, the correct-question probe, and question-name matching were 20/20; recency and a generic probe were 7/20. Name matching kept the evidence line; the probe did not. The correct probe sent 10.69 MiB vs 49.13 MiB for full relay. Shared and original scoring agreed. Same 20 validate examples, reused, not a new independent set.

## What’s next

Rerun the same **10 hop_dev examples with no filler** in any condition, into a new folder so the filled run is not resumed. Corrected reporting: processing time includes compression; `found_both` is only the text retriever; do not compare per-method GPU peaks.

```bash
python -m src.stage_hop --split hop_dev
```

Download `/kaggle/working/results_stage_hop_nofiller`. If retrieval and oracle text stay healthy, freeze and run the 40:

```bash
python -m src.stage_hop --split hop_test
```

On Kaggle, re-upload this folder, GPU T4, internet on, then run [`notebooks/kaggle_stage_hop.ipynb`](notebooks/kaggle_stage_hop.ipynb). Run the hop_dev cell first. Run the hop_test cell only if that checklist is healthy. Do not load the Stage C test split.

