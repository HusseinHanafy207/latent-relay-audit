# latent-relay-audit

Under limited communication, when can agents use one another’s intermediate computation, especially when the sender does not initially know which results the receiver needs?

Working title: **Communicating Evidence and Intermediate Results in Multi-Agent LLM Systems**

## Overview

The study asks when B can use A’s intermediate computation under a limited message, especially when A does not yet know which results B needs. That splits into three later checks: whether A’s extra work helps B’s own subtask; which of that benefit survives text versus a compressed KV cache; and whether revealing B’s needs, or a request/reply, improves transfer enough to justify the extra cost.

Clarification is a later intervention, after diagnosing the failure. Concise intermediate results may be enough when A knows what B needs. When relevance depends on B’s private information, a narrow message can drop the useful result or its conditions. Text, latent messages, or neither can win, depending on the condition.

Completed inventory-lookup experiments (Main Results) compared compressed KV with targeted text on private evidence. They are diagnostic background. They are not the current claim.

The standing question is: when the receiver depends on evidence or intermediate results held by the sender, under what conditions does compressed latent communication offer a better accuracy–cost tradeoff than targeted or generated text handoffs?

The three-carrier Study 1 run is the simple-task reference. Communication works there, and it gave an initial cost comparison. Four-carrier written calculation and full-KV collaboration are established on `plan4_dev`. Whole-cache compression at budgets 32, 64, and 256 is a completed development diagnosis. The frozen four-way comparison on fresh families is complete. The same-span KV-versus-text run is complete. The generation-stable result-KV comparison is complete on development families. The next GPU run is one held-out evaluation of that frozen procedure on new families (`plan4_hold`). `plan4_fresh` was already used and is not an unused confirmation set.

## Current study (sparse-cache continuation diagnostic)

Agent A still has operations; Agent B still has prices and budget; neither sees the other’s private sheet. There are four carriers (P, Q, R, S) and three frozen difficulty levels:

- **L1** — capacity, refrigeration, and driver are given directly.
- **L2** — A computes effective capacity from vehicle capacity, productive hours, and loading rate.
- **L3** — productive hours are \(\max(0, H-\tau-b)\); effective capacity is \(\min(C, \lfloor r h\rfloor)\). Example: 20-box vehicle, 5 working hours, 2 hours travel, 1 hour breaks, 4 boxes/hour → 2 productive hours and effective capacity 8, which cannot handle a 12-box shipment.

Development data: `data/plan4_dev.jsonl` (30 families, 10 per level, 90 variants). Each family has a base, an A-only change, and a B-only change. A-changes mix making a carrier eligible and ineligible. B-changes mix price increases and decreases. Base and B-change reuse the same saved A states. Gold labels come from a deterministic solver. Do not rewrite `plan_dev.jsonl`.

Sender calibration is complete on the locked T4 (`results_stage_plan4_cal`, 360 rows, peak 7.68 GiB). A has 60 unique operational sheets; base and B-change reuse the same A states, so the 90-example table double-counts those 30 sheets. Report unique sheets for A. B oracle is one decision per example (90).

On unique sheets, A’s four-line `q` plus eligibility was 10/60 with no extra rollout, 9/60 with the written prompt, and 10/60 after 40 latent steps. Paired on the 90-row table: written versus direct \(\Delta=-2\) (2 fixes, 4 hurts); latent versus direct \(\Delta=-1\) (2 fixes, 3 hurts). Latent added 2.20 s of rollout and broke the `q=` format on 5/60 unique sheets (7/90 rows). Keep that: extra latent steps did not help.

By level, unique sheets:

- **L1** — quantities 19/20 direct, 17/20 written, 16/20 latent. All four lines 10/20, 9/20, 8/20. Fridge and driver labels were 25/25 on direct; capacity-below-shipment was only 4/15 (A copied `q` but still marked those carriers eligible). L1 is an easy quantity reference, not an easy eligibility reference.
- **L2** — all four lines 0/20, 0/20, 2/20. All four quantities 1/20, 1/20, 2/20. Direct often wrote `rate × hours` and skipped `min(vehicle, ·)`.
- **L3** — all four lines 0/20 in every mode, and 0/20 complete quantity sets. Mean quantities correct were 0.65–0.90 of 4.

The written prompt asked for arithmetic then the four lines. A did not do that: mean new tokens stayed ~59–62 (compact ~56), none truncated against 512, and some written messages were identical to direct.

B with solver intermediates: 76/90 (L1 25/30, L2 27/30, L3 24/30). Parse 90/90. Four trap, ten other; 86/90 still feasible and affordable. The diagnostic target was around 90%, not a validity proof. B can use correct intermediates; remaining misses are cheapest-eligible, plus four ignored ineligible lines.

Do not resume `results_stage_plan4_cal`. Keep those 360 rows.

The six-family diagnostic is complete (`results_stage_plan4_diag`, locked T4, peak 7.74 GiB). Twelve unique operational sheets; 18 examples because B-change reuses A. Written calculation produced the intended working (~460 tokens vs ~58 compact; not compute-matched). Unique sheets, all four capacities and eligibility labels:

- Direct from C0: 4/12 (L1 4/4, L2/L3 0/8).
- Written, ordinary one-turn: 12/12 (L2/L3 8/8).
- Written, cached continuation: 11/12.
- 40 latent steps then compact: 2/12. Accepting a missing `q=` is a parsing diagnostic only and raises that to 3/12; it does not explain the arithmetic failures.

A now has useful computed results to communicate. If later we keep written calculation for A and compare KV versus text handoff of those results, label that clearly: A’s computation was textual; the handoff can still be latent.

B with solver intermediates: 15/18. All three misses booked an ineligible carrier. The previous brief receiver was 17/18 on these same examples, so the long four-step plan did not help. Final answers were not truncated; 3/18 plans hit the 192-token limit with correct final answers. After rescoring, plan extraction is missing on 5/18, agrees on 11/18, disagrees on 2/18. Missing extraction is not disagreement. `**RESULTS**` headings are now counted (12/12 unique).

The identical-token cache check is complete (`results_stage_plan4_cachecheck`). Ordinary generation and a cached prefix-plus-continuation used the same input tokens on three prompts (short “ping”, easy L1, arithmetic L3). All three matched; none were truncated. Matching text is not correctness: on the L3 prompt both paths reported P as `q=40` (gold is 10), so splitting the prompt did not introduce that mistake. This checks basic prefix caching only. It does not validate the 40-step latent rollout, compression, or the full agent-to-agent handoff, and identical text does not mean every internal value or output probability matched. The earlier diagnostic’s ordinary/cached mismatch used differently formatted prompts, so it was not evidence of cache corruption. A mismatch on this check would warrant investigation; numerical differences can also change generated output. There was no mismatch here.

The oracle-B prompt comparison is complete (`results_stage_plan4_bprompt`). Keep the short line-by-line B prompt and stop tuning it. On 90 examples with program-generated intermediates: brief 76/90, lines 80/90; feasible-and-affordable 86/90 vs 89/90; mean process time 4.89 s vs 3.93 s; no plan or answer truncation. Lines fixed 9 and hurt 5. Of the 10 remaining line-prompt errors, 9 were eligible and affordable but not cheapest, and 1 booked an ineligible carrier. Six plan/answer disagreements: four corrected a wrong plan, two changed a correct plan. Forcing the answer to copy the plan would not remove those errors. 80/90 is a correct-message reference, not a ceiling for every communication method. Do not add another B-prompt branch.

The full development chain is complete (`results_stage_plan4_chain`, locked T4, peak 7.68 GiB). Freeze this task and these prompts. On 60 unique sheets, written A got all four capacities and eligibility labels on 59/60; every eligibility label was correct; none truncated. The one capacity miss did not change eligibility, and B still answered that example correctly.

On 90 variants, frozen line-B:

- A’s actual RESULT lines: 77/90 (85.6%).
- Program-generated correct message: 80/90 (88.9%).
- No message: 17/90 (18.9%).
- Another example’s message: 26/90 (28.9%).

The drop without a matched message supports that B uses information specific to A’s sheet. Changing A’s private information changed B’s answer in 25/30 families; changing B’s prices changed it in 22/30. Of the 13 collaboration errors, 10 booked a suitable affordable carrier that was not cheapest, and 3 booked an ineligible carrier. A’s transmitted `q` and eligibility were correct on those misses. The 77-versus-80 gap is not three A calculation errors: oracle lines include short reasons, while A’s handoff is capacity and eligibility only.

Collaboration `process_time_s` (3.87 s) is B only. Written A averages 30.52 s; charging A plus B is about 34.39 s, excluding network transfer. Transmitted text averages 78 bytes; the longer working stays on A.

This is a working collaboration example for the revised proposal: A computes suitability; B combines that with private prices and budget. It is text only. It does not show that KV is better, or that receiving completed calculations beats receiving source facts and recalculating. These are related development variants, not 90 independent confirmations. Do not add another prompt branch.

Text versus full KV of the same written calculation is complete (`results_stage_plan4_comm`, locked T4, peak 7.88 GiB). A’s outputs matched the chain run. Concise RESULT text 77/90 (85.6%), full KV 82/90 (91.1%), mismatched KV 28/90 (31.1%). KV fixed nine and broke four versus text. The mismatch drop supports that B uses example-specific cache content. The five-answer edge is preliminary: a paired interval that keeps families together spans both no advantage and a positive advantage.

Payloads: text ~78 bytes; full KV ~162 MiB (~1153 cache positions). Recorded processing, including the shared written generation: 32.13 s text vs 32.40 s KV. B alone was 3.72 s vs 4.00 s. Some cache cloning and CPU–GPU copies sit outside those timers; network transfer was not measured. This run does not show a speed benefit from full KV.

KV carries source facts, working, and results. Text carries only the four result lines. The finding is that the full-cache procedure scored somewhat higher than a very concise text handoff on these development examples. It does not show that internal representations beat text with comparable information.

Keep the task and prompts fixed. Fuller text plus dummy-token headwise budgets is complete (`results_stage_plan4_compress`, locked T4, peak 7.96 GiB). Same written A on every branch.

- Concise RESULT text: 77/90 (~78 B).
- Fuller text (A’s sheet plus the already-written working, no extra A generation): 81/90 (~2.4 KB).
- Full KV: 82/90 (~162 MiB, ~1153 positions).
- Dummy-token headwise 32: 28/90 (~5.06 MiB, 36 positions).
- Dummy-token headwise 64: 29/90 (~9.56 MiB, 68 positions).

Fuller text closed most of the concise-text versus KV gap. Dummy-token budgets collapsed to the mismatch range (28–29/90 vs mismatch KV 28/90). Changing A’s private facts changed B’s answer in 27/30 families under full KV, but only 3/30 at budget 32 and 5/30 at budget 64. That is a selector failure, not a verdict on question-conditioned compression.

Question-conditioned compression at those budgets is complete (`results_stage_plan4_probe`, locked T4, peak 7.88 GiB). Same written A as the dummy-token run; full KV stayed 82/90 on all 90 examples. The probe scored source and written-calculation positions and dropped its own appended keys. It saw shipment requirements, not B’s prices, budget, or gold carrier.

- Full KV: 82/90 (~162 MiB, ~1153 positions).
- Probe 32: 21/90 (~5.06 MiB, 36 positions), versus dummy-token 28/90.
- Probe 64: 29/90 (~9.56 MiB, 68 positions), the same total as dummy-token 64.
- Probe + OBF 32: 21/90.
- Probe + OBF 64: 29/90.

OBF changed individual answers (2 fixes and 2 hurts at 32; 1 and 1 at 64) but produced no net accuracy gain. Changing A’s operational facts changed B’s answer in 27/30 families under full KV, 2/30 at probe-32, and 3/30 at probe-64. Compressed B often treats an unsuitable carrier as eligible. These messages often fail to preserve A’s useful contribution. That does not say whether the information was removed or became hard to use. The probe also has to cover all four carriers, not one named inventory line.

Recorded local processing, averaging saved `process_time_s` (written A, probe, compression/OBF, B): KV 32.83 s, probe-32 32.70 s, probe-64 32.72 s, OBF-32 33.18 s, OBF-64 33.22 s. Network is not included. The small probe timing edge is not a dependable speedup; OBF adds cost. An earlier summary helper omitted probe and compression from those averages; the records were already correct, and the summary now uses them.

This is a limitation of this whole-cache compression setup on these development examples. It is not a general conclusion that OBF or latent communication fails.

The predeclared budget-256 check is complete (`results_stage_plan4_budget`, locked T4, peak 7.88 GiB). Same written A and full-KV predictions as the probe run. No truncation; compressed caches stayed finite.

- Probe 32: 21/90 (23.3%).
- Probe 64: 29/90 (32.2%).
- Probe 256: 41/90 (45.6%), 260 positions, 36.56 MiB.
- Full KV: 82/90 (91.1%), ~1153 positions, 162.09 MiB.

Versus budget 64, budget 256 fixed 19 examples and broke 7 (net +12). More space helped overall, but not every example. Trap bookings were 46/90 at budget 256 versus 3/90 with full KV. Changing A’s facts moved B in 11/30 families versus 27/30 with full KV. Recorded processing was 31.77 s for full KV versus 31.65 s for probe-256. That ~0.11 s difference is too small to claim a speed advantage, especially with the accuracy loss. Network is unmeasured. The summary averages match the individual records.

Stop increasing budgets on `plan4_dev`. Smaller-budget and OBF runs stay development diagnostics. The supported development finding is: larger compressed messages recover some performance, but this selector still sacrifices substantial accuracy while preserving only a small recorded processing-time advantage. That is a result about this tested procedure, not a verdict against latent communication generally.

The frozen comparison on fresh families is complete (`results_stage_plan4_fresh`, locked T4, peak 7.95 GiB). Same procedures, 30 disjoint families, 90 variants. Written A recovered all four capacities and eligibility labels on 59/60 unique sheets.

- Compact RESULT text: 78/90 (~78 B).
- Fuller text: 82/90 (~2.5 KB).
- Full KV: 72/90 (~163 MiB, ~1159 positions).
- Probe-256: 40/90 (~36.56 MiB, 260 positions).
- No message: 17/90.
- Mismatch: 26/90.

Changing A's facts moved B in 26/30 families under compact text, 28/30 under fuller text, 21/30 under full KV, and 10/30 at probe-256; no-message and mismatch stayed at 0/30. Recorded processing stayed about 32–33 s. Network is unmeasured.

On this split, full KV did not beat text (72 vs 78 compact and 82 fuller). Probe-256 stayed near its development result (40/90 vs 41/90) and far below text. The collaboration drop without a matched message still holds. Do not resume `results_stage_plan4_fresh`. Do not retune prompts. Do not add budgets 32/64 or OBF.

The same-span run is complete (`results_stage_plan4_span`, locked T4, peak 7.88 GiB). Same written A on every branch. Shared positions across heads. Development results, 30 families with three variants each:

- Full KV: 82/90.
- A's result lines as text: 79/90.
- KV states for those result lines: 0/90.
- Probe-selected text: 23/90.
- Probe-selected KV: 21/90.

A performed written calculations. It did not use the 40-step latent rollout. Do not resume `results_stage_plan4_span`.

The probe did not select the operational results. Across 90 examples, none of the probe-selected messages contained a numeric `q=` result line, and all contained chat-format markers such as `<|im_start|>`. That is a finding about this selector and its candidate pool, not all question-conditioned selection. Do not revisit the probe until the KV continuation path is interpretable.

The result-only KV failure is a generation breakdown, not a strict parser miss. In most examples the output is repeated `</think>` tags. All 90 selected caches were finite, so NaNs do not explain it. Removing the initial cache positions is a plausible contributor; the saved outputs cannot establish the cause.

The original `plan_all_four` metric tried to parse B's plan with A's `q=` format, so it undercounted recovery. Rescoring B's actual plan lines wrote `results_stage_plan4_span/plan_rescore.json` and did not rewrite the original span files. Capacities are not scored: B was never asked to print `q=`.

Corrected plans:

- Full KV: all four eligibility labels 90/90, all four prices 90/90, final answers 82/90. Six plans named a different carrier than the final answer.
- Result text: eligibility 90/90, prices 90/90, final answers 79/90.
- Result KV: 0/90 final answers; 87/90 outputs collapsed to thinking tags; 86/90 plans missing.
- Probe KV / probe text: eligibility labels 6/90 and 7/90. None of the 90 probe messages contained a `q=` result line; all 90 contained chat markers.

Remaining full-KV and result-text errors are not lost eligibility information. B sometimes wrote the correct labels and then failed the cheapest-eligible choice, or changed a correct plan on the answer pass. That distinction belongs in the evaluation.

The nine-example cache diagnostic is complete (`results_stage_plan4_kvdiag`, locked T4, peak 7.88 GiB). Same three families (`plan4_dev-fam-006` L1, `plan4_dev-fam-001` L2, `plan4_dev-fam-000` L3), including variants. Identity gather passed. The first four cache positions stopped the repeated-tag breakdown, but they restored usable generation more than they restored the eligibility labels.

- Full KV: 7/9 correct; eligibility 9/9; parseable 9/9.
- Full KV through the selection function, keeping every position: 7/9. Tensors, B plans, and final answers matched full KV on all nine. `gathered_equals_source` 9/9.
- Result-only KV: 0/9; all nine collapsed to repeated tags and truncated.
- Result KV plus the first four original positions: 5/9 correct; parseable 9/9; eligibility 3/9. Three of the five correct finals still had a wrong eligibility label elsewhere.
- Result text: 8/9; eligibility 9/9.

Payload: result KV about 10.89 MiB; adding the four opening positions took it to 11.45 MiB (+0.56 MiB). Those extras were prompt-opening positions, not more carrier-result lines. Token-id capture in that run intercepted `model.generate()`, which the wrapper never calls, so every artifact stayed `ids_source: retokenize` and `sequences_eq_retokenize: null`. Null meant not checked. The summary converted that into a zero count; do not read it as a mismatch. The one-position `pos_cursor` versus cache-length gap is EOS counted in `gen_lens` but not written into KV. Do not resume `results_stage_plan4_kvdiag`.

The initial-four-positions-only control is complete (`results_stage_plan4_initctrl`, locked T4, peak 7.88 GiB). Same nine examples. Token alignment is now verified: all six unique sender states used decode-loop ids whose lengths matched the cache and matched the previously reconstructed ids. Selected result positions came from those ids. The one-position `pos_cursor` gap is EOS counted but not cached.

- Full KV: 7/9 correct; eligibility 9/9; parseable 9/9.
- Result-only KV: 0/9; generation collapsed on every example.
- First four positions only: 1/9 correct; eligibility 1/9; parseable 9/9 with no truncation. Those positions are the prompt opening, not A's results.
- Result KV plus the first four positions: 5/9 correct; eligibility 3/9; parseable 9/9. Versus four-only, this fixed four answers and hurt none.
- Result text: 8/9; eligibility 9/9.

Payload: four-only 0.56 MiB; result KV 10.89 MiB; result KV plus four 11.45 MiB. Across individual carrier labels, adding result states improved correctness from 22/36 to 30/36, but three of the five correct finals still had a wrong eligibility label elsewhere. Do not resume `results_stage_plan4_initctrl`.

The 90-example development comparison is complete (`results_stage_plan4_stable`, locked T4, peak 7.88 GiB). Same five conditions, 30 families. Token-to-cache alignment passed. A's eligibility labels were correct on all 60 unique sender sheets.

- Full KV: 82/90 correct; all-four eligibility 90/90; parseable 90/90.
- Result-only KV: 0/90; outputs unusable.
- First four positions only: 23/90; parseable 90/90. Those positions are the prompt opening, not A's results.
- Result KV plus the first four positions: 42/90; parseable 90/90; all-four eligibility 25/90. Versus four-only: 25 fixes, 6 hurts (family-clustered \(\Delta=+0.21\), 95% CI 0.10–0.33).
- Result text (A's written capacity and eligibility lines, not the original inventory): 79/90; all-four eligibility 90/90.

In the result-KV-plus-four condition there were 76 ineligible-marked-eligible errors across 360 carrier judgments, and 22 of the 42 correct finals still had a wrong eligibility label elsewhere. Final-answer accuracy hides some communication failures. The 82-versus-79 gap between full KV and result text does not establish a reliable channel advantage; remaining misses after correct eligibility are B's later choice or answer production.

Payload: full KV 162.09 MiB; result KV plus four 11.31 MiB (~93% smaller); result text ~312 bytes. Recorded processing was about 35 s for those three, including A's written computation. Network transfer was not measured. Do not resume `results_stage_plan4_stable`.

The next GPU run is one held-out evaluation of this frozen procedure. Same five conditions, prompts, model, selection rule, and task distribution. New families in `data/plan4_hold.jsonl` (seed 2070), checked for no overlap with `plan4_dev` or `plan4_fresh`. `plan4_fresh` is not an unused confirmation set. Report final-answer accuracy and all-four eligibility recovery, with uncertainty calculated across families. After this, stop expanding the pilot and write it up.

```bash
python -m src.stage_plan4_hold --split plan4_hold
```

Download `/kaggle/working/results_stage_plan4_hold`. Do not write into `results_stage_plan4_stable` or earlier plan4 folders. Do not rewrite `plan4_dev.jsonl` or `plan4_fresh.jsonl`.

## Study 1 (simple-task reference)

Study 1 asked whether extra work on Agent A’s side helps Agent B on the original three-carrier task, and whether that help survives text versus uncompressed KV. C0 is A after reading its local task; C40 is 40 additional latent steps. The 40 steps are extra latent processing, not “reasoning versus no reasoning.”

On `data/plan_dev.jsonl` (12 families, 36 examples, locked T4): text C0 32/36, text C40 33/36, full KV C0 33/36, full KV C40 31/36, source-evidence 33/36, no-message 12/36, mismatched C40 text 15/36, mismatched C40 KV 14/36. \(\Delta_{\text{text}}=+1\), \(\Delta_{\text{KV}}=-2\). A’s eligibility labels were 36/36 from both C0 and C40. The KV drop sat on B-flip (prices change). Uncompressed C0 is about 24.5 MiB and 5.4 s process time; C40 is about 30.2 MiB and 8.8 s. Keep `results_stage_study1`. Do not resume that folder.

## Study 0 (completed setup)

Study 0 asked whether this task measures collaboration. Development: `data/plan_dev.jsonl`. Each family has a base item, an A-only intervention, and a B-only intervention. There is no `plan_test` file. These are 12 scenarios with three versions each, not 36 independent items. On A-flip the gold carrier becomes B’s cheapest option, so B-alone can look correct without seeing A’s change.

The first T4 run is preserved in `results_stage_plan0` (A alone 14/36, B alone 13/36, collaboration 30/36, mismatch 13/36, oracle 27/36, pooled 33/36). The six collaboration “errors” were eligible and affordable but not cheapest; the prompt had not yet stated the cost objective. Do not resume that folder.

The corrected rerun is preserved in `results_stage_plan0_corrected` (B alone 13/36, collaboration 16/36, oracle 17/36, mismatch 14/36, pooled 30/36, extra B 12/36). Keep the lowest-cost objective, shared oracle wrapper, untruncated A messages, and two-pass extra-B planning. Do not resume that folder.

A filled A-template (`P: eligible`, `Q: ineligible`, `R: eligible`) caused copying. The diagnostic in `results_stage_plan0_diag` neutralized that template. A labeled all three carriers correctly on 36/36. With evaluator-supplied eligibility, reason-then-answer reached 35/36 compact and 36/36 sentences; direct answering stayed at 17/36 and 23/36. Prompts are frozen. Keep the one compact miss (fam-004-b_flip).

The end-to-end check on `data/plan_fresh.jsonl` (6 families, 18 examples) sends A’s generated compact message to B with that frozen procedure, plus a mismatched-message control. Runner: `python -m src.stage_plan0_e2e --split plan_fresh`. Do not resume `results_stage_plan0_e2e` from Study 1.

## Main Results

The numbers below are from the earlier inventory-lookup experiments. They are completed work, not Study 0.

All main numbers are from one Kaggle Tesla T4. Cache sizes are mebibytes (bytes / 1024²), matching `mean_relay_mb` in the result JSON. Text payloads are UTF-8 bytes of the message sent to B, not the whole receiver prompt. Do not mix sizes across stages: a shorter A prompt (inventory without the question) produces a smaller full cache. Budgets 32 and 64 refer to selected prompt-cache positions per KV head and layer, in addition to four initial positions and 40 retained latent reasoning positions.

### One-record lookup, 80 held-out examples

Each example contains a 20-line inventory and a question about one package. Lines have the form “Package Luma is in locker 7.” A saw the question while encoding. Inferential unit: the example, with paired bootstrap over 80.

**Sender-only (B depends on the relay)**

| method | correct | cache |
|---|---|---|
| Full KV | 76/80 | 56.0 MiB |
| Headwise eviction, budget 32 | 45/80 | 10.69 MiB |
| OBF after eviction, budget 32 | 47/80 | 10.69 MiB |
| Headwise, budget 64 | 67/80 | 15.19 MiB |
| OBF, budget 64 | 66/80 | 15.19 MiB |

On the 80 held-out sender-only examples, full relay achieved 95% accuracy, compared with 56.25% for headwise compression at budget 32. The eviction penalty versus full relay is 38.8 points at budget 32 (95% CI 27.5–50.0) and 11.2 points at budget 64 (95% CI 1.3–21.3). OBF’s paired recovery at budget 32 is +2 examples; the CI includes 0. This is not a ranking of OBF.

**Shared inventory (B can read the document).** Full, headwise-32, and OBF-32 are all 78/80. Compression that looks acceptable here is the setting that hid the sender-only loss.

**Controls.** Direct text (B reads the inventory) is 79/80. Relaying the unmatched inventory’s full cache moved B to the donor locker 80/80 times. A predeclared no-filler check on the same 80 examples still showed a large budget-32 penalty (44/80 for both headwise and OBF, versus 79/80 full). The compression penalty persisted without filler, so filler alone does not explain the loss.

### Two-hop lookup, 40 held-out examples

Each example contains a 20-line inventory (package→shipment records and shipment→locker records) and a question about one package. A encoded the inventory **without** the question. B did not see the full inventory. No filler. Six methods per example.

On the 40 held-out two-hop examples, retrieved and generated text each achieved 40/40, full KV relay achieved 38/40, and compressed relay achieved 32/40 and 34/40 at budgets 32 and 64.

| method | correct | payload | process time |
|---|---|---|---|
| Iterative retrieved text | 40/40 | 84.5 B | 0.58 s |
| Sender-generated text | 40/40 | 242 B | 5.13 s |
| Oracle supporting text (diagnostic) | 40/40 | 84.5 B | 0.59 s |
| Full KV | 38/40 | 52.64 MiB | 3.34 s |
| Probed KV, budget 64 | 34/40 | 15.19 MiB | 3.50 s |
| Probed KV, budget 32 | 32/40 | 10.69 MiB | 3.50 s |

Reported timings cover local message preparation and receiver processing, including compression or generation where applicable. They exclude network transfer between separate machines. The retriever found both gold records on all 40 examples and sent the same two sentences as the oracle. KV process time includes A’s ~2.7 s inventory encode and 40 latent steps; text methods do not pay that rollout. Iterative text fixed the two full-relay misses and lost none. Probe-32 lost 8 paired examples versus iterative text. n=40 is still small: a few extra correct answers are not a ranking by themselves. On this structured two-hop task, targeted text was smaller, faster, and at least as accurate as the latent channel.

## Diagnostic Experiments

These sets are smaller than the held-out comparisons above. The 20-example **validate** split was used for more than one mechanism check; it is not a second independent confirmation. The original 80 Stage C items in `data/test.jsonl` stay frozen.

When A **already knew** the question (20 validate examples, budget 32): a correct-question probe, a generic probe, and recency (last 32 prompt slots) were all 20/20 under the shared answer extractor, versus 15/20 for ordinary headwise ranking. Recency did not keep the gold evidence line. Later prompt tokens and the 40 latent states can carry enough context for this format once A has processed the asked question.

When A **did not know** the question (same 20 validate examples): full KV and the correct-question probe stayed 20/20; recency and the generic probe dropped to 7/20. Question-name matching (keep the inventory line named in B’s question) was also 20/20 and kept that line on every head. The probe recovered accuracy without keeping the complete evidence line. That does **not** mean it transmitted no evidence: other cached positions, including latents, can still hold useful context. On this format, name matching already captured the accuracy gain, so the probe is not a stronger method here.

On the original 20 **development** examples, forcing the relevant inventory line inside budget 32 recovered full-relay accuracy; forcing an unrelated line did not. Ordinary headwise ranking kept a complete evidence line on almost no heads.

A no-filler development check on 10 two-hop examples (not held-out) was healthy before the 40-example run: oracle and iterative text 10/10, full KV 10/10, no-message 3/10.

## Reproduction

Pinned hardware, model, and splits are in `configs/pinned.json`. Frozen one-record test items are `data/test.jsonl`. Frozen two-hop test items are `data/hop_test.jsonl`. Do not regenerate those files. Study 1 used `data/plan_dev.jsonl`. Four-carrier development used `data/plan4_dev.jsonl`. `data/plan4_fresh.jsonl` was already evaluated and is not an unused confirmation set. The new held-out split is `data/plan4_hold.jsonl` (seed 2070). Do not rewrite `plan4_dev.jsonl` or `plan4_fresh.jsonl`. There is no `plan_test` file.

- Hardware and precision: Kaggle Tesla T4; FP16 inference; FP32 for alignment and OBF.
- Frozen model: Qwen3-4B, revision `1cfa9a7208912126459214e8b04321603b3df60c`. Upstream LatentMAS commit `36f43183e84d2fbf3b84c9bf4e3e06b7c442a523`. `transformers==4.57.3`.
- Decoding: greedy, temperature 0, `max_new_tokens` 48 for B’s locker line. Sender-generated hop messages use 128 new tokens and record truncation.
- One-record task: 20 inventory lines, four locker choices. Development n=20 (`data/dev.jsonl`); held-out n=80 (`data/test.jsonl`).
- Two-hop task: 10 package–shipment pairs (20 lines), shuffled. Development n=10 (`data/hop_dev.jsonl`); held-out n=40 (`data/hop_test.jsonl`).
- Scoring: exact letter and locker. The shared extractor uses the last `Answer: <letter>) locker <n>` line. Earlier first-match scores are stored as `correct_legacy` where that comparison exists.
- Baselines are not interchangeable. Uniform random among four choices has 25% expected accuracy. On the 20 development one-record items, gold letter C appears 8 times, so an always-C rule scores 40%. Majority-class and uniform chance are different numbers.
- Result JSON labels some cache sizes `mean_relay_mb`; those values are mebibytes (1024²), not decimal megabytes.

CPU tests:

```bash
python -m pytest tests -q
```

Completed GPU result folders (T4 logs): `results_stage_c` (80 one-record), `results_stage_hop_test` (40 two-hop), plus development and mechanism checks under `results_stage_*`. Study 0 folders, `results_stage_study1`, and plan4 folders through `results_stage_plan4_stable` are preserved. The next writer is `results_stage_plan4_hold`. Stage runners live in `src/stage_*.py`. Upstream code is fetched with `scripts/fetch_upstream.py`.

## Limitations and Planned Work

The inventories are synthetic and easy to parse. Name matching is unusually strong when the question contains the package name. n=80 and n=40 are enough for a pilot, not a publication-scale ranking. One model, one GPU class. Full-relay misses on the two-hop set show that some errors are not “information lost through compression.” Peak GPU memory is reset once per example; do not compare per-method peaks.

The current GPU run is one held-out evaluation of the frozen five-condition procedure on new families (`plan4_hold`). Do not add OBF, another probe, or more budgets. After that, stop expanding this pilot and write it up. The focused question is: when agents communicate completed intermediate results, what must a selected KV message retain to preserve task-relevant constraints, beyond merely keeping generation stable? Clarification (revealing B’s needs, or a request/reply) is a later intervention, not the default method.
