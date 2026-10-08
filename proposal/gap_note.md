# Gap note (Phase 1)

**Working title:** Evaluating Evidence Preservation in Compressed Latent Agent Communication

## What is already shown

LatentMAS relays layer-wise KV caches: the sender’s cache holds input-derived states plus later latent-computation states, and the receiver prepends that cache rather than re-encoding the sender’s text. Deleting the cache slots of an evidence line therefore does not prove the fact is gone.

Li, An, and Du compress **prompt** states only, leaving latent reasoning and inherited history intact. Full relay, attention-based (H2O-style) eviction, and Orthogonal BackFill (OBF) are compared on shared-question benchmarks. Compression can change accuracy; OBF often recovers part of what eviction drops. Those evaluations do not manipulate whether the receiver can see the evidence in text.

The causal audit of relayed KV caches already answers a closely related question: pairing of the sender’s cache with the query matters when the receiver lacks sender-private facts, and is near-null when the receiver can solve the item alone. Direct-text, sender-only, and mismatch controls are their instrument, not a new discovery. They intervene on **uncompressed** caches.

## What is not shown

The missing comparison is the **interaction**: receiver evidence access × prompt-cache budget × OBF recovery.

That is the proposed contribution. It is not “communication matters more when the receiver lacks the information” (already shown). It is whether a compression configuration that looks acceptable under shared information remains reliable when the receiver depends on the relay, and whether OBF changes that boundary.

The main result to present is sender-only accuracy as the communication budget decreases, for ordinary eviction versus OBF. The shared condition interprets that curve; if it sits at ceiling, that is expected and must not carry the claim.

## Scope of this study

One frozen model, a synthetic inventory lookup, a two-agent sequential handoff (a controlled adaptation, not a LatentMAS reproduction), and a small sample. Enough for an initial proposal; not a publication-scale claim.
