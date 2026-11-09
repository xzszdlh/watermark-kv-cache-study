# BART KV-cache compartment ablation

This is a new, explicitly implemented cache intervention study, not a rerun of the old no-op `kv_compress` parameter. It does not reproduce HalluKV, HeadKV, H2O or StreamingLLM. The encoder-decoder architecture separates source cross-attention from generated-history self-attention, unlike a decoder-only shared attention pool.

## Fixed design

- Same summarization-finetuned BART checkpoint, full model-visible 1,024-token source input and exact original 200 calibration + 500 test documents.
- Nucleus sampling top-p=.9, temperature=1, max_new_tokens=128, gamma=.25.
- Deltas 0 and 2, seeds 42/43/44, batch size 32 for **all** cache conditions.
- Four 2x2 conditions: `full`, `cross-half`, `self-half`, `both-half`; each has 4,200 summaries. Total 16,800 outputs, still 500 independent test documents.
- Per-condition no-watermark calibration and within-condition watermark comparisons. Cross-condition contrasts additionally use the full-cache condition's fixed detection threshold and paired document-level changes.
- Existing sensitivity jobs and their files are not modified. This worker uses GPU 2 with a 22% allocator cap and checks for a further 6 GiB free at admission. No controlled latency claims while sharing a GPU.

## What is actually compressed

### Source cross-attention K/V

Encode the **entire** source first. For each decoder layer, project full source keys/values once, physically gather `ceil(0.5 * valid_source_length)` uniformly spaced original positions, and discard the full K/V tensors. Preserve source padding masks and per-document retained-position lists. Both endpoints are retained when at least two positions are kept.

A reduced view of the already-contextualized encoder output provides the matching source dimension required by HF's cached-cross-attention reuse path. The original source is never re-encoded after pruning. Projection hooks assert exactly one K and V projection per layer, and data pointers assert the compressed cross cache is reused rather than silently regenerated on every step. Storage-size assertions check that slices do not retain the full backing allocation.

Retained encoder states can contain information from discarded source positions. Thus this is cross-attention memory downsampling, not literal deletion of source facts and not a guarantee of hallucinations.

### Generated-history self-attention K/V

After each decoder step, keep a budget of `max(1,ceil(0.5 * positions_seen))`: the decoder-start attention sink plus the most recent cached positions. Physically compact K/V tensors with `index_select`; evicted states are not reconstructed.

BART normally computes a new token's learned absolute position from cached sequence length. That would be wrong after eviction. A scoped, per-model embedding wrapper supplies the actual monotonically increasing decoder input position, then restores the original method on exit. This does not patch library files or another process.

50% retention **within each compartment is not an equal-total-memory-budget comparison**: the source cache is usually much larger than the short generated-history cache. Report component sizes and total physical KV bytes, not merely the nominal ratio.

## Required checks and evidence

- Tiny deterministic tests: full-cache logits match full-prefix computation; cross-cache K/V physically shrinks and changes fixed-prefix logits; self-cache compaction retains correct absolute positions; both-half achieves half of both compartment sizes at applicable steps; hooks restore even on exceptions.
- Real-checkpoint tests: full-cache adapter must match the already-native-validated sampler for delta=0 and 2. At a fixed teacher-forced prefix, every intervention must change logits while recording the actual cache layout.
- Four-condition end-to-end smoke with NLI and analysis. Formal runs start only when these pass. Smoke records never enter the formal results.
- Every generation batch stores original/source-retained positions, layer projection counts, absolute decoder positions, self slots before append and after eviction, actual backing-tensor sizes, and cache history. Per-sequence final cache snapshots handle unequal EOS times.
- Evaluation uses the **full model-visible source**, not the retained cache subset. NLI remains an automatic support proxy; existing human audits remain pending. Both raw-token and re-tokenized text watermark signals are retained.

## Operations

```bash
cd ~/watermark_study/rebuild
screen -dmS wm-kv-cache bash -c 'export CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; exec .venv/bin/python -u kvstudy/runner.py'
tail -F logs/kv-cache.log
```

Outputs: `runs/kv-cache/<condition>/{smoke,formal}/`; `runs/kv-cache/status.json`; engineering checks in `real_checkpoint_checks.json`; aggregate paired contrasts in `cache_comparison_formal.json`. Completed batches resume after an explicit restart only if the source/configuration identity remains unchanged. Failures stop this worker without touching other jobs.
