# KV supplementary experiments

The supplementary settings were fixed after observing the original KV results and before generating any new supplementary outputs. They reuse the same BART checkpoint and the original 200 calibration plus 500 test documents. These are sensitivity analyses, not independent confirmation.

## Fixed run matrix

| Group | New conditions | Question |
|---|---:|---|
| Retention | cross/self/both at 75% and 25%: 6 | Is the finding specific to 50% retention? |
| Longer output | full/cross50/self50/both50 at 256 steps: 4 | Is the cache contrast sensitive to the output ceiling? |
| Second sampling rule | full/cross50/self50/both50 with top-k=50: 4 | Is the contrast specific to nucleus sampling? |
| Second source-selection policy | random cross50 and both50: 2 | Is the contrast specific to evenly spaced source positions? |

Each condition uses delta 0/2 and seeds 42/43/44 with batch size 32. Total: 16 conditions, 48 seed shards, 67,200 generated summaries, 2,112 generation batches and 2,112 evaluation batches. The original four conditions are reused as the matched reference where their configuration is identical. New full-cache references for top-k and the longer limit use batch size 32 to match their cache interventions; the earlier batch-64 sensitivity runs are not substituted as exact paired references.

`protocol.json` is the fixed design. `runs/kv-supplement/frozen_plan.json` records its hash and freeze time. Every condition has its source/configuration manifest and source snapshot. `cache_adapter.py` changes source selection only for the random-policy conditions. Original `kvstudy/`, generation, watermark and evaluation code remains unchanged.

Random selection retains source endpoints and draws the other positions using a fixed source-specific CPU RNG. The chosen positions stay the same across watermarks and generation seeds. This tests one alternative deterministic policy realization, not uncertainty over many policy seeds.

## Execution

Twelve engineering checks passed before admission, including exact tokens for an unchanged full-cache reference batch, new ratio/policy storage sizes, end-to-end evaluation on new paths, and absolute decoder positions beyond step 128. Engineering samples never enter result tables.

The fleet uses an exclusive lock, separate condition/seed views, atomic checkpoints and hard-link publication. Each process has a 16% PyTorch allocator cap; admission reserves another 1 GiB per worker and 6 GiB free per GPU. Existing non-study services are left alone. At launch, nine workers fit across four GPUs.

```bash
cd ~/watermark_study/rebuild
tail -F logs/kv-supplement.log
# Condition example:
tail -F logs/supplement-retention-cross75-s42.log
```

Aggregate status: `runs/kv-supplement/status.json`.
Per-shard status: `runs/kv-supplement/shards/<condition>/seed<seed>/status.json`.

On a failed worker, completed files remain. No unconditional retry or deletion of committed results occurs. Resume only with matching source/configuration identities.

## Analysis

The original 2x2 data are already reanalyzed in:

- `runs/kv-cache/paired_analysis.json`
- `runs/kv-cache/paired_analysis.md`

This analysis first averages the three seed-level paired differences within each document, then resamples the 500 documents 20,000 times. It includes cross-vs-self and the interaction `both - cross - self + full`, with pointwise intervals and 16-comparison Bonferroni intervals. An interval crossing zero does not prove equivalence or exact additivity.

After the supplementary suite finishes, a separate waiting process runs `postprocess/supplement_analysis.py`. Its plan is frozen in `postprocess/supplement_analysis_plan.json`; it provides 50,000-replicate document-bootstrap intervals and a declared 80-comparison family across all five supplementary groups. It also reports random-vs-evenly-spaced contrasts and paired changes in cache effects between decoding settings.

```bash
tail -F logs/supplement-analysis.log
```

Final files will be `runs/kv-supplement/supplement_analysis.json` and `.md`. They are not present until the full analysis succeeds. `runs/kv-supplement/postprocess/status.json` distinguishes waiting, running, complete and failed.

All tables must retain measured FPR, NLI missing counts, alignment errors, length-cap rates, and actual cache-byte accounting. The byte fraction is relative to full KV at the same decoder step; it is not whole-GPU memory savings or a controlled speed measurement. NLI remains an automatic source-support proxy.

## Human audit

The local `human_audit/开始人工核验.html` and `human_audit/人工核验说明.md` provide the human task. There are 260 blank items, with original text, sampled sentence and optional summary context. Conditions and model scores are not embedded in the public page. The private selection key remains under `runs/human-audit-private/` and must not be shown to annotators.

The user or another human must choose the labels and evidence. Browser navigation, storage, import and export were tested in an isolated Chrome profile; synthetic test labels were removed and never exported as human results.

Validate a returned file locally without the private key:

```bash
python postprocess/validate_human_audit.py path/to/human-export.json
```

After all 260 records are complete, analyze on the server where the private key resides:

```bash
.venv/bin/python postprocess/analyze_human_audit.py --export path/to/human-export.json
```

The analysis refuses to produce scientific results from an incomplete export. Its mapping, diagnostic threshold and document-level uncertainty plan were fixed before any human labels. The initial audit and paired-cache audit remain separate sampling strata.
