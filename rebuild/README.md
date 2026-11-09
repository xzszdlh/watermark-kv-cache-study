# Watermark/source-support study: rebuilt protocol

This directory replaces the invalid experimental implementation for a new run. The parent directory's old measurements are retained as historical inputs and are not mixed into these results.

## Question and scope

Can local Kirchenbauer watermark strength predict source support after accounting for generation entropy, span length, position, repetition and lexical source overlap? The direction of the relationship is **not** assumed. This is an observational diagnostic study, not a new guaranteed hallucination detector. No KV-cache compression, HalluKV replication, or entropy-gated watermark is implemented or claimed.

## Fixed protocol

`protocol.json` fixes checkpoint revisions, dataset selection, hyperparameters and analysis choices before inspecting new outcomes.

- `facebook/bart-large-cnn` is a summarization-finetuned seq2seq checkpoint.
- Complete CNN/DailyMail v3 validation/test splits are already cached. Draw 200 validation documents for detection-threshold calibration and 500 independent test documents; record IDs and source hashes. Downloading a complete split is distinct from evaluating all its examples.
- Gamma .25, deltas 0/1/2/4, generation seeds 42/43/44. This is 8,400 summaries (700 documents x 4 strengths x 3 seeds), not 8,400 independent documents.
- Same model/input truncation, temperature 1, unrestricted multinomial sampling (top-k 0, top-p 1), decoder-start and forced BOS, minimum 24 lexical tokens, maximum 128 decoder steps. Special tokens except EOS are masked from sampling; EOS is suppressed until the lexical minimum. Forced BOS and special tokens are excluded from detection. No forced EOS is inserted at the length cap; record truncation.
- Batched cached decoding uses a separate reproducible RNG per document/seed, reset for each delta. Microbatch size is recorded; hardware floating-point effects can still prevent exact equality across different batch sizes. Cache mechanics are tested against full-prefix decoding.
- Green-list PRF: simple_1, seed=15485863*previous_token, CPU torch.randperm, matching the upstream **CPU** implementation. GPU-only upstream random permutations differ; CPU is used consistently at embedding and detection. Source: `reference/watermark_processor.py`, Apache-2.0 header, upstream commit recorded in `reference/UPSTREAM.txt`.
- Exclude tokens after EOS/PAD and other specials. Exclude the first lexical token because a portable text-only detector has no preceding lexical token. Primary z counts unique bigrams; retain all-bigram scores as diagnostics. Normal p-values are approximations, not calibrated empirical false-positive guarantees. Include a wrong-key detector diagnostic.

## Measurements

1. ROUGE-1/2/L versus the human reference; paired deltas versus unwatermarked outputs.
2. Watermark detection: use **displayed text re-tokenized by BART** for deployable detection, with original sampled-token scores retained separately as internal diagnostics. Record the fraction of noncanonical BPE sequences. Threshold chosen only from the unwatermarked validation split, targeting empirical FPR <=1%; report held-out FPR, TPR and AUROC. Report z>4 separately. 200 calibration documents provide limited tail resolution; no 0.1% FPR claim.
3. Sentence-level source support: `cross-encoder/nli-deberta-v3-base`, explicitly verified label mapping. Cover **all model-visible source text** using 256-NLI-token windows at stride 192. Evaluate each summary sentence against every window, retaining maximum entailment plus evidence and all window probabilities. Pair lengths are checked without truncation; overlong/empty hypotheses are marked missing, not silently replaced by a different metric.
4. NLI is a proxy with known error modes: maximum-over-windows selection, cross-sentence reasoning, omissions, and hallucination confounds. A simple true/contradictory pair is an engineering direction check, not factuality validation. A fixed random blinded 100-sentence annotation sheet is produced for human checking. Until annotated, conclusions remain about the **automatic support proxy**.
5. Original generated tokens are mapped through their byte-level BPE bytes to decoded character offsets, without assuming re-tokenization recovers sampled token IDs. Reconstructed text must exactly match the tokenizer's displayed text. Explicitly count invalid generated UTF-8 sequences and test multibyte characters across token boundaries. Assign each token to one sentence by maximum character overlap; ties go to the earlier sentence, whitespace gaps to the following sentence. Keep internal-token and text-only sentence watermark scores separate. Minimum 10 unique bigrams per eligible sentence. The initial smoke test exposed the incorrect canonical-retokenization assumption; its records and code are retained under `audit/smoke_alignment_failure/`. Formal experiment has not used those records.
6. Primary association: **internal sampled-token signal** on test sentences at delta=2, seed=42; text-only signal is a separately labeled sensitivity analysis. Other strengths/seeds are sensitivity analyses, not opportunities to pick the best result. Raw Pearson/Spearman and adjusted partial correlation; adjust entropy, log(unique bigrams), relative position, source trigram overlap, repeated-bigram fraction. Resample whole documents for bootstrap CIs and refit adjustment each replicate. Results are associations, not causal mechanisms.

## Operations

- Use the task-local `.venv` over the existing CUDA-compatible torch 2.5.1 environment. Shared conda packages are not upgraded or uninstalled.
- `prepare.py`: pinned mirror downloads, verified LFS SHA-256, HTTP Range resume and exponential backoff, byte/percentage/speed logs. TLS verification stays on.
- `launch.py`: exclusive job lock; run unit tests, wait for verified downloads, run a small smoke test (including real-checkpoint comparison with HF `generate` and cached-vs-full-prefix logits), check checkpoint resume, then run the fixed formal experiment. Stop on code/metric errors rather than endlessly rerunning from sample zero.
- Every generation and evaluation batch is saved atomically and bound to a source/config/environment hash. Changed code cannot silently resume into existing measurements.
- GPU physical index 2 only; a 70% per-process GPU-memory cap leaves room for existing workloads. Actual memory and batch timing are logged. There is no promised runtime before measurement.

```bash
# Start downloads if not already active; resume existing .part files.
.venv/bin/python -u prepare.py >> logs/download.log 2>&1
# Start the single supervisor detached. Do not run two copies.
screen -dmS wm-study .venv/bin/python -u launch.py
# Logs and current machine-readable status:
tail -F logs/download.log logs/run.log
# runs/formal/status.json is updated after every checkpoint.
```

Outputs: `runs/formal/manifest.json`, `samples.json`, `generation/*.json`, `evaluation/*.json`, `analysis.json`, `human_audit_blinded.csv`. Smoke measurements live separately and never enter formal tables. No report is drafted from unvalidated or incomplete results.

## Supplementary decoding/length experiments

These settings were fixed **after inspecting the initial unrestricted-sampling run** and before running the supplements. They are sensitivity analyses on the same test documents, not a new independent confirmation and not a search for positive results. The original run and its executed source snapshot remain unchanged in `runs/formal/` and `runs/formal/source/`.

| Run | GPU | Sampling | Maximum decoder steps | PyTorch allocation cap |
|---|---|---|---|---|
| `nucleus-128` | 0 | top-p 0.9, top-k 0, temperature 1 | 128 | 25% of GPU memory |
| `topk-128` | 2 | top-k 50, top-p 1, temperature 1 | 128 | 25% |
| `nucleus-256` | 3 | top-p 0.9, top-k 0, temperature 1 | 256 | 20% |

Each evaluates all four deltas (0,1,2,4) and all three seeds on the exact original 200 calibration + 500 test documents, giving 8,400 outputs per setting and 25,200 supplementary outputs total. Batch size 64 in all three settings; actual microbatch size is retained if memory pressure requires splitting. Compare `nucleus-256` with `nucleus-128` for the length ablation, not directly with unrestricted sampling. NLI pair/chunk limits and missing-value handling remain unchanged; any unscored long sentences must be disclosed.

Watermark bias is applied **before** top-k/top-p filtering, using the same order as the pinned Transformers generation implementation. The logged raw/base green mass precedes candidate filtering and is not the final filtered green probability. Native-decoder parity checks cover both delta=0 and delta=2 in each new sampling path. The human audit remains pending; GPU runs do not substitute for human labels.

`parallel_runner.py` runs one worker per GPU, with a separate smoke gate before each formal run, exclusive scheduler lock, unbuffered per-job logs, and conservative admission checks. It requires the task's allocator cap plus a further 4 GiB to be free before admitting a job; it never kills other workloads. Computational failures stop the affected job, while independent jobs continue. Restarting resumes committed batches only when the source/config identity matches.

```bash
screen -dmS wm-sensitivity .venv/bin/python -u parallel_runner.py
tail -F logs/parallel.log
# Detailed worker logs:
tail -F logs/nucleus-128.log logs/topk-128.log logs/nucleus-256.log
```

Per-job output: `runs/<run-name>/formal/`; scheduler status: `runs/parallel_status.json`. A comparison file containing all settings (no best-setting selection) is written to `runs/sensitivity_comparison.json` only if all supplementary runs finish. Detection-rate Wilson intervals are also reported for these runs to avoid treating bootstrap `[1,1]` at an observed 100% detection rate as a population guarantee.


## Report build location

The 3-page Research Skill Test report is no longer built inside the experiment tree. Layout sources, figure scripts and page previews now live in the workspace at `.cv_build/yue_dong_skill_test/`, alongside the other ReportLab CV builders, so measurement files and submission files stay separated:

```bash
# Regenerate figures, then both report languages.
python -X utf8 .cv_build/yue_dong_skill_test/source/plot_figures.py
python -X utf8 .cv_build/yue_dong_skill_test/source/build_skill_test_report.py
```

Inputs are read from this directory's `runs/` outputs; the English submission PDF and the Chinese reading PDF are written to `09_Yue_Dong_UCR/`. Older report copies under `.cv_build/yue_dong_skill_test/build/output_before_organization/` are superseded by that step and are kept only as layout history.
