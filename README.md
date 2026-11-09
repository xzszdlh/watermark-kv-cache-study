# Watermark and KV-cache study

Research code for sentence-level watermark diagnostics and source/history KV-cache ablations in BART summarization.

本仓库整理自董悦老师 Research Skill Test 的重建版实验代码。包含水印生成、句子对齐、NLI 评估、KV 缓存干预、统计分析及辅助审计程序；不包含简历、成绩单、邮件、服务凭据、新闻原文或实验原始输出。

## Experiments

- **Local watermark diagnostics:** compare sampled-token and rendered-text watermark signals with windowed source-support scores; adjust sentence-level associations for entropy, length, position, overlap, and repetition.
- **KV-cache interventions:** physically prune source cross-attention and generated-history self-attention caches separately; preserve absolute decoder positions and verify cache storage.
- **Sensitivity analyses:** decoding regime, generation limit, retention ratio, and random source-position retention.
- **Evaluation:** document-cluster paired analyses, blinded human-audit tooling, and optional API-judge tooling. API judgments are not human annotations.

The implementation uses `facebook/bart-large-cnn`, CNN/DailyMail v3.0.0, and `cross-encoder/nli-deberta-v3-base`, with checkpoint/dataset revisions in the protocol files. This is an encoder-decoder cache-ablation study, not an implementation or reproduction of HalluKV.

## Repository layout

```text
rebuild/
  protocol.json             Base model/data/measurement protocol
  prepare.py                Pinned asset download and integrity checks
  study.py                  Generation, evaluation, analysis entry point
  watermark.py              Green-list watermark and detector
  generation.py             Cached sampling and runtime measurements
  alignment.py              Generated-token to sentence alignment
  evaluation.py             Windowed NLI and overlap metrics
  analysis.py               Detection and association statistics
  protocols/                Decoding/length sensitivity protocols
  kvstudy/                  Source/self-attention cache interventions
  execution/                KV worker and GPU scheduler dependencies
  supplements/              Cache-sensitivity suite
  postprocess/              Paired analysis and human-audit tools
  ai_audit/                 Optional API-judge code and safe config examples
  reference/                Upstream watermark implementation reference
```

## Environment

The original GPU environment used Python 3.10, PyTorch 2.5.1 with CUDA 12.4, and Transformers 4.44.2. Generation and process-locking entry points target Linux with NVIDIA CUDA; Windows can be used for selected offline tests.

Install a CUDA-compatible PyTorch build for your machine first, in an isolated environment, then install the remaining dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
# Install the appropriate PyTorch 2.5.1 build separately.
python -m pip install -r requirements.txt
cd rebuild
mkdir -p logs
python prepare.py
```

`prepare.py` downloads model assets. The dataset is loaded by the study runner unless a matching local cache exists. Downloads and model weights are not stored in Git. `rebuild/setup.sh` is the original server-specific setup script; review its Python path and package index before using it elsewhere.

## Run the base experiment

```bash
# From rebuild/, in the configured environment
python -m unittest test_correctness -v
python kvstudy/test_cache.py
CUDA_VISIBLE_DEVICES=0 python -u study.py --mode smoke
# Inspect smoke outputs before a full run
CUDA_VISIBLE_DEVICES=0 python -u study.py --mode formal
```

The base protocol selects 200 validation documents for calibration and 500 disjoint test documents. Four watermark strengths and three seeds give 8,400 generated summaries **including calibration and test**, not 8,400 independent test documents. Additional protocols reuse the original sample selection.

```bash
# Example decoding supplement, after the base formal run
# Review the frozen physical_gpu field before choosing the device.
CUDA_VISIBLE_DEVICES=0 python -u study.py --mode smoke --protocol protocols/nucleus-128.json --run-name nucleus-128
CUDA_VISIBLE_DEVICES=0 python -u study.py --mode formal --protocol protocols/nucleus-128.json --run-name nucleus-128
```

For KV ablations, see `rebuild/kvstudy/README.md`. For retention/decoding supplements, see `rebuild/supplements/README.md`. These stages depend on existing base-run manifests and samples. GPU numbers, admission checks, memory caps, and local interpreter paths in the orchestration scripts reflect the original server; review them before launching. Changing a frozen configuration requires a new run identity rather than resuming into old outputs.

## Offline checks

```bash
cd rebuild
python -m unittest test_correctness -v
python kvstudy/test_cache.py

# Optional API-judge tests use mocks, not real API calls.
# Copy the sanitized examples locally first; keep the .invalid endpoints.
cp ai_audit/protocol.example.json ai_audit/protocol.json
cp ai_audit/scheduling_amendment.example.json ai_audit/scheduling_amendment.json
cp ai_audit/terra_extension.example.json ai_audit/terra_extension.json
python ai_audit/test_judge.py
python ai_audit/test_panel_runner.py
```

Tests requiring real checkpoints, CUDA, or withheld audit items may be skipped when those assets are absent. `docs/UPLOAD_CHECKS.md` records what was actually checked before upload. No GPU experiment or paid API evaluation is launched by repository preparation.

## Optional API audit

`rebuild/ai_audit/*.example.json` are sanitized configuration templates. Service endpoints use `example.invalid`, dataset identities are placeholders, and no original user authorization or live capability evidence is included. The historical runners retain their preflight and frozen-manifest checks: they are not ready-to-run provider clients until you prepare your own items, configuration, capability checks, and authorization. Credentials must come from process environment variables. Never commit populated configuration files, response logs, or secrets.

Packaging adjustments in the audit tests use the placeholder endpoint and skip the frozen-item parity test if withheld items are absent. The scientific experiment implementation itself is unchanged.

## Data and provenance

- Obtain source datasets and checkpoints through their original distributions and comply with their terms.
- Generated outputs, source articles, sampled audit items, and judgment logs are excluded. Reproduce them locally with the recorded protocols.
- The parent directory's historical prototype and measurements are intentionally not included.
- Existing inner READMEs describe the protocol at different development stages. In particular, the base README's original no-cache statement refers to the initial diagnostic experiment; cache interventions are in `kvstudy/` and `supplements/`. Its old three-page report-build note is historical; report/CV build assets are outside this code repository.
- Scientific code files are copied from the local rebuilt project. Sanitized examples and this packaging do not replace the original frozen experiment archive.

## Third-party code

`rebuild/reference/watermark_processor.py` retains its upstream Apache-2.0 notice. The exact upstream commit and attribution are in `rebuild/reference/UPSTREAM.txt`; see `THIRD_PARTY_NOTICES.md`. No new license is assigned to the project's original code by this upload.
