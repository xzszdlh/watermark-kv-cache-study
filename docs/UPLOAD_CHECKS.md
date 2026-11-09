# Repository preparation checks

Prepared on 2026-11-09 from the local rebuilt experiment project.

## Checks performed

- Parsed every included Python source with `ast.parse`.
- Parsed every included JSON protocol/configuration example.
- `ai_audit/test_judge.py`: 4 tests passed using mocked HTTP responses.
- `ai_audit/test_panel_runner.py`: 3 tests passed, 2 skipped (Linux-only cross-process locking; frozen audit items intentionally excluded).
- No real API evaluations were requested.
- Original scientific implementation files were copied without algorithm changes. Packaging edits affect two audit tests: the internal endpoint assertion now uses `example.invalid`, and the frozen-item parity test skips when private inputs are absent.

## Model tests on the upload workstation

The workstation has Python 3.9.13, PyTorch 1.13.1, Transformers 4.24.0, and Tokenizers 0.11.4. These do not match the recorded experiment environment (Python 3.10, PyTorch 2.5.1, Transformers 4.44.2, Tokenizers 0.19.1). Its unrelated TensorFlow installation also fails to load a DLL.

With `USE_TF=0`, `USE_FLAX=0`, and Hugging Face offline mode:

- `test_correctness`: 7 passed, 1 skipped, 3 dependency errors. The errors were missing `rouge_score` and the missing `GenerationConfig` API in the older Transformers installation.
- `kvstudy/test_cache.py`: 4 dependency errors because PyTorch 1.13.1 lacks `Tensor.untyped_storage`, used to verify physical cache compaction.

The global Python installation was not modified. The full suite must be rerun in the documented environment before claiming fresh validation. This upload does not rerun the GPU experiments, establish statistical results, or validate service model identities.

## Files deliberately excluded

News article samples, generated summaries, audit items/responses, human annotations, logs, model weights, local populated API configs, original user authorization records, capability-response records, credentials, CVs, transcripts, emails, and PDF application materials.

API configuration examples use nonfunctional placeholder endpoints. Real judging requires independently prepared local items, configuration and permission; the copied historical runners keep their original preflight gates.
