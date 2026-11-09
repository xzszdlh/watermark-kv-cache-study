# Parallel placement of the already-frozen KV experiment

This is an execution-layer change only. It does not add a model, scientific condition, dataset, seed, decoding parameter, or cache policy.

- Split the existing four cache conditions x three seeds into 12 disjoint work units.
- Preserve original canonical manifests and completed files byte-for-byte. A one-time hash index is retained in `runs/kv-cache/execution/pre_takeover_checkpoints.json`.
- Original manifests describe the initial GPU-2 placement. Actual new GPU UUID, device index, process ID, memory cap, selected seed, and scheduling-source hashes are recorded separately in `runs/kv-cache/shards/<condition>/seed<seed>/execution.json`. Do not infer every record was executed on GPU 2 from the old manifest.
- Mathematical generation and evaluation functions are imported unchanged and verified against their frozen source hashes. A reference batch was regenerated on GPU 1 with exactly matching token IDs; a reference NLI batch matched within the recorded numeric tolerance. These are execution checks, not research measurements.
- Each shard writes only its seed-qualified filenames in an exclusive, locked view directory. Existing canonical files are hard-linked into the view; completed results are hard-linked back, never overwritten. Hard links share bytes, so this is not another copied dataset or duplicate set of measurements.
- A pre-existing serial KV worker may be stopped only after its exact PID, command and working directory are verified. Other services and the independent sensitivity jobs are untouched.
- The fleet takes the old serial-worker lock so the serial runner cannot restart into the same outputs concurrently.
- Per-worker PyTorch allocator cap: 16% of one A100-40GB; additionally reserve 1 GiB for context/overhead per worker and 6 GiB free per GPU at admission. Reserve the full allocation before CUDA initialization appears in nvidia-smi. Up to three total project tasks on GPUs 0/1/2, one on GPU 3 while accounting for the ongoing sensitivity jobs and memory used by unrelated services.
- Per-document RNGs, batch size 32, source order and cache rules remain unchanged. GPU assignment may change; therefore all runtimes remain shared-resource operational timings, not a controlled speed benchmark.
- On a worker error, preserve completed files and report the failed shard. Do not blindly rerun all conditions. Restarting the fleet resumes completed atomic checkpoints.
- Final statistics run only after all 528 generation and 528 evaluation batches exist. The original condition-level statistical functions and cross-condition comparison are reused. Human validation is still separate and pending.

## Logs and status

```bash
tail -F ~/watermark_study/rebuild/logs/kv-fleet.log
# Individual example:
tail -F ~/watermark_study/rebuild/logs/kv-cross-half-seed43.log
```

Aggregate status: `runs/kv-cache/execution/fleet_status.json`.
Per-shard status: `runs/kv-cache/shards/<condition>/seed<seed>/status.json`.
Final analysis log: `logs/kv-finalize.log`.

The previous single-worker log `logs/kv-cache.log` is historical after takeover; use the fleet log for new progress. No extra experiments are invented merely to fill VRAM, and no application-quality or positive-result claim follows from GPU utilization.
