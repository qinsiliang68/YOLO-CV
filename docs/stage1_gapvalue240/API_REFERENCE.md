# Run API

Every `run_NNN.py` exports:

```python
prepare(machine_config, attempt_id=None, allow_new_attempt_after_validated=False)
train(machine_config, attempt_id)
evaluate(machine_config, attempt_id)
validate(machine_config, attempt_id)
run(machine_config, attempt_id=None, allow_new_attempt_after_validated=False)
```

The module constant `RUN_SLOT` is the only run-specific code value. All scientific parameters are looked up from the signed frozen matrix. This permits single-run deployment, independent recovery, and future maintenance without a monolithic 240-run function.

## Isolated prediction worker

`scripts/stage1_gapvalue240/predict_split_worker.py` runs exactly one `val_cal` or
`val_op` prediction in a disposable process. It requires explicit checkpoint,
manifest, dataset-root, local-YOLO-root, GPU, batch, worker and image-size
arguments. The prediction CSV and worker result JSON are atomic and
non-overwriting. The JSON records exact input hashes, PID, timing, row counts,
output hash, status and exit code.

`stage1_gapvalue240.prediction_controller.run_prediction_workers(...)` requires
the order `val_cal`, then `val_op`, launches one worker at a time, validates each
worker result and output checksum, and writes a controller result JSON. Process
exit releases the worker CUDA context before the next training run.

`stage1_gapvalue240.subprocesses.run_logged(...)` writes
`<log>.result.json` for pass, nonzero exit, startup failure, interruption and
timeout. Timeout or interruption terminates the complete subprocess tree,
including PyTorch DataLoader descendants, on Windows and Linux.
