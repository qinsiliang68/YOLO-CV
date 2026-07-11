# Failure recovery

The code deliberately exposes simple recovery states for the existing AIOps layer; it does not implement a second scheduler or monitoring platform.

- Exit `0`: complete, already complete, or no task.
- Exit `20`: retryable worker, lock, timeout, training, or prediction failure.
- Exit `30`: frozen input, release, checksum, machine snapshot, or scientific-contract error. Do not retry until corrected.
- A stale `RUNNING` attempt resumes only from a checkpoint that can be reloaded by the local YOLO runtime and contains native resume state.
- Resume is recorded as `native_approximate`, including resume count, checkpoint hash, epoch range, and segment identity.
- `TRAIN_COMPLETED` with failed evaluation reruns evaluation only. `EVALUATED` with failed validation reruns validation only.
- Corrupt or missing `last.pt` remains as evidence; a new attempt is created from the fixed initial checkpoint.
- A replacement attempt supersedes the prior VALIDATED attempt only after the replacement itself passes postflight.
- Controlled failure terminates the complete Python/DataLoader subprocess tree. Parent-process crash detection, alerting, retry limits, and reserve takeover remain AIOps responsibilities.
- Disk pressure may be handled by deleting reproducible staging/cache and restarting. Never delete the frozen selections, contracts, machine asset report, validated predictions, metrics, or artifact manifests.

See `HANDOFF_RATIONALE_AND_STATUS_v1_2.md` for the evidence state and release gate.
