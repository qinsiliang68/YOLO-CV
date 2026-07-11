# Failure recovery

- Input, checksum, schema, candidate-pool, matching, or step-count failures stop before GPU use.
- Failed attempts remain immutable audit records.
- A permanently interrupted triad is rerun in full on a reserve machine.
- Any validated arm from the abandoned triad is marked `SUPERSEDED` and excluded.
- The package does not claim exact-resume equivalence unless the existing trainer exposes and proves restoration of model, optimizer, scheduler, scaler, RNG, sampler, and augmentation state.
- Evaluation can be rerun from saved predictions without retraining.
