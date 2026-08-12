# Dynamic replay budget-efficiency preregistration

## Frozen question

Can the same OOF-ranked normal pool produce a more reliable tail benefit when replay is
conditioned on training stage and cumulative exposure, without changing the canonical
240-run learner?

The estimand is conditional: `V(selection | theta_t, schedule, exposure, seed, context)`.
No universal static `V(x)` and no weighted primary score are registered.

## Canonical training lock

Every formal arm uses the exact 240-run canonical configuration. The immutable lock file
SHA256 is `7AFD96784F4994903892BD5BA8477396AAF5EFFBFF158A1C57225428BF01F74E`. Replay ratio, replay schedule, selection composition, training seed,
device/data/output paths, and valid resume identity are the only registered variations.
OOM is a failed attempt; it never authorizes silent batch, image-size, optimizer, precision,
augmentation, worker, or schedule changes.

## Seven-day staged releases

1. Cycle 1: high-pressure continuous replay, same-peak taper, and no replay.
2. Cycle 2: dose-matched timing separation plus 0.5% and 1.0% transfer.
3. Cycle 3: weak-defect guard templates; parameters remain unbound until Cycle 2 freezes.
4. Cycle 4: six-arm unseen-seed confirmation; parameters remain unbound until Cycle 3 freezes.

Cycle 1/2 contain 80 frozen logical runs over eight paired discovery seeds. Cycle 1 jobs
begin at `ENGINEERING_GATE`; Cycle 2 remains `HELD`. Cycle 3/4 have no executable jobs yet.
All arms use restart boundaries 140, 150, 160, and 200 so checkpoint/resume mechanics are
not confounded with policy. Continuous and same-peak taper share an exact epoch-140 prefix.

## Interpretation

Primary reporting is safety first: the raw FN=0..95 frontier, `TN_at_FN95`, and
`FN_at_TN68253`, paired by training seed. Process trajectories and gradients are mechanism
evidence, not substitutes for endpoint replication. Blind holdout access is forbidden until
the final policy and analysis are frozen.
