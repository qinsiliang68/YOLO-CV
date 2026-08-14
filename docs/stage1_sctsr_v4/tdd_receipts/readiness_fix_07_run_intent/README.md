# Readiness fix 07: mandatory run-intent acknowledgement

## Finding

The implementation had detailed scientific and runtime contracts, but a formal
runner could consume an owner-signed execution token without proving that the
operator or training-machine AI had read the exact runbook and bound its
understanding to the exact role, arm, seed, dataset, output, parent, schedule,
identity pool, release, token and resume state.

## Failure-first evidence

- `RED_BEHAVIOR.junit.xml`: 1,087 bytes; SHA-256
  `D97346B38CAC868E3984E679019B7034F037FF20553E30944AF5E11DF0E0FC3A`;
  collection failed because the validator module did not exist.
- `RED_INTEGRATION.junit.xml`: 1,219 bytes; SHA-256
  `A580758C10185D500A0115864ED196DACA42C36D9BB639A0B52A236ABB91C46E`;
  snapshot/binding APIs did not exist.
- `RED_FORMAL_ENFORCEMENT.junit.xml`: 2,871 bytes; SHA-256
  `F7648BF43CCA684D2FB1A926E85DBABE16CD983183C96734A275B3B72B3A56F5`;
  a formal runner reached dataset iteration without run-intent evidence.
- `RED_CONTROL_PLANE_CLI.junit.xml`: 1,736 bytes; SHA-256
  `AF747975BDB1B1ACDE99558D2FF02DDBFB3656FD05BF425A1A27341BBEC532E2`;
  the operator-facing build CLIs did not exist.

## Fix

- A strict acknowledgement binds 23 exact job-context fields and sixteen
  explicit understanding statements.
- The formal parent and branch CLIs require both the acknowledgement and the
  SHA-bound runbook manifest.
- Validation is performed before the one-use token claim and before trainer
  construction.
- Every START/RESUME acknowledgement is atomically snapshotted into an
  append-only attempt chain; terminal receipts, formal manifest and closeout
  bind the latest snapshot.
- Runbook or acknowledgement drift, stale acknowledgement, job substitution,
  missing statements, copied claim registry, placeholders and altered snapshot
  bytes all fail closed.
- Builder/validator CLIs do not claim a token and explicitly report
  `formal_training_started=false`.

## Green evidence

- `GREEN_BEHAVIOR.junit.xml`: 2,840 bytes; SHA-256
  `00F88CF178F58DD6D5B3A1242796878B82CDF9E4D8359B38C04BA80155D66EE9`;
  16 initial validator/schema tests passed.
- `GREEN_INTEGRATION.junit.xml`: 9,840 bytes; SHA-256
  `D236DBAD3BBA6290032CF8F9FCB30F4E0944E52DCE65A5AA2800999AB2F08841`;
  59 validator, CLI, execution-claim, resume, closeout-neighbor and schema tests
  passed.

No formal training, seed, assignment, engineering gate, pilot release,
blind/test access or method-effectiveness claim was generated.
