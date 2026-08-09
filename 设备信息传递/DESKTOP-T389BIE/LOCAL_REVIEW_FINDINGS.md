# Local Code Review Findings — DESKTOP-T389BIE

## Review identity

- Review date: 2026-08-09 (Asia/Shanghai)
- Reviewing device: DESKTOP-T389BIE
- Repository: qinsiliang68/YOLO-CV
- Branch reviewed: push-info-sampling-lite
- Commit reviewed: 3a08d22fb683be409a308e20503822e9d8d38a82
- Review type: local read-only source, frozen-asset, documentation, and test audit
- Remote fleet access during this review: none
- Training or deployment started during this review: none

This report records findings from the local review performed before any training
machine deployment. It is intended to transfer the review state to another
device or operator. The findings below apply to the exact commit above and must
be rechecked after any corrective commit.

## Executive verdict

Formal deployment is blocked at the reviewed commit.

The campaign is correctly defined as a YOLO11l image-classification campaign,
not a segmentation campaign. The frozen scientific matrix and physical job
graph are internally consistent. However, the reviewed runtime has several
control-plane, data-identity, recovery, lease, and release-lifecycle defects that
can prevent the canary from running, allow incorrect inputs, or promote an
invalid segment to COMPLETE.

The repository label CODE_READY_FOR_OWNER_CANARY should therefore be treated as
a declared project state, not as a locally verified deployment authorization.
Do not start the formal matrix from this commit without resolving the blocking
findings and rerunning the complete gate.

## Confirmed-correct foundations

The following items were independently checked and were consistent:

1. The task is classification. The dynamic trainer passes task=classify and the
   intended classes are no_target=0 and target_defect=1.
2. The active assets are 03_preregistration_v2 and 04_run_queue_v2.
3. The experiment matrix contains 80 unique logical runs:
   - Cycle 1: 24 logical runs, all at ENGINEERING_GATE.
   - Cycle 2: 56 logical runs, all HELD.
4. The physical graph contains 296 unique jobs:
   - Cycle 1: 88 jobs.
   - Cycle 2: 208 jobs.
5. All 240 dependency edges reference existing jobs and preserve seed identity
   and queue order.
6. JOB_EXECUTION_REGISTRY.csv contains the same 296 job IDs as the physical
   graph, with queue_order 1 through 296 and no duplicate job ID.
7. The queue registry SHA-256 is:
   E1D88B7114469B515D7A13D5979C32A59F8DA219D0E6980DED60BB56BB29CFA9
8. The canonical training lock file SHA-256 is:
   7AFD96784F4994903892BD5BA8477396AAF5EFFBFF158A1C57225428BF01F74E
9. The expected initial checkpoint is 28,553,700 bytes with SHA-256:
   6B56513A5D8BDAE6B8F0A36DACAF01B26D5A522BA1B34197C3BAC9FA6463366C
10. The 17 selection templates, their row counts, the 1,200-row monitor, and
    the active canonical-lock copy were internally consistent.
11. Blind holdout state remains UNBOUND and must stay inaccessible.

## Blocking findings

### F-01 — Critical — Canary, gate, release, and assignment form a circular dependency

The ten-machine real-data canary command builder consumes
assignment/STANDALONE_JOB_COMMANDS.csv. An assignment requires a RELEASED
release. A release requires a complete engineering gate. The engineering gate
requires the ten-machine real-data canary plus standalone-entry and assignment
reassignment evidence.

The resulting dependency is circular:

canary commands <- assignment <- release <- engineering gate <- canary aggregate

Evidence:

- stage1_gapvalue240/campaign_engineering_gate.py:18
- stage1_gapvalue240/campaign_controller.py:185
- stage1_gapvalue240/campaign_assignment.py:278
- stage1_gapvalue240/campaign_canary.py:398
- stage1_gapvalue240/campaign_contract_validation.py:189
- stage1_gapvalue240/campaign_contract_validation.py:349

Impact: the documented formal gate sequence cannot be bootstrapped honestly from
the reviewed repository state.

Required direction: introduce an explicitly bounded pre-release canary
authorization or provisional assignment whose identity is tied to the candidate
source tree, queue, canonical lock, and machine configurations. The formal
release must continue to require the completed gate.

### F-02 — Critical — The documented real-data canary is not an executable one-epoch canary

The command builder copies formal assignment commands without changing their
training duration. It writes required_epochs=1 only into template metadata. The
frozen first formal segments cover epochs 1 through 140, and the formal worker
has no canary epoch override. No production node-report writer for
stage1.ten_machine_real_data_canary_node.v1 was found; only an aggregator exists,
while tests synthesize PASS node reports directly.

The operations document also shows --machine-configs-dir, but the actual CLI
requires --expected-machine-ids and does not accept the documented argument.

Evidence:

- stage1_gapvalue240/campaign_canary.py:398
- stage1_gapvalue240/campaign_canary.py:445
- scripts/stage1_gapvalue240/build_ten_machine_real_data_canary_commands.py:19
- scripts/stage1_gapvalue240/dynamic_campaign_train_worker.py:17
- docs/stage1_gapvalue240/DYNAMIC_REPLAY_CAMPAIGN_OPERATIONS_V2.md:134
- tests/stage1_gapvalue240/test_campaign_canary.py:116

Impact: following the operator documentation cannot produce the promised
ten-machine, one-job, one-epoch evidence.

### F-03 — Critical — The actual CLI machine configuration is not bound to the assignment

The worker loads the machine YAML supplied on the command line. Assignment
validation separately verifies the path and SHA recorded inside the assignment,
then checks only that the requested job is assigned to the same machine_id. It
does not prove that the actual CLI YAML path and SHA are the assignment-bound
configuration.

Evidence:

- stage1_gapvalue240/campaign_worker.py:730
- stage1_gapvalue240/campaign_assignment.py:569
- stage1_gapvalue240/campaign_assignment.py:609

Impact: another YAML with the same machine_id can change dataset, staging,
output, coordination root, GPU, or executable paths while the assignment's
different YAML still validates.

Required fix: compare the resolved CLI configuration path and SHA-256 to the
selected assignment row before any lease, staging, or GPU operation.

### F-04 — Critical — Live source code is not bound to the gate or release

The worker records Git HEAD, branch, and tracked cleanliness, but it does not
compare the live HEAD or a source-tree digest to the engineering-gate or release
identity. Untracked files are ignored. The CLI also exposes an
--allow-dirty-code escape.

Evidence:

- scripts/stage1_gapvalue240/dynamic_campaign_train_worker.py:48
- stage1_gapvalue240/campaign_worker.py:495
- stage1_gapvalue240/campaign_worker.py:809

Impact: a clean but incorrect commit or branch can run as FORMAL, and the
recorded provenance is only after-the-fact evidence.

Required fix: fail closed unless the current source tree matches the immutable
release identity. Do not permit the dirty-code option for a formal release.

### F-05 — Readiness blocker — Committed machine configs cannot start the formal worker

The committed Windows machine YAML files omit coordination_root, but the worker
requires it before claiming a lease. machine_config.schema.json has
additionalProperties=false yet does not define multiple runtime fields that the
committed YAMLs already contain, including staging_root and
machine_asset_report. The Python loader allows those fields but also fails to
require coordination_root during initial config validation.

Evidence:

- configs/stage1_gapvalue240/machines/machine_01.yaml:1
- schemas/stage1_gapvalue240/machine_config.schema.json:5
- stage1_gapvalue240/machine.py:9
- stage1_gapvalue240/machine.py:51
- stage1_gapvalue240/campaign_worker.py:774

Impact: the checked-in configs fail later at lease setup, and the frozen JSON
schema does not describe the effective runtime contract.

### F-06 — Critical — Live image bytes and the actual dataset root are not transactionally verified

The worker validates a previously generated machine-asset report and rehashes
four live manifest files plus the checkpoint. The report validator does not
accept the current MachineConfig, bind its dataset_root to the active YAML, or
restat and rehash all live images at job start.

Evidence:

- stage1_gapvalue240/campaign_worker.py:514
- stage1_gapvalue240/machine_assets.py:312

Impact: a YAML can point at a different same-layout dataset tree, or image bytes
can change after the report was created, while the formal worker still accepts
the cached report.

Required fix: bind machine ID, resolved dataset root, every manifest SHA, and a
current canonical image-content digest into the release and per-job input
receipt. Revalidate at a bounded immutable snapshot boundary.

### F-07 — Critical — Reused hardlink cache does not prove the actual staged dataset

Base-cache reuse checks metadata, snapshot ID, expected-count metadata,
dataset_root text, and the existence of four class directories. It does not
enumerate the actual staged files, reject extra class directories, confirm
hardlink identity, or hash content.

Evidence:

- stage1_gapvalue240/hardlink_staging.py:315
- stage1_gapvalue240/campaign_dynamic_training.py:994
- stage1_gapvalue240/campaign_process_telemetry.py:333

Concrete silent-drift risks:

1. An extra empty class directory can change the Ultralytics classifier output
   dimension while the telemetry class-name check validates only indices 0 and
   1 instead of requiring exactly two classes.
2. Missing, extra, stale-inode, or incorrectly placed validation files are not
   covered by training criterion telemetry.
3. In-place source-image modification changes hardlink-backed bytes; atomic
   source replacement can leave the cache on an old inode. Both can preserve
   manifest metadata.

Required fix: before each formal segment, enforce exactly the two intended class
directories, exact per-split relative paths and counts, hardlink/source identity
where required, and the frozen image-content digest.

### F-08 — High — Key-checkpoint predictions are not bound to input manifests

The worker uses live val_op manifests for checkpoint prediction. Existing
prediction reuse validates checkpoint SHA, output SHA, and row count, but the
split and final sidecars do not bind the defect/normal input manifest SHA-256.

Evidence:

- stage1_gapvalue240/campaign_worker.py:945
- stage1_gapvalue240/campaign_checkpoint_predictions.py:273
- stage1_gapvalue240/campaign_checkpoint_predictions.py:309
- stage1_gapvalue240/campaign_checkpoint_predictions.py:389

Impact: same-row-count val_op replacement can produce results for the wrong
samples, or old predictions can be accepted after input drift.

### F-09 — Critical — A failed segment can be falsely promoted on retry

completed_epochs is written at train-epoch end, before checkpoint, telemetry,
finite-loss, resolved-argument, and final output validation. The runtime calls
recorder.complete before _validate_outputs. If final validation then fails,
recorder.fail marks the segment FAILED but leaves completed_epochs at the
boundary. Retry resolution returns SKIP_COMPLETE solely from that epoch number.

Evidence:

- stage1_gapvalue240/campaign_dynamic_training.py:633
- stage1_gapvalue240/campaign_dynamic_training.py:939
- stage1_gapvalue240/campaign_dynamic_training.py:1141
- stage1_gapvalue240/campaign_worker.py:450
- stage1_gapvalue240/campaign_worker.py:685

Impact: a NaN/Inf loss, canonical-argument drift, missing retained checkpoint,
or other final-validation failure can be marked COMPLETE on the next attempt.

Required fix: define a signed or hashed segment-completion receipt written only
after every validation passes. SKIP_COMPLETE must validate that receipt and all
referenced artifacts, not completed_epochs alone.

### F-10 — High — Role-loss telemetry is outside the worker completion transaction

Per-epoch process telemetry validation checks Parquet plus its JSON sidecar but
does not validate the sidecar-referenced role-loss summary. The standalone
all-epoch validator performs this check, but the worker does not invoke it before
publishing job COMPLETE.

Evidence:

- stage1_gapvalue240/campaign_process_telemetry.py:246
- stage1_gapvalue240/campaign_worker.py:697
- stage1_gapvalue240/campaign_all_epoch_validation.py:182

Impact: a missing or corrupt role-loss summary can coexist with a COMPLETE job.

### F-11 — Critical — Lease expiry depends on unsynchronized wall clocks

The claimant compares local time.time() with heartbeat_at_unix written by
another machine. A clock offset larger than the TTL can reap a live job; the
opposite offset can block recovery. The coordination canary currently declares
that the lease does not depend on clock synchronization without validating that
claim.

Evidence:

- stage1_gapvalue240/campaign_lease.py:264
- stage1_gapvalue240/campaign_lease.py:274
- stage1_gapvalue240/campaign_canary.py:219

Impact: duplicate ownership or unrecoverable stale claims can occur on a
multi-node Windows fleet.

### F-12 — Critical — Lease release is not atomically fenced with result publication

JobLease.release does not assert that the active claim still exists and matches
the holder. It archives a matching claim when present but writes the holder's
terminal heartbeat unconditionally. The worker checks the lease once, then
writes result and job state in separate operations before context exit.

Evidence:

- stage1_gapvalue240/campaign_lease.py:450
- stage1_gapvalue240/campaign_worker.py:982

Impact: after stale reaping or assignment replacement, an old holder can publish
local COMPLETE state or a terminal heartbeat while a replacement claim is live.

Required fix: terminal publication must be one fenced transaction that validates
active assignment SHA and lease token immediately before and during publication.

### F-13 — Critical — Engineering-gate evidence can be rebound to unrelated identities

bind_validation_evidence checks only payload schema and PASS status, then writes
the caller-provided source-tree, queue, and canonical-lock identity into the
envelope. It does not prove that the raw PASS report was generated for those
identities. Envelope validation rechecks the envelope identity and payload hash,
but not an identity inside the underlying payload.

Evidence:

- stage1_gapvalue240/campaign_engineering_gate.py:88
- stage1_gapvalue240/campaign_engineering_gate.py:131

Impact: a stale PASS report can be rebound to new code, queue, or canonical
assets.

Required fix: each lower-level evidence payload must contain and validate the
same immutable identity tuple; binding must reject identity-free or mismatched
payloads.

### F-14 — Blocking integrity defect — Frozen preregistration hashes do not match the clean checkout bytes

PREREGISTRATION_VALIDATION.json reports artifact_hashes_verified=true and PASS,
but all 18 listed artifacts differ in raw SHA-256 and byte size in the reviewed
checkout. Converting the current LF files back to CRLF makes all 18 expected
hashes and sizes match exactly. This is a serialization/newline defect, not
scientific table drift.

The stale CRLF hashes also propagate into RUN_QUEUE_VALIDATION.json
preregistration_inputs.

Evidence:

- artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/03_preregistration_v2/PREREGISTRATION_VALIDATION.json:2
- artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/04_run_queue_v2/RUN_QUEUE_VALIDATION.json:17
- stage1_gapvalue240/campaign_preregistration.py:872
- .gitattributes:1

Impact: the frozen preregistration cannot be reproduced byte-for-byte from a
clean checkout.

Required fix: select one committed newline policy, regenerate the validations
from post-checkout bytes, and add a clean-checkout raw-hash test.

### F-15 — Critical campaign-lifecycle gap — Confirmatory and Cycle-2 release paths are incomplete

The release builder creates a RELEASED two-seed pilot and HOLD manifests for the
remaining Cycle-1 seeds and future cycles. No explicit, validated pilot-closeout
promotion path was found. The release loader also rejects any released job set
whose cycle identity is not exactly CYCLE_1 with ENGINEERING_GATE state.

Evidence:

- stage1_gapvalue240/campaign_controller.py:223
- stage1_gapvalue240/campaign_controller.py:282
- stage1_gapvalue240/campaign_controller.py:397

Impact: the reviewed implementation does not provide a legitimate path from the
small pilot to the remaining six Cycle-1 seeds or to Cycle 2.

### F-16 — High — Bare Python execution conflicts with the frozen environment policy

Standalone assignment commands correctly use uv run python. The optional
controller and checkpoint-prediction subprocesses instead use
machine.data.python_executable. Every committed Windows config currently sets
that value to bare python.

Evidence:

- stage1_gapvalue240/campaign_assignment.py:178
- scripts/stage1_gapvalue240/run_dynamic_campaign_controller.py:298
- stage1_gapvalue240/campaign_checkpoint_predictions.py:229
- configs/stage1_gapvalue240/machines/machine_01.yaml:17

Impact: subprocesses can leave the project uv environment, import a different
package set, or fail on a node where PATH Python is not the trained environment.

## Test evidence

The following commands were run locally without SSH, deployment, or training.
The pytest cache plugin was disabled to avoid adding test cache state.

### Focused v2 runtime suite

Result:

- 143 passed
- 3 failed
- 0 skipped
- Duration: 219.91 seconds

All three failures occurred because the repository-root yolo11l-cls.pt was
absent. The positive-path tests depend on the exact ignored external checkpoint
rather than a hermetic fixture.

### Explicit tracked test directory

Command scope: tests

Result:

- 524 passed
- 4 failed
- 2 skipped
- Duration: 198.04 seconds

The same three checkpoint-dependent failures occurred. The fourth failure was a
legacy HN-band validation test whose skip condition checks only that a dataset
directory exists. The local directory contains manifests but no full image
payload, so all 120 checked runs fail their image prerequisites.

The two skips were environment gates for unavailable canonical 240-run evidence
and the unavailable canonical source ledger.

### No-argument pytest discovery

The literal no-argument command stopped during collection with 14 errors because
the local workstation contains unrelated untracked duplicate repositories and
artifact package builds. The repository has no pytest testpaths/norecursedirs
boundary, so discovery traverses those local trees and triggers import-path
mismatches.

These local asset/discovery failures do not negate the passing unit coverage,
but the upstream statement 463 passed, 1 skipped was not reproduced in this
workspace.

## Scheduling conclusion

The user's fleet contains 13 physical positions, but the reviewed formal
protocol cannot keep 12 GPU machines simultaneously training.

The current assignment rules enforce one physical machine per cycle/seed block.
Cycle 1 has only eight seeds. Each Cycle-1 seed has 11 physical jobs and 460
segment-epochs, so the exact formal concurrency limits are:

| Phase | Independent seed blocks | Maximum formal GPU machines |
| --- | ---: | ---: |
| Default S001/S002 pilot | 2 | 2 |
| Full Cycle 1 | 8 | 8 |
| Full Cycle 2 | 8 | 8 theoretical; currently HELD |

Evidence:

- stage1_gapvalue240/campaign_assignment.py:229
- stage1_gapvalue240/campaign_assignment.py:312
- stage1_gapvalue240/campaign_controller.py:228

The provisional safe deployment plan is:

1. Resolve the blocking findings and rerun the complete local gate.
2. Preflight physical nodes serially in this order:
   P12, P13, P14, P24, P25, P26, P34, P35, P36, P56, P62, P63, P64.
3. Never overlap SSH, SCP, or SFTP across the fleet.
4. Keep P36 as the provisional reserve because historical records identify it
   as a 185 W slower card with SSH/reboot recovery history. Final reserve choice
   must use current serial preflight evidence.
5. Run the genuine ten-machine canaries only after the bootstrap path is fixed.
6. Assign S001 and S002 to the two fastest thermally stable nodes for the pilot.
7. After an explicit pilot closeout and promotion, assign all eight Cycle-1 seed
   blocks to the eight fastest healthy nodes.
8. Keep all repeated-read inputs on real C-drive SSD storage. D/mechanical
   storage is output-only.

If twelve simultaneous training machines are mandatory, the execution protocol
must be revised. One possible later-phase design is to run eight seed prefixes
through epoch 140, then distribute verified arm continuations across more nodes.
That changes the current co-location and cross-machine-resume policy and requires
new preregistration, queue, identity evidence, gate, release, and assignment
assets. It must not be introduced as an informal scheduler override.

## Recommended remediation order

1. Establish a clean, immutable source-tree identity and regenerate frozen
   validation hashes using committed LF bytes.
2. Design and test a non-circular, one-epoch real-data canary with a production
   node-report writer.
3. Bind the actual machine configuration and live source tree to assignment and
   release identities.
4. Close the live dataset, hardlink-cache, validation-split, and checkpoint
   prediction identity chains.
5. Replace completed_epochs-based skip logic with a fully validated segment
   completion receipt.
6. Include role-loss evidence in the worker's completion transaction.
7. Make lease expiry and terminal publication safe under cross-node clock skew
   and assignment fencing.
8. Require identity-bearing lower-level evidence before engineering-gate
   binding.
9. Implement explicit pilot closeout, confirmatory release, and Cycle-2
   scientific release paths.
10. Remove bare-Python project subprocesses in favor of the frozen uv
    environment.
11. Add regression tests for every defect above, then rerun focused, tracked,
    clean-checkout, canary, failure-injection, lease, and documentation suites.
12. Only after all gates pass, deploy one physical node at a time and perform
    visible foreground training launches.

## Operational constraints to preserve

- At most one SSH/SCP/SFTP connection across the entire fleet at any instant.
- Check local SSH-family processes and established TCP/22 before and after every
  remote operation.
- Deploy and verify one physical machine before touching the next.
- Use StrictHostKeyChecking=yes with fixed known_hosts; never accept-new.
- Use uv for project commands; never bare Python.
- Keep datasets, manifests, staging, workdirs, active caches, code, environment,
  and initial weights on real C-drive SSD storage.
- Use D/mechanical disks only for outputs, artifacts, logs, and packages.
- Do not change frozen scientific hyperparameters during resource preflight.
- Start formal training in a visible foreground or interactive task.
- Preserve failed-attempt evidence and prevent duplicate ownership.
- Do not kill ToDesk and do not perform broad cleanup.

## Handoff state

No source fix was made as part of this review. No training machine was contacted,
no assignment was activated, and no formal or canary job was launched.

The next device should treat this document as a defect report, reproduce each
finding against the referenced commit, implement fixes on a new reviewed commit,
and update this handoff with resolved commit IDs and validation results.
