# P56 / S003 / T_U deployment note

Recorded: 2026-08-25 (Asia/Shanghai)

Status: `DEPLOYED_AND_FORMAL_TRAINING_ACTIVE`

- Node: `P56` / `DESKTOP-1KDVL14` / RTX 3090
- Run: `SCTSR_DISCOVERY_S003_T_U`
- Arm / seed: `T_U` / `322761319`
- Task: `SCTSR_P56_DISCOVERY_S003_T_U_20260825_V1`
- Output: `D:\ssh\AI\artifacts\sctsr_v4_formal_discovery_s003_t_u_20260825_20a9558`
- Frozen source commit: `20a9558e36b8782857f54670ae8cf79d3fb2554d`
- P56 source-tree digest: `2CF4A100203B46EF69E5BF47AE378F1241BB59985E393C844D8324290C047A90`
- Parent checkpoint SHA-256: `1AE55561CE2178E7879803A0D53613E519863096B2845B7743A8F44D858417C8`
- Schedule digest: `2BEAB0016B677EB548FC5CA9288C29FDFDCD746777C058D4D98A7D5E85110CA4`
- T-pool digest: `D9702F54DA3D9C7C4E27B657B7EC7A5FD235DEC72E2257AE5029E0C62D7482C7`

The source-tree digest is intentionally P56-specific because the formal runtime identity binds the installed NVIDIA driver (`610.62`). It was rebound to the machine's measured runtime instead of copying P64's driver-bound digest.

## Data and startup

No 28 GiB dataset transfer was performed. P56 used 384,000 same-volume hardlinks for the canonical dataset and 143,996 hardlinks for the classification view. The canonical logical size is 82,637,967,451 bytes; image-content network copy was zero.

The first claimed attempt completed the required physical content SHA pass once, then failed before training because Ultralytics' auxiliary AMP equivalence probe found no bundled `assets/bus.jpg`. That attempt and its generation-1 terminal evidence remain preserved. A legal pre-output generation-2 resume reused the exact completed validation receipt and immutable hardlink receipts, so the 82.6 GB content was not read a second time. Training AMP remains enabled; only the offline auxiliary equivalence probe was skipped after the RTX 3090 identity/capability check.

An earlier pre-claim setup attempt also remains preserved. It stopped because two compact parent-index byte counts had been copied from another seed even though their hashes matched; the P56 values were corrected before the formal claim.

## First complete epoch validation

`P56_E121_VALIDATION.json` is the final validation receipt and has status `PASS`; its SHA-256 is `6F98B2B10DA919F1F39F37619965DC109959711046606CA886BA254F9E8AB046`.

- E121 generation digest: `FA5E479475C9A849F55F9844B336D404F8205E6367FD4FD0807C9B7147885161`
- E121 checkpoint SHA-256: `52CB20F26AEF3B6CD54622C000C859E919FC820B2FF7D9AD6BD731BDB23ECD62`
- All seven generation-manifest files match their recorded bytes and SHA-256.
- Occurrence ledger: 120,600 rows = 120,000 base + 600 replay across 938 base steps.
- All 600 replay IDs are unique, match the frozen E121 schedule, match the reconstructed step-slot mapping, and belong to the frozen T identity pool.
- Replay evidence binds `selection_policy=T_STRESS`, `identity_pool_id=T_STRESS_POOL`, and the training identity manifest's `normal_replay` role.
- At the final snapshot, E122 was complete and E123 was active; the scheduled task remained `Running`.

Resource snapshot: GPU 7,184/24,576 MiB, 81°C, 256.47/370 W; 81°C is inside the 78–82°C normal operating band, so no power adjustment was made. Free resources were 13.48 GiB system memory, 60.31 GiB on C, and 231.87 GiB on D.

All failed attempts, claim evidence, epoch ledgers, checkpoints, caches, and intermediate products remain on P56. This note records deployment correctness only and makes no scientific-effectiveness claim.
