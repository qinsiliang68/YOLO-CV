# Twelve-machine deployment

The central preparation step creates `machine_01_jobs.csv` through `machine_10_jobs.csv`, each containing eight complete triads (24 runs). Machine 11 and 12 shards are empty and reserved for failure takeover.

All 12 shard files are SHA-bound by runtime contract v1.2. A controller refuses a shard that is modified, incomplete, assigned to another machine ID, or no longer forms the unique 240-run allocation.

T, R1, and R2 of a triad should stay on the same physical machine. Machine-local JSONL registries are merged after collection; machines do not concurrently write one network SQLite database.

A reserve takeover is an AIOps/operator action, not an automatic code decision. It copies the frozen release and machine assets, creates new attempt IDs, and normally reruns the complete T/R1/R2 triad. Old attempts remain auditable; a replacement becomes active only after validation.

Before formal dispatch, every machine must have a PASS machine-asset report for the same content snapshot. The code owns run/GPU/staging locks and exit codes; AIOps owns restart, cleanup, alerting, retry caps, and approval to use a reserve machine.
