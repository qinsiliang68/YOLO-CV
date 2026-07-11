# Twelve-machine deployment

The central preparation step creates `machine_01_jobs.csv` through `machine_10_jobs.csv`, each containing eight complete triads (24 runs). Machine 11 and 12 shards are empty and reserved for failure takeover.

T, R1, and R2 of a triad should stay on the same physical machine. Machine-local JSONL registries are merged after collection; machines do not concurrently write one network SQLite database.

A reserve takeover copies the frozen matrix, selections, contract, and site checkpoint binding. It creates new attempt IDs and reruns all three arms. Old isolated arms are preserved but superseded.
