# Rolling seven-day release protocol

- Local real-data smoke and failure injection precede every fleet release.
- Cycle 1 is released only after canonical-lock, all-epoch telemetry, resume, and artifact
  integrity gates pass on the ten-machine canary.
- Every completed seed block is validated and summarized immediately; daily operations
  reports are generated without changing the frozen within-cycle hypothesis.
- Cycle 2 jobs are prepared but held until the registered Cycle 1 decision artifact exists.
- Cycle 3 and Cycle 4 are templates only. They cannot be queued by an operator override.
- A cycle may finish before seven days when its complete preregistered block is done. It is
  never shortened by dropping unfavorable runs or changing endpoints.
