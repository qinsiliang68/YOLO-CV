# Dynamic replay physical run queue

This queue contains all frozen Cycle 1/2 physical jobs, compact selection templates,
the fixed OOF-only causal monitor panel, and a byte-identical canonical training lock.

No job is READY at build time. Cycle 1 is held at the engineering gate; Cycle 2 is held
at the registered Cycle 1 scientific decision. Cycle 3/4 do not have physical jobs yet.
Absolute replay rows are derived from percentage schedules and are implementation fields.
