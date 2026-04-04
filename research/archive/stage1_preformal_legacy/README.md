# Stage-1 Preformal Legacy Archive

This archive namespace is the **official home for exploratory stage-1 lines**.

It exists to separate:

- legacy exploratory evidence
- pilot ablations
- appendix-only materials

from:

- new thesis-facing formal capacity-scan outputs

## Archive Buckets

- `general/`
- `gate_capacity/`
- `gate_calibration/`
- `gate_hn/`
- `gate_maxfilter/`
- `gate_ptsg/`
- `gate_supcon/`
- `gate_rcis/`

## Current Compatibility Policy

Some existing exploratory materials still physically remain under old `research/materials/` and `research/results/` paths because active scripts still reference them.

That compatibility is temporary.

Effective immediately:

- **new formal outputs** must go to `research/materials/stage1_formal/` and `research/results/stage1_formal/`
- **new exploratory outputs** should be placed under this archive namespace instead of creating new top-level stage-1 folders

## Current Legacy Mapping

The current top-level legacy families map into archive buckets as follows:

| Current family | Archive bucket |
| --- | --- |
| legacy stage-1 memory, old traceability notes | `general/` |
| binary/six-class exploratory capacity materials, raw model-selection summaries | `gate_capacity/` |
| temperature scaling and calibration comparison materials | `gate_calibration/` |
| HN backflow, HN sweep, HN ratio materials | `gate_hn/` |
| max-filter and hard-sample training-side exploratory reports | `gate_maxfilter/` |
| PTSG and trust-gate exploratory materials | `gate_ptsg/` |
| SupCon / strong-embedding exploratory materials | `gate_supcon/` |
| RCIS exploratory design notes and result templates | `gate_rcis/` |

If a line spans multiple themes, archive it under its **primary experimental role**, not under every possible bucket.
