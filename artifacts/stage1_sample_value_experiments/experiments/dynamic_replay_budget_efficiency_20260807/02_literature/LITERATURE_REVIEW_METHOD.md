# Literature review method

## Scope and reading levels

- OpenAlex shortlisted metadata records preserved: 303
- Included primary papers: 155
- Abstract screens: 67
- Method-level reads: 55
- Deep or near-full reads: 33
- Logged exclusions: 181

`ABSTRACT_SCREEN` means title and primary abstract were checked for scope.
`METHOD_READ` means the primary abstract exposed both a method mechanism and an
experimental scope; it does not claim a cover-to-cover read. `DEEP_READ` is the
manually curated mechanism set for which method, experiments, and limitations or
scope boundaries were inspected in the primary paper.

## Selection rules

1. Prefer conference or journal proceedings, DOI landing pages, OpenReview, or arXiv primary records.
2. Deduplicate normalized titles and let a manual deep-read record supersede its metadata candidate.
3. Exclude surveys, obvious domain collisions, unsupported hosts, and records without a mechanism-relevant title.
4. Balance the retained abstract pool across training dynamics, subset selection, attribution, noisy labels, replay, optimization variance, and operational-tail evaluation.
5. Use literature to define falsifiable measurements and boundaries, not to claim that an uncollected quantity was observed.

## Interpretation boundary

The matrix is a decision evidence map, not a meta-analysis. Paper counts do not vote on the correct Stage1 mechanism.
The ranked directions prioritize direct agreement with observed same-selection seed reversals, missing-field closure,
feasibility before 2026-09-10, and ability to falsify the proposed mechanism.
