# table_hn_ratio_coverage_manifest

1. This asset answers: Which ratio-level HN materials are currently available in the repo working set.
2. Source files:
   - research/materials/stage1_formal/gate_hn_m_sweep
   - research/materials/stage1_formal/gate_hn_x_crosscheck
   - research/results/stage1_formal/gate_hn_overview/ingest_manifest.json
3. Ranking/selection rule: Coverage is derived from the current repo paths, with source-archive provenance carried over only in the note field.
4. Key finding: The table reports repo working-set availability rather than the fuller external source archive.
5. Limitation: PT checkpoints and raw per-epoch trees are intentionally tracked as unavailable in this repo working set.
