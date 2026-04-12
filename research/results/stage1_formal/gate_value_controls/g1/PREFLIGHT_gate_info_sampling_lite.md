# PREFLIGHT: Gate Info Sampling Lite

- mode: `dry_run`
- smoke_epochs: `0`
- smoke_setting: `none`
- teacher ratio: `hn00`
- teacher checkpoint exists: `False`
- pool exact reuse source: `C:\GitHub\YOLO-CV\research\materials\stage1_formal\gate_hn_assets\yolo11m_train_normal_scores\top_false_positive_normals.csv`
- fixed budget count: `151`
- checked settings: `G1`
- pool provenance matches current teacher: `True`
- core inputs ready: `False`
- rerun cleanup scope safe: `True`
- ready_for_full_train: `False`

## Required inputs
- hn_summary_csv: `True`
- teacher_best_manifest: `True`
- uniform_best_manifest: `True`
- pool_top_csv: `True`
- pool_scores_csv: `True`
- pool_summary_json: `True`
- source_dataset: `False`

## Per-setting score checks
### G1
- score csv exists: `False`
- schema ok: `False`
- row_count: `NA`
- candidate_top_k_matches: `NA`
- pool_exact_reuse_matches_top250: `NA`
- sum(pi): `NA`
- sum(pi) close to 1: `NA`
- score_has_nan/inf: `NA` / `NA`
- probability_has_nan/inf: `NA` / `NA`
- duplication_total matches budget: `NA`
- dataset duplication_total_matches_expected: `NA`
