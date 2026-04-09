# PREFLIGHT: Gate Info Sampling Lite

- mode: `smoke`
- smoke_epochs: `3`
- smoke_setting: `A4`
- teacher ratio: `hn00`
- teacher checkpoint exists: `True`
- pool exact reuse source: `C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\research\materials\stage1_formal\gate_hn_assets\yolo11m_train_normal_scores\top_false_positive_normals.csv`
- fixed budget count: `151`
- checked settings: `A4`
- pool provenance matches hn00 teacher: `True`
- core inputs ready: `True`
- rerun cleanup scope safe: `True`
- ready_for_full_train: `True`

## Required inputs
- hn_summary_csv: `True`
- teacher_best_manifest: `True`
- uniform_best_manifest: `True`
- pool_top_csv: `True`
- pool_scores_csv: `True`
- pool_summary_json: `True`
- source_dataset: `True`

## Per-setting score checks
### A4
- score csv exists: `True`
- schema ok: `True`
- row_count: `250`
- candidate_top_k_matches: `True`
- pool_exact_reuse_matches_top250: `False`
- sum(pi): `1.0000008099999942`
- sum(pi) close to 1: `True`
- score_has_nan/inf: `False` / `False`
- probability_has_nan/inf: `False` / `False`
- duplication_total matches budget: `True`
- dataset duplication_total_matches_expected: `True`
