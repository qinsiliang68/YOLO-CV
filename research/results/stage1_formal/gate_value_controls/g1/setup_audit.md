# Setup Audit

- base model: `yolo11m-cls`
- teacher ratio: `hn00`
- teacher checkpoint: `C:\GitHub\YOLO-CV\YOLOv11\runs\stage1_formal_gate\yolo11m_gate2_formal_200ep\weights\epoch_078.pt`
- teacher checkpoint exists: `False`
- uniform anchor ratio: `hn14`
- budget anchor ratio: `hn14`
- fixed budget count: `151`
- fixed pool size: `250`
- pool provenance matches current teacher: `True`
- source dataset exists: `False`
- core inputs ready: `False`

Required inputs:
- hn_summary_csv: `True`
- teacher_best_manifest: `True`
- uniform_best_manifest: `True`
- pool_top_csv: `True`
- pool_scores_csv: `True`
- pool_summary_json: `True`
- source_dataset: `False`

Notes:
- Teacher, candidate pool width, and probability concentration can be varied by config for controlled experiments.
- This suite keeps the replay budget fixed to the hn14-equivalent extra-normal count unless explicitly redefined.
- Anchor settings may be reused while only the configured new settings are newly trained.
