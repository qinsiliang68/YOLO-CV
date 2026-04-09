# Setup Audit

- base model: `yolo11m-cls`
- teacher ratio: `hn00`
- teacher checkpoint: `C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\runs\stage1_formal_gate\yolo11m_gate2_formal_200ep\weights\epoch_078.pt`
- teacher checkpoint exists: `True`
- uniform anchor ratio: `hn14`
- budget anchor ratio: `hn14`
- fixed budget count: `151`
- fixed pool size: `250`
- pool provenance matches hn00 teacher: `True`
- source dataset exists: `True`
- core inputs ready: `True`

Required inputs:
- hn_summary_csv: `True`
- teacher_best_manifest: `True`
- uniform_best_manifest: `True`
- pool_top_csv: `True`
- pool_scores_csv: `True`
- pool_summary_json: `True`
- source_dataset: `True`

Notes:
- Teacher defaults to hn00 unless the existing hard-normal pool provenance contradicts this assumption.
- This suite keeps the top250 hard-normal pool fixed and only redistributes replay probability under the hn14-equivalent budget.
- A0/A1 are reused anchors; only A2/A3/A4 are newly trained.
