# Gate Bucket Pilot Preflight

- task_name: `stage1_formal_gate_bucket_pilot_machine_b`
- teacher_ratio_id: `hn00`
- teacher_checkpoint_path: `C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\runs\stage1_formal_gate\yolo11m_gate2_formal_200ep\weights\epoch_078.pt`
- teacher_best_epoch: `78`
- fixed_budget_count: `151`
- candidate_top_k: `250`
- experiments_planned: `C-Q3, C-Q4, C-Q5, D-Q1, D-Q2, D-Q3, D-Q4, D-Q5`
- ready_for_full_train: `True`

Required inputs:
- source_dataset: `True`
- split_csv: `True`
- teacher_summary_dir: `True`
- teacher_best_manifest: `True`
- teacher_checkpoint_path: `True`
- hn_summary_csv: `True`
- pool_top_csv: `True`
