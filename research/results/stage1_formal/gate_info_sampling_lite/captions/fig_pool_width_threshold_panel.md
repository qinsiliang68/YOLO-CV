# fig_pool_width_threshold_panel

1. This asset answers: Whether the top250 hard-normal pool is a narrow extreme tail or a relatively wide high-risk region under the hn00 teacher.
2. Source files:
   - research/materials/stage1_formal/gate_info_sampling_lite/score_inputs/pool_source_manifest.json
   - C:/Users/ASUS/Desktop/YOLOv11/YOLO-CV/research/materials/stage1_formal/gate_hn_assets/yolo11m_train_normal_scores/train_normal_scores.csv
3. Ranking/selection rule: The figure uses the hn00 teacher thresholds tau_r995 and tau_r990 together with the fixed top250 cutoff.
4. Key finding: The selected top250 region extends far below the formal threshold anchors, which supports the view that the hard-normal pool is wide rather than near-homogeneous.
5. Limitation: This figure summarizes the pool scores only and does not by itself identify which subregion is most learnable.
