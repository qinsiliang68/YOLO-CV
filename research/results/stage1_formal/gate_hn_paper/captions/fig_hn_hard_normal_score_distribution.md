# fig_hn_hard_normal_score_distribution

1. This asset answers: Where the selected hard normals sit within the train-normal abnormal-score distribution.
2. Source files:
   - research/materials/stage1_formal/gate_hn_assets/yolo11m_train_normal_scores/train_normal_scores.csv
   - research/materials/stage1_formal/gate_hn_assets/yolo11m_train_normal_scores/top_false_positive_normals.csv
3. Ranking/selection rule: The threshold is taken from the minimum score among the selected top false-positive normals.
4. Key finding: The selected HN pool comes from the high-score tail rather than ad hoc manual picking.
5. Limitation: This provenance view does not itself evaluate whether the selected normals are semantically diverse.
