# table_hn_hard_normal_selection_stats

1. This asset answers: What portion of the train-normal pool is selected into the hard-normal candidate set and what the selected-pool score range looks like.
2. Source files:
   - research/materials/stage1_formal/gate_hn_assets/yolo11m_train_normal_scores/train_normal_scores.csv
   - research/materials/stage1_formal/gate_hn_assets/yolo11m_train_normal_scores/top_false_positive_normals.csv
   - research/materials/stage1_formal/gate_hn_assets/yolo11m_train_normal_scores/summary.json
3. Ranking/selection rule: The selected hard normals are summarized using the existing score assets rather than any new inference run.
4. Key finding: The table reports selected-pool score statistics, while the threshold column records the minimum selected score.
5. Limitation: Heuristic labels in the source CSV are not used here because the key evidence is the score-tail behavior.
