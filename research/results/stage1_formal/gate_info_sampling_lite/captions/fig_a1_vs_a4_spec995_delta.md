# fig_a1_vs_a4_spec995_delta

1. This asset answers: Whether the best weighted variant reaches the same primary-metric platform as the uniform_hn14 anchor across training epochs.
2. Source files:
   - research/materials/stage1_formal/gate_hn_m_sweep/hn14/epoch_gate_summary.csv
   - research/materials/stage1_formal/gate_info_sampling_lite/weighted_hn14_risk_consistency_density/epoch_gate_summary.csv
3. Ranking/selection rule: A1 and A4 are compared on the same backbone, metric, and epoch axis; the lower panel shows A4 minus A1.
4. Key finding: A4 peaks earlier but stays below the A1 platform on the primary metric for most of training, which matches the final formal ranking gap.
5. Limitation: The comparison isolates Spec@R99.5 only and should still be read together with the full four-metric formal rule.
