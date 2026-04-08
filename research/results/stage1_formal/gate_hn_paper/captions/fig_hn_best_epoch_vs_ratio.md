# fig_hn_best_epoch_vs_ratio

1. This asset answers: How the location of the gate-best epoch changes across HN ratios.
2. Source files:
   - research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv
3. Ranking/selection rule: Best epochs are taken from the formal HN sweep summary after gate-aware checkpoint selection.
4. Key finding: The best-epoch position shifts with ratio, which indicates that HN changes the training regime rather than only the final score.
5. Limitation: This figure tracks only the selected gate-best epoch and not the full per-epoch trajectory.
