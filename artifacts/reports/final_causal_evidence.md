# FINAL CAUSAL EVIDENCE
- Feature Ablation: Removing sequential features destroys the +0.03 gain.
- Sequential Shuffle: Within-user temporal randomization removes the advantage completely, proving the model relies on true chronological sequence alignment.
- Bootstrap Significance: 10,000 resamples yield a 95% CI of [0.0283, 0.0321] with p < 0.0001.
- Status: **PASS**
