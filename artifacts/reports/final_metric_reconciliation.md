# FINAL METRIC RECONCILIATION
**Phase 4 (0.5493) vs Phase 5B (0.5142)**:
- Discrepancy is an evaluator artifact, not a data leakage issue.
- Phase 4 explicitly ignored users with `pos_items == 0` retrieved in candidate generation.
- Phase 5B included all evaluated users, assigning a rigid `0.0` to users with no retrieved positives.
- Phase 5B also used a 15-million row subset (19,699 users) rather than the entire set for permutation testing constraints.
- Status: **RESOLVED (Phase 5B evaluator is the authoritative, stricter metric)**
