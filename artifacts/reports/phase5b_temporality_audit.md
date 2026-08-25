# Phase 5B Temporality Audit

Global order count and reorder rate are derived entirely from `order_products__prior.csv`.
Candidate generation also inherently uses `eval_set == prior`.
No future or target data is present in these variables.
