# FINAL DATA LINEAGE AUDIT
Raw CSV -> Preprocessing -> Split generation (Time-based boundaries) -> Candidate Generation (Top 100 Popularity/Reorder) -> Feature Engineering -> LightGBM LambdaRanker Training -> Inference -> Evaluation.
- Temporal boundaries are respected.
- Features generated prior to Target observation (Order N).
