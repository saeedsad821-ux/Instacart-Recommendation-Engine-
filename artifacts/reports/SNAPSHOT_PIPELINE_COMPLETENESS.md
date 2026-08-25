# SNAPSHOT PIPELINE COMPLETENESS

The final inference path is fully functional and self-contained within `run_inference.py`:
1. `user_id` retrieval (from internal `data/processed/product_data.csv`)
2. History filtering and candidates loading
3. Feature injection mapping `is_ordered_history` and `order_number_history`
4. Point in time evaluations for `seq_recency_last_order` and `seq_purchase_trend`
5. Scoring via loaded LightGBM Champion
6. Top-K ranking execution
