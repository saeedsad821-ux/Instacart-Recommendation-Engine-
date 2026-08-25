# Phase 3 — Temporal Forensic Audit & Holdout Protocol Report

## 1. Relative Temporal Protocol (Orders $1 \dots N-1 ightarrow N$)
- **Why no calendar dates are invented**: The raw Instacart Market Basket Analysis dataset contains relative inter-order intervals (`days_since_prior_order`, `order_dow`, `order_hour_of_day`) but zero absolute timestamp timestamps. In accordance with Hard Constraint H1, no calendar dates were fabricated.
- **Forward-Looking Temporal Protocol**:
  - For every user $u$, all historical interactions are strictly bounded to orders $1 \dots N-1$.
  - The target order to rank is strictly order $N$ (the user's maximum observed order).
  - Target order $N$ never contributes to candidate pools, reorder rates, item counts, or sequential embeddings.

## 2. Temporal Lifecycle Stability Analysis (`Early`, `Middle`, `Late`)
We evaluated recommendation stability across three user lifecycle periods based on target order $N$:
- **Early Period** ($N \in [3, 5]$, `969 users`): Predicting early replenishment baskets when the user has only 1–2 historical orders.
- **Middle Period** ($N \in [6, 15]$, `1,403 users`): Predicting mid-stage replenishment baskets.
- **Late Period** ($N > 15$, `1,141 users`): Predicting mature replenishment baskets after 15+ historical orders.

### Temporal Stability Table (`NDCG@10`)
| Model | Early Period ($N=3..5$) | Middle Period ($N=6..15$) | Late Period ($N>15$) | Mean | Std | Worst Period |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity** | 0.4386 | 0.3215 | 0.2521 | 0.3374 | 0.0769 | Late |
| **SASRec** | 0.2910 | 0.2845 | 0.2810 | 0.2855 | 0.0041 | Late |
| **LightGBM** | **0.5860** | **0.5177** | **0.4818** | **0.5285** | **0.0432** | Late |
| **Ensemble** | 0.5851 | 0.5150 | 0.4777 | 0.5259 | 0.0445 | Late |

### Key Findings
1. **Mature User Robustness**: In late-period mature baskets ($N > 15$), Global Popularity degrades severely to `0.2521 NDCG@10` because mature users buy personalized, idiosyncratic items. LightGBM maintains `0.4818 NDCG@10` (nearly **2x higher** than Popularity).
2. **Early Period Personalization**: Even with only 1–2 historical orders ($N \in [3,5]$), LightGBM achieves `0.5860 NDCG@10`, beating Popularity (`0.4386`) by `+0.1474`.
3. **Sequential Model Failure**: SASRec (`0.2855 Mean NDCG@10`) fails across all three lifecycle periods because grocery replenishment is dominated by item repeat frequency rather than sequential transition syntax.
