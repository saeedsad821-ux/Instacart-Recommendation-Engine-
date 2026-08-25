# Phase 3 — Forensic Metric Consistency & Discrepancy Audit Report

## 1. Popularity Discrepancy Resolution (`+0.0067` vs `+0.1864`)
- **What population produced `+0.0067`**: The **Full Labeled Test Set (`18,438 users`)**, where LightGBM achieves `NDCG@10 = 0.5171` and Global Popularity achieves `NDCG@10 = 0.5104` (`0.5171 - 0.5104 = +0.0067`).
- **What population produced `+0.1864`**: The **Validation History-Length Bucket Sample (`2,185 users`)** from `final_validation_metrics.json`, where Popularity averaged `~0.33` across the sampled buckets while LightGBM averaged `~0.52`.
- **Why do they differ**: Global popularity is sensitive to the distribution of basket sizes in sample subsets versus the full population. On the complete 18,438-user Test set, global staple products (e.g., bananas, organic whole milk) appear in a large fraction of baskets, raising the baseline floor to `0.5104`.
- **True Bootstrap 95% CI on Full Labeled Test Set**: Over all `18,438` Labeled Test users, the paired difference `LightGBM - Popularity` is **`+0.0067`** (`95% CI: [+0.0058, +0.0076]`, statistically significant, $p < 0.001$).

## 2. Prior-Row Discrepancy Resolution (`32,434,489` vs Train Rows)
- **Total Raw Prior Interaction Rows**: `32,434,489` rows across all `206,209` users in the raw Instacart dataset.
- **Train Split Prior Rows (`144,346 users`)**: **`22,750,089` rows** (`70.14%` of total prior rows).
- **Validation Split Prior Rows (`30,931 users`)**: `4,837,442` rows (`14.91%`).
- **Test Split Prior Rows (`30,932 users`)**: `4,846,958` rows (`14.94%`).

## 3. Candidate Count Resolution (`65.54` vs `100.0` vs `13,514,162`)
- **Static Historical Candidate Rows**: `13,514,162` rows in `product_data.csv` (`65.54 average static items/user` across `206,209` users).
  - Train static rows: `9,479,779` (`65.67 items/user`).
  - Validation static rows: `2,017,612` (`65.23 items/user`).
  - Test static rows: `2,016,771` (`65.20 items/user`).
- **Dynamic Candidate Pool Size**: Padded dynamically with Top Global Popularity fallback to exactly **100.0 candidates/user**.
- **Candidate Recall@100**: **94.06%** across all Labeled Test users.

## 4. Test Population Discrepancy Resolution (`30,932` vs `18,438`)
- **Total Test Split Users**: `30,932` users.
- **Labeled Test Users**: **`18,438` users** with public ground-truth target baskets (`eval_set == 'train'`).
- **Unlabeled Test Users**: `12,494` users with unlabelled Kaggle leaderboard targets (`eval_set == 'test'`).
- **Strict Separation**: All metric tables report exclusively on the `18,438` Labeled Test users.
