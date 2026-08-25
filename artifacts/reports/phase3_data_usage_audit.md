# Phase 3 — Forensic Data Usage Audit Report

## 1. Complete Dataset Population Audit
- **Total Dataset Users**: `206,209 unique users`
- **Catalog Products**: `49,688 unique items`
- **Total Raw Prior Interactions**: `32,434,489 rows` across all `206,209 users`
- **Target Labeled Baskets**: `131,209 baskets` (`eval_set == 'train'`, `1,384,617 items`)
- **Unlabeled Kaggle Baskets**: `75,000 baskets` (`eval_set == 'test'`)

## 2. Canonical Split (`split_manifest.json`, `seed=42`)
- **Train Split (70%)**: `144,346 users` (`91,847 labeled train users`; **`22,750,089 prior interaction rows`**; `9,479,779 static candidate rows`)
- **Validation Split (15%)**: `30,931 users` (`20,924 labeled validation users`; `4,837,442 prior interaction rows`; `2,017,612 static candidate rows`)
- **Test Split (15%)**: `30,932 users` (`18,438 labeled test users`; `4,846,958 prior interaction rows`; `2,016,771 static candidate rows`)
- **Disjointness Guarantee**: `Train ∩ Val = 0`, `Train ∩ Test = 0`, `Val ∩ Test = 0` (100% verified).

## 3. Candidate Pool Density
- **Static Historical Candidate Density**: `65.54 average unique historical items / user`
- **Dynamic Prediction Pool**: Dynamically padded with Top Global Popularity fallback to exactly **`100.0 items / user`** (`94.06% Candidate Recall@100`).
