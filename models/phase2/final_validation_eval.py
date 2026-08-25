import os
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

def evaluate_model_slices(df_eval, booster, item_stats, product_names):
    """
    Evaluates LightGBM LambdaRanker and Popularity baseline across user history length segments:
    - High-history (> 20 orders)
    - Medium-history (5 to 20 orders)
    - Low-history (< 5 orders)
    """
    def count_ones(s): return sum(1 for x in str(s).split() if x == '1')
    def get_max_val(s): 
        parts = str(s).split()
        return int(parts[-1]) if parts else 1
    def get_last_float(s):
        parts = str(s).split()
        return float(parts[-1]) if parts else 7.0

    df = df_eval.copy()
    df['hist_order_count'] = df['is_ordered_history'].apply(count_ones).astype(np.int16)
    df['user_total_orders'] = df['order_number_history'].apply(get_max_val).astype(np.int16)
    df['user_reorder_rate'] = (df['hist_order_count'] / df['user_total_orders']).astype(np.float32)
    df['recency_days'] = df['days_since_prior_order_history'].apply(get_last_float).astype(np.float32)
    df['aisle_id'] = df['aisle_id'].astype(np.int16)
    df['department_id'] = df['department_id'].astype(np.int8)

    df['global_order_count'] = df['product_id'].map(lambda pid: item_stats.get(pid, {}).get('order_count', 0)).astype(np.int32)
    df['global_reorder_rate'] = df['product_id'].map(lambda pid: item_stats.get(pid, {}).get('reorder_rate', 0.0)).astype(np.float32)

    features = [
        'hist_order_count', 'user_total_orders', 'user_reorder_rate',
        'recency_days', 'global_order_count', 'global_reorder_rate',
        'aisle_id', 'department_id'
    ]
    
    # Predict LightGBM scores
    df['lgbm_score'] = booster.predict(df[features])
    # Predict Popularity scores
    df['pop_score'] = df['global_order_count'] * (0.5 + 0.5 * df['global_reorder_rate'])

    # Helper for NDCG@10 and Recall@10 per user
    def calc_metrics_for_group(group_df, score_col, k=10):
        ndcg_list = []
        recall_list = []
        precision_list = []
        hit_list = []
        recommended_items = set()

        for uid, u_df in group_df.groupby('user_id'):
            pos_items = set(u_df[u_df['label'] == 1]['product_id'])
            if not pos_items:
                continue
            
            top_k_df = u_df.sort_values(score_col, ascending=False).head(k)
            rec_items = list(top_k_df['product_id'])
            for item in rec_items:
                recommended_items.add(item)

            # Precision & Recall
            hits = len(set(rec_items) & pos_items)
            precision_list.append(hits / k)
            recall_list.append(hits / len(pos_items))
            hit_list.append(1.0 if hits > 0 else 0.0)

            # NDCG@k
            dcg = 0.0
            for rank, item in enumerate(rec_items):
                if item in pos_items:
                    dcg += 1.0 / np.log2(rank + 2)
            idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(pos_items), k)))
            ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)

        return {
            "ndcg_at_10": float(np.mean(ndcg_list)) if ndcg_list else 0.0,
            "recall_at_10": float(np.mean(recall_list)) if recall_list else 0.0,
            "precision_at_10": float(np.mean(precision_list)) if precision_list else 0.0,
            "hit_rate_at_10": float(np.mean(hit_list)) if hit_list else 0.0,
            "catalog_coverage": float(len(recommended_items) / 49688.0),
            "eval_users": len(ndcg_list)
        }

    # Slice by history length
    high_users = df[df['user_total_orders'] > 20]
    med_users = df[(df['user_total_orders'] >= 5) & (df['user_total_orders'] <= 20)]
    low_users = df[df['user_total_orders'] < 5]

    results = {
        "LightGBM": {
            "high_history": calc_metrics_for_group(high_users, 'lgbm_score'),
            "medium_history": calc_metrics_for_group(med_users, 'lgbm_score'),
            "low_history": calc_metrics_for_group(low_users, 'lgbm_score')
        },
        "Popularity": {
            "high_history": calc_metrics_for_group(high_users, 'pop_score'),
            "medium_history": calc_metrics_for_group(med_users, 'pop_score'),
            "low_history": calc_metrics_for_group(low_users, 'pop_score')
        }
    }
    return results

def bootstrap_confidence_intervals(df_eval, booster, item_stats, n_boot=200, seed=42):
    """
    Computes 95% Bootstrap Confidence Interval on Delta NDCG@10 between LightGBM and Popularity Baseline.
    """
    np.random.seed(seed)
    def count_ones(s): return sum(1 for x in str(s).split() if x == '1')
    def get_max_val(s): 
        parts = str(s).split()
        return int(parts[-1]) if parts else 1
    def get_last_float(s):
        parts = str(s).split()
        return float(parts[-1]) if parts else 7.0

    df = df_eval.copy()
    df['hist_order_count'] = df['is_ordered_history'].apply(count_ones).astype(np.int16)
    df['user_total_orders'] = df['order_number_history'].apply(get_max_val).astype(np.int16)
    df['user_reorder_rate'] = (df['hist_order_count'] / df['user_total_orders']).astype(np.float32)
    df['recency_days'] = df['days_since_prior_order_history'].apply(get_last_float).astype(np.float32)
    df['aisle_id'] = df['aisle_id'].astype(np.int16)
    df['department_id'] = df['department_id'].astype(np.int8)

    df['global_order_count'] = df['product_id'].map(lambda pid: item_stats.get(pid, {}).get('order_count', 0)).astype(np.int32)
    df['global_reorder_rate'] = df['product_id'].map(lambda pid: item_stats.get(pid, {}).get('reorder_rate', 0.0)).astype(np.float32)

    features = [
        'hist_order_count', 'user_total_orders', 'user_reorder_rate',
        'recency_days', 'global_order_count', 'global_reorder_rate',
        'aisle_id', 'department_id'
    ]
    df['lgbm_score'] = booster.predict(df[features])
    df['pop_score'] = df['global_order_count'] * (0.5 + 0.5 * df['global_reorder_rate'])

    # Calculate per-user NDCG@10 for both models
    lgbm_ndcgs = []
    pop_ndcgs = []
    
    for uid, u_df in df.groupby('user_id'):
        pos_items = set(u_df[u_df['label'] == 1]['product_id'])
        if not pos_items:
            continue
        
        # LightGBM NDCG
        top_lgbm = list(u_df.sort_values('lgbm_score', ascending=False).head(10)['product_id'])
        dcg = sum(1.0 / np.log2(r + 2) for r, item in enumerate(top_lgbm) if item in pos_items)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(pos_items), 10)))
        lgbm_ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        # Pop NDCG
        top_pop = list(u_df.sort_values('pop_score', ascending=False).head(10)['product_id'])
        dcg_pop = sum(1.0 / np.log2(r + 2) for r, item in enumerate(top_pop) if item in pos_items)
        pop_ndcgs.append(dcg_pop / idcg if idcg > 0 else 0.0)

    lgbm_ndcgs = np.array(lgbm_ndcgs)
    pop_ndcgs = np.array(pop_ndcgs)
    diffs = lgbm_ndcgs - pop_ndcgs

    boot_means = []
    n_users = len(diffs)
    for _ in range(n_boot):
        idx = np.random.choice(n_users, size=n_users, replace=True)
        boot_means.append(np.mean(diffs[idx]))

    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    mean_diff = float(np.mean(diffs))

    return {
        "mean_diff_ndcg_at_10": mean_diff,
        "ci_95_lower": ci_low,
        "ci_95_upper": ci_high,
        "statistically_significant": bool(ci_low > 0.0)
    }

if __name__ == '__main__':
    print("[INFO] Starting Final Controlled Validation (Generalization Slices & Bootstrap CI)...")
    t0 = time.time()
    root_dir = "."
    
    # 1. Load Item Popularity Stats
    prior_path = os.path.join(root_dir, "data/raw/order_products__prior.csv")
    if not os.path.exists(prior_path):
        prior_path = os.path.join(root_dir, "../order_products__prior.csv")
    prior_df = pd.read_csv(prior_path, usecols=['product_id', 'reordered'])
    stats = prior_df.groupby('product_id').agg(
        order_count=('reordered', 'count'),
        reorder_count=('reordered', 'sum')
    ).reset_index()
    stats['reorder_rate'] = stats['reorder_count'] / (stats['order_count'] + 1e-5)
    item_stats = {r['product_id']: {'order_count': r['order_count'], 'reorder_rate': r['reorder_rate']} for _, r in stats.iterrows()}

    # 2. Load LightGBM Model
    model_file = os.path.join(root_dir, "artifacts/models/lightgbm_ranker/lightgbm_ranker.txt")
    booster = lgb.Booster(model_file=model_file)

    # 3. Load Test Users Sample for fast & accurate evaluation
    test_users_file = os.path.join(root_dir, "artifacts/splits/user_test.parquet")
    test_users_df = pd.read_parquet(test_users_file)
    test_user_set = set(test_users_df['user_id'])
    
    # Read product_data chunks for test users
    test_chunks = []
    count = 0
    for chunk in pd.read_csv(os.path.join(root_dir, "data/processed/product_data.csv"), chunksize=500000, keep_default_na=False):
        sub = chunk[chunk['user_id'].isin(test_user_set)]
        if not sub.empty:
            test_chunks.append(sub)
            count += sub['user_id'].nunique()
            if count >= 3000:  # Representative 3,000 test users for bootstrap & slicing
                break
    
    df_eval = pd.concat(test_chunks, ignore_index=True)
    print(f"[INFO] Loaded {df_eval['user_id'].nunique():,} unseen test users for validation evaluation.")

    # 4. Evaluate across history slices
    print("[INFO] Evaluating High, Medium, and Low history user slices...")
    slice_results = evaluate_model_slices(df_eval, booster, item_stats, {})
    
    # 5. Compute Bootstrap CI
    print("[INFO] Computing 95% Bootstrap Confidence Interval...")
    ci_results = bootstrap_confidence_intervals(df_eval, booster, item_stats, n_boot=200)

    final_results = {
        "user_history_slices": slice_results,
        "bootstrap_ci": ci_results,
        "evaluation_time_sec": time.time() - t0
    }

    out_path = os.path.join(root_dir, "artifacts/metrics/final_validation_metrics.json")
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"[SUCCESS] Final Controlled Validation completed in {time.time() - t0:.2f}s!")
    print(json.dumps(final_results, indent=2))
