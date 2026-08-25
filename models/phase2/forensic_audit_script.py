import os
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

def run_forensic_audit(root_dir="."):
    print("=========================================================================")
    print("              PHASE 0: EXHAUSTIVE FORENSIC AUDIT SCRIPT                  ")
    print("=========================================================================")
    
    # A. Split audit
    splits_dir = os.path.join(root_dir, "artifacts/splits")
    u_train = set(pd.read_parquet(os.path.join(splits_dir, "user_train.parquet"))['user_id'])
    u_val = set(pd.read_parquet(os.path.join(splits_dir, "user_validation.parquet"))['user_id'])
    u_test = set(pd.read_parquet(os.path.join(splits_dir, "user_test.parquet"))['user_id'])
    
    print(f"[SPLIT AUDIT] Train users: {len(u_train):,}, Val users: {len(u_val):,}, Test users: {len(u_test):,}")
    print(f"[SPLIT AUDIT] Train INTERSECT Val: {len(u_train & u_val)}, Train INTERSECT Test: {len(u_train & u_test)}, Val INTERSECT Test: {len(u_val & u_test)}")
    
    # B. Checkpoint audit
    lgbm_dir = os.path.join(root_dir, "artifacts/models/lightgbm_ranker")
    lgbm_txt = os.path.join(lgbm_dir, "lightgbm_ranker.txt")
    lgbm_meta = json.load(open(os.path.join(lgbm_dir, "metadata.json")))
    print(f"[CHECKPOINT] LightGBM File Size: {os.path.getsize(lgbm_txt) / 1024:.2f} KB")
    print(f"[CHECKPOINT] LightGBM Meta: {json.dumps(lgbm_meta, indent=2)}")

    sas_dir = os.path.join(root_dir, "artifacts/models/sasrec")
    sas_pt = os.path.join(sas_dir, "sasrec.pt")
    sas_meta = json.load(open(os.path.join(sas_dir, "metadata.json")))
    print(f"[CHECKPOINT] SASRec File Size: {os.path.getsize(sas_pt) / (1024*1024):.2f} MB")
    print(f"[CHECKPOINT] SASRec Meta: {json.dumps(sas_meta, indent=2)}")

    # C. Evaluate across 6 History Buckets on Test set
    print("[INFO] Evaluating Generalization across 6 User History Buckets...")
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

    booster = lgb.Booster(model_file=lgbm_txt)

    test_chunks = []
    count = 0
    for chunk in pd.read_csv(os.path.join(root_dir, "data/processed/product_data.csv"), chunksize=500000, keep_default_na=False):
        sub = chunk[chunk['user_id'].isin(u_test)]
        if not sub.empty:
            test_chunks.append(sub)
            count += sub['user_id'].nunique()
            if count >= 3500:
                break
    df_eval = pd.concat(test_chunks, ignore_index=True)

    def count_ones(s): return sum(1 for x in str(s).split() if x == '1')
    def get_max_val(s): 
        parts = str(s).split()
        return int(parts[-1]) if parts else 1
    def get_last_float(s):
        parts = str(s).split()
        return float(parts[-1]) if parts else 7.0

    df_eval['hist_order_count'] = df_eval['is_ordered_history'].apply(count_ones).astype(np.int16)
    df_eval['user_total_orders'] = df_eval['order_number_history'].apply(get_max_val).astype(np.int16)
    df_eval['user_reorder_rate'] = (df_eval['hist_order_count'] / df_eval['user_total_orders']).astype(np.float32)
    df_eval['recency_days'] = df_eval['days_since_prior_order_history'].apply(get_last_float).astype(np.float32)
    df_eval['aisle_id'] = df_eval['aisle_id'].astype(np.int16)
    df_eval['department_id'] = df_eval['department_id'].astype(np.int8)

    df_eval['global_order_count'] = df_eval['product_id'].map(lambda pid: item_stats.get(pid, {}).get('order_count', 0)).astype(np.int32)
    df_eval['global_reorder_rate'] = df_eval['product_id'].map(lambda pid: item_stats.get(pid, {}).get('reorder_rate', 0.0)).astype(np.float32)

    features = [
        'hist_order_count', 'user_total_orders', 'user_reorder_rate',
        'recency_days', 'global_order_count', 'global_reorder_rate',
        'aisle_id', 'department_id'
    ]
    df_eval['lgbm_score'] = booster.predict(df_eval[features])
    df_eval['pop_score'] = df_eval['global_order_count'] * (0.5 + 0.5 * df_eval['global_reorder_rate'])

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
            hits = len(set(rec_items) & pos_items)
            precision_list.append(hits / k)
            recall_list.append(hits / len(pos_items))
            hit_list.append(1.0 if hits > 0 else 0.0)

            dcg = sum(1.0 / np.log2(r + 2) for r, item in enumerate(rec_items) if item in pos_items)
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

    # Define 5 non-zero buckets (plus cold-start 0 orders reported from cold_start_metrics.json)
    buckets = {
        "1-2 orders": df_eval[(df_eval['user_total_orders'] >= 1) & (df_eval['user_total_orders'] <= 2)],
        "3-4 orders": df_eval[(df_eval['user_total_orders'] >= 3) & (df_eval['user_total_orders'] <= 4)],
        "5-10 orders": df_eval[(df_eval['user_total_orders'] >= 5) & (df_eval['user_total_orders'] <= 10)],
        "11-20 orders": df_eval[(df_eval['user_total_orders'] >= 11) & (df_eval['user_total_orders'] <= 20)],
        ">20 orders": df_eval[df_eval['user_total_orders'] > 20]
    }

    bucket_results = {}
    for name, b_df in buckets.items():
        bucket_results[name] = {
            "LightGBM": calc_metrics_for_group(b_df, 'lgbm_score'),
            "Popularity": calc_metrics_for_group(b_df, 'pop_score')
        }

    out_file = os.path.join(root_dir, "artifacts/metrics/forensic_generalization_buckets.json")
    with open(out_file, "w") as f:
        json.dump(bucket_results, f, indent=2)
    print(f"[SUCCESS] Saved bucket results to {out_file}")
    print(json.dumps(bucket_results, indent=2))

if __name__ == "__main__":
    run_forensic_audit()
