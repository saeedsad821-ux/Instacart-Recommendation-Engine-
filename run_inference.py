import os
import time
import pandas as pd
import lightgbm as lgb
import numpy as np
import argparse

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts/models/phase4/lightgbm_hybrid_seq/lightgbm_hybrid_seq.txt")
DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed/product_data.csv")

def run(user_id, top_k):
    start_total = time.time()
    
    print(f"[INFO] Loading Model...")
    bst = lgb.Booster(model_file=MODEL_PATH)
    feature_names = bst.feature_name()
    
    start_data = time.time()
    print(f"[INFO] Accessing authentic dataset...")
    user_df = None
    for chunk in pd.read_csv(DATA_PATH, chunksize=100000, keep_default_na=False):
        users = chunk[chunk['eval_set'] == 'train']
        if user_id in users['user_id'].values:
            user_df = users[users['user_id'] == user_id].copy()
            break
            
    if user_df is None or user_df.empty:
        print(f"[FAIL] User {user_id} not found in candidates.")
        return
        
    def count_ones(s): return sum(1 for x in str(s).split() if x == '1')
    def get_max_val(s): parts = str(s).split(); return int(parts[-1]) if parts else 1
    def get_last_float(s): parts = str(s).split(); return float(parts[-1]) if parts else 7.0
    def ordered_in_last_observed(s): parts = str(s).split(); return int(parts[-1]) if parts else 0
    def seq_trend(s):
        parts = [int(x) for x in str(s).split()]
        if not parts: return 0.0
        mid = len(parts) // 2
        early = sum(parts[:mid]) + 1e-5
        late = sum(parts[mid:]) + 1e-5
        return float(late / (early + late))

    print("[INFO] Computing Point-In-Time Sequential Features...")
    user_df['hist_order_count'] = user_df['is_ordered_history'].apply(count_ones).astype(np.int16)
    user_df['user_total_orders'] = user_df['order_number_history'].apply(get_max_val).astype(np.int16)
    user_df['user_reorder_rate'] = (user_df['hist_order_count'] / user_df['user_total_orders']).astype(np.float32)
    user_df['recency_days'] = user_df['days_since_prior_order_history'].apply(get_last_float).astype(np.float32)
    user_df['aisle_id'] = user_df['aisle_id'].astype(np.int16)
    user_df['department_id'] = user_df['department_id'].astype(np.int8)
    user_df['global_order_count'] = 1000 
    user_df['global_reorder_rate'] = 0.5
    user_df['seq_recency_last_order'] = user_df['is_ordered_history'].apply(ordered_in_last_observed).astype(np.int8)
    user_df['seq_purchase_trend'] = user_df['is_ordered_history'].apply(seq_trend).astype(np.float32)

    X = user_df[feature_names]

    print("[INFO] Executing LightGBM Champion Inference...")
    inf_start = time.time()
    preds = bst.predict(X)
    inf_end = time.time()

    user_df['score'] = preds
    top_items = user_df.sort_values('score', ascending=False).head(top_k)

    total_time = time.time() - start_total
    inf_latency_ms = (inf_end - inf_start) * 1000

    print(f"\n[RESULT] Inference Success!")
    print(f"User ID: {user_id}")
    print(f"History Depth (Orders): {user_df['user_total_orders'].iloc[0]}")
    print(f"Candidate Count: {len(user_df)}")
    print(f"Model-Only Latency: {inf_latency_ms:.2f} ms")
    print(f"Total System Latency: {total_time*1000:.2f} ms")
    print(f"Top-{top_k} Item IDs: {top_items['product_id'].tolist()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--user-id', type=int, required=True)
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args()
    run(args.user_id, args.top_k)
