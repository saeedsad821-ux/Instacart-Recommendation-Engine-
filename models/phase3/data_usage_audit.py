import os
import json
import pandas as pd
import numpy as np

def run_data_usage_audit(root_dir="."):
    print("=========================================================================")
    print("            PHASE 3 — FORENSIC DATA USAGE AUDIT                          ")
    print("=========================================================================")
    
    splits_dir = os.path.join(root_dir, "artifacts/splits")
    manifest_path = os.path.join(splits_dir, "split_manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    u_train = set(pd.read_parquet(os.path.join(splits_dir, "user_train.parquet"))['user_id'])
    u_val = set(pd.read_parquet(os.path.join(splits_dir, "user_validation.parquet"))['user_id'])
    u_test = set(pd.read_parquet(os.path.join(splits_dir, "user_test.parquet"))['user_id'])
    
    print(f"[INFO] Total Users in Manifest: {manifest['train_users'] + manifest['validation_users'] + manifest['test_users']:,}")
    print(f"       Train Users:      {len(u_train):,}")
    print(f"       Validation Users: {len(u_val):,}")
    print(f"       Test Users:       {len(u_test):,}")
    
    # Check prior rows belonging to Train vs Val vs Test
    prior_path = os.path.join(root_dir, "data/raw/order_products__prior.csv")
    if not os.path.exists(prior_path):
        prior_path = os.path.join(root_dir, "../order_products__prior.csv")
    orders_path = os.path.join(root_dir, "data/raw/orders.csv")
    if not os.path.exists(orders_path):
        orders_path = os.path.join(root_dir, "../orders.csv")
        
    orders_df = pd.read_csv(orders_path, usecols=['order_id', 'user_id', 'eval_set'])
    order_to_user = orders_df.set_index('order_id')['user_id'].to_dict()
    
    # Load prior csv in chunks to count rows per split
    train_prior_rows = 0
    val_prior_rows = 0
    test_prior_rows = 0
    total_prior_rows = 0
    for chunk in pd.read_csv(prior_path, chunksize=1000000, usecols=['order_id']):
        uids = chunk['order_id'].map(order_to_user)
        total_prior_rows += len(chunk)
        train_prior_rows += uids.isin(u_train).sum()
        val_prior_rows += uids.isin(u_val).sum()
        test_prior_rows += uids.isin(u_test).sum()
        
    print("\n[VERIFIED PRIOR ROW DISCREPANCY AUDIT]")
    print(f"Total Raw Prior Rows across all 206,209 users: {total_prior_rows:,}")
    print(f"Prior Rows belonging to Train (144,346 users): {train_prior_rows:,} ({(train_prior_rows/total_prior_rows)*100:.2f}%)")
    print(f"Prior Rows belonging to Val (30,931 users):    {val_prior_rows:,} ({(val_prior_rows/total_prior_rows)*100:.2f}%)")
    print(f"Prior Rows belonging to Test (30,932 users):   {test_prior_rows:,} ({(test_prior_rows/total_prior_rows)*100:.2f}%)")

    # Check candidate rows in processed data
    processed_path = os.path.join(root_dir, "data/processed/product_data.csv")
    total_cand_rows = 0
    train_cand_rows = 0
    val_cand_rows = 0
    test_cand_rows = 0
    for chunk in pd.read_csv(processed_path, chunksize=500000, usecols=['user_id']):
        total_cand_rows += len(chunk)
        train_cand_rows += chunk['user_id'].isin(u_train).sum()
        val_cand_rows += chunk['user_id'].isin(u_val).sum()
        test_cand_rows += chunk['user_id'].isin(u_test).sum()

    print("\n[VERIFIED CANDIDATE ROW AUDIT]")
    print(f"Total Static Candidate Rows in product_data.csv: {total_cand_rows:,} ({total_cand_rows/206209:.2f} static items/user)")
    print(f"Train Candidate Rows:      {train_cand_rows:,} ({train_cand_rows/len(u_train):.2f} items/user)")
    print(f"Validation Candidate Rows: {val_cand_rows:,} ({val_cand_rows/len(u_val):.2f} items/user)")
    print(f"Test Candidate Rows:       {test_cand_rows:,} ({test_cand_rows/len(u_test):.2f} items/user)")

    audit_summary = {
        "total_users": 206209,
        "train_users": len(u_train),
        "val_users": len(u_val),
        "test_users": len(u_test),
        "total_raw_prior_rows": int(total_prior_rows),
        "train_prior_rows": int(train_prior_rows),
        "val_prior_rows": int(val_prior_rows),
        "test_prior_rows": int(test_prior_rows),
        "total_static_candidate_rows": int(total_cand_rows),
        "train_static_candidate_rows": int(train_cand_rows),
        "val_static_candidate_rows": int(val_cand_rows),
        "test_static_candidate_rows": int(test_cand_rows),
        "avg_static_candidates_per_user": float(total_cand_rows / 206209),
        "dynamic_candidates_per_user": 100.0
    }
    
    out_dir = os.path.join(root_dir, "artifacts/reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase3_data_usage_summary.json")
    with open(out_path, "w") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"[SUCCESS] Phase 3 Data Usage audit summary saved to {out_path}")
    return audit_summary

if __name__ == "__main__":
    run_data_usage_audit()
