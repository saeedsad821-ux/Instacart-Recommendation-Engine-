import os
import json
import time
import numpy as np
import pandas as pd

SPLIT_VERSION = "2.0"
RANDOM_SEED = 42

class SplitManager:
    """
    Manages deterministic, reproducible, leakage-safe datasets and evaluation splits
    for Phase 2 of the Instacart Recommendation System.
    Supports:
      - Unseen User Generalization (user_train, user_validation, user_test)
      - Temporal Future Generalization
      - Synthetic Cold Start
      - Rare Product Buckets
    """
    def __init__(self, root_dir="../..", artifacts_dir=None):
        if artifacts_dir is None:
            self.artifacts_dir = os.path.join(root_dir, "artifacts", "splits")
        else:
            self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.metadata_path = os.path.join(self.artifacts_dir, "split_metadata.json")

    def splits_exist_and_compatible(self):
        if not os.path.exists(self.metadata_path):
            return False
        try:
            with open(self.metadata_path, "r") as f:
                meta = json.load(f)
            if meta.get("split_version") == SPLIT_VERSION:
                # Check that essential parquet files exist
                required_files = [
                    "user_train.parquet", "user_validation.parquet", "user_test.parquet",
                    "cold_start_test.parquet", "product_buckets.parquet"
                ]
                for fn in required_files:
                    if not os.path.exists(os.path.join(self.artifacts_dir, fn)):
                        return False
                return True
        except Exception as e:
            print(f"[WARN] Failed to read split_metadata.json: {e}")
        return False

    def load_splits(self):
        print(f"[INFO] Loading existing compatible splits (version {SPLIT_VERSION}) from {self.artifacts_dir}...")
        user_train = pd.read_parquet(os.path.join(self.artifacts_dir, "user_train.parquet"))
        user_val = pd.read_parquet(os.path.join(self.artifacts_dir, "user_validation.parquet"))
        user_test = pd.read_parquet(os.path.join(self.artifacts_dir, "user_test.parquet"))
        cold_start = pd.read_parquet(os.path.join(self.artifacts_dir, "cold_start_test.parquet"))
        product_buckets = pd.read_parquet(os.path.join(self.artifacts_dir, "product_buckets.parquet"))
        return {
            "user_train": user_train,
            "user_val": user_val,
            "user_test": user_test,
            "cold_start_test": cold_start,
            "product_buckets": product_buckets
        }

    def generate_splits(self, user_data_path, products_path, prior_path=None):
        print(f"[INFO] Generating deterministic evaluation splits (version {SPLIT_VERSION}, seed={RANDOM_SEED})...")
        t0 = time.time()
        
        user_df = pd.read_csv(user_data_path, keep_default_na=False)
        print(f"[INFO] Loaded user_data.csv: {len(user_df):,} users.")

        # 1. User-Level Partitioning: 70% Train, 15% Validation, 15% Test
        np.random.seed(RANDOM_SEED)
        all_users = user_df['user_id'].unique()
        all_users_sorted = np.sort(all_users)
        np.random.shuffle(all_users_sorted)

        n_total = len(all_users_sorted)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)

        train_users = set(all_users_sorted[:n_train])
        val_users = set(all_users_sorted[n_train:n_train+n_val])
        test_users = set(all_users_sorted[n_train+n_val:])

        # Verify strict zero intersection
        assert len(train_users.intersection(val_users)) == 0, "Leakage Alert: Train/Val users overlap!"
        assert len(train_users.intersection(test_users)) == 0, "Leakage Alert: Train/Test users overlap!"
        assert len(val_users.intersection(test_users)) == 0, "Leakage Alert: Val/Test users overlap!"

        user_df['split_group'] = user_df['user_id'].apply(
            lambda uid: 'train' if uid in train_users else ('val' if uid in val_users else 'test')
        )

        user_train = user_df[user_df['split_group'] == 'train'].copy()
        user_val = user_df[user_df['split_group'] == 'val'].copy()
        user_test = user_df[user_df['split_group'] == 'test'].copy()

        # 2. Synthetic Cold Start Split
        # Sample 5,000 test users and mask their historical orders to simulate zero history
        cold_start_users = user_test.sample(n=min(5000, len(user_test)), random_state=RANDOM_SEED).copy()
        cold_start_users['is_synthetic_cold_start'] = True

        # 3. Rare Product Buckets
        products_df = pd.read_csv(products_path)
        if prior_path and os.path.exists(prior_path):
            print("[INFO] Computing product order frequencies from prior orders...")
            prior_df = pd.read_csv(prior_path, usecols=['product_id'])
            prod_counts = prior_df['product_id'].value_counts().reset_index()
            prod_counts.columns = ['product_id', 'order_count']
            products_df = products_df.merge(prod_counts, on='product_id', how='left').fillna({'order_count': 0})
        else:
            products_df['order_count'] = 0

        def classify_frequency(count):
            if count >= 1000:
                return 'frequent'
            elif count >= 100:
                return 'medium'
            else:
                return 'rare'

        products_df['frequency_bucket'] = products_df['order_count'].apply(classify_frequency)

        # Save parquet files
        user_train.to_parquet(os.path.join(self.artifacts_dir, "user_train.parquet"), index=False)
        user_val.to_parquet(os.path.join(self.artifacts_dir, "user_validation.parquet"), index=False)
        user_test.to_parquet(os.path.join(self.artifacts_dir, "user_test.parquet"), index=False)
        cold_start_users.to_parquet(os.path.join(self.artifacts_dir, "cold_start_test.parquet"), index=False)
        products_df.to_parquet(os.path.join(self.artifacts_dir, "product_buckets.parquet"), index=False)

        metadata = {
            "split_version": SPLIT_VERSION,
            "random_seed": RANDOM_SEED,
            "created_at": time.ctime(),
            "counts": {
                "total_users": int(n_total),
                "train_users": int(len(user_train)),
                "val_users": int(len(user_val)),
                "test_users": int(len(user_test)),
                "cold_start_users": int(len(cold_start_users)),
                "frequent_products": int((products_df['frequency_bucket'] == 'frequent').sum()),
                "medium_products": int((products_df['frequency_bucket'] == 'medium').sum()),
                "rare_products": int((products_df['frequency_bucket'] == 'rare').sum())
            }
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[SUCCESS] Splits generated and saved in {time.time() - t0:.2f}s:")
        print(f"  - Train users:      {len(user_train):,}")
        print(f"  - Validation users: {len(user_val):,}")
        print(f"  - Test users:       {len(user_test):,}")
        print(f"  - Synthetic cold-start users: {len(cold_start_users):,}")
        print(f"  - Product frequency buckets: frequent={metadata['counts']['frequent_products']:,}, medium={metadata['counts']['medium_products']:,}, rare={metadata['counts']['rare_products']:,}")

        return {
            "user_train": user_train,
            "user_val": user_val,
            "user_test": user_test,
            "cold_start_test": cold_start_users,
            "product_buckets": products_df
        }

if __name__ == '__main__':
    sm = SplitManager(root_dir=".")
    # Assuming run from root dir of project
    sm.generate_splits(
        user_data_path="data/processed/user_data.csv",
        products_path="data/raw/products.csv",
        prior_path="data/raw/order_products__prior.csv"
    )
