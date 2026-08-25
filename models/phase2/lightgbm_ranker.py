import os
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from split_manager import SplitManager, SPLIT_VERSION
from checkpoint_manager import CheckpointManager
from evaluation import Evaluator
from candidate_generation import CandidateGenerator

class LightGBMRanker:
    """
    Phase 2B: Professional LightGBM LambdaRanker for Instacart Next-Basket Recommendation.
    Uses strictly leakage-safe point-in-time features, lambdaRank objective, group-wise evaluation,
    and Mandatory Rule 4 / 9 Checkpoint verification.
    """
    def __init__(self, root_dir="../.."):
        self.root_dir = root_dir
        self.models_dir = os.path.join(root_dir, "artifacts", "models")
        self.metrics_dir = os.path.join(root_dir, "artifacts", "metrics")
        self.preds_dir = os.path.join(root_dir, "artifacts", "predictions")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.preds_dir, exist_ok=True)

        self.evaluator = Evaluator(k=10)
        self.cm = CheckpointManager(root_dir=root_dir)
        self.model_name = "lightgbm_ranker"
        self.config = {
            "dataset_version": "1.0",
            "preprocessing_version": "option_b_v1",
            "split_version": SPLIT_VERSION,
            "model_version": "2.0-lambdarank",
            "hyperparameters": {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [10],
                "boosting_type": "gbdt",
                "n_estimators": 300,
                "learning_rate": 0.08,
                "num_leaves": 63,
                "min_child_samples": 50,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": -1
            }
        }

    def prepare_features(self, df, item_stats):
        """
        Engineers point-in-time features from candidate sequences in df.
        All features depend strictly on orders 1...N-1.
        """
        print(f"[INFO] Engineering features for {len(df):,} candidate rows...")
        t0 = time.time()

        # Vectorized feature computation
        # 1. Historical purchase count
        def count_ones(s):
            return sum(1 for x in str(s).split() if x == '1')
        
        df['hist_order_count'] = df['is_ordered_history'].apply(count_ones).astype(np.int16)

        # 2. Total user history length from order_number_history
        def get_max_val(s):
            parts = str(s).split()
            return int(parts[-1]) if parts else 1

        df['user_total_orders'] = df['order_number_history'].apply(get_max_val).astype(np.int16)
        df['user_reorder_rate'] = (df['hist_order_count'] / df['user_total_orders']).astype(np.float32)

        # 3. Recency (days since prior order)
        def get_last_float(s):
            parts = str(s).split()
            return float(parts[-1]) if parts else 7.0

        df['recency_days'] = df['days_since_prior_order_history'].apply(get_last_float).astype(np.float32)

        # 4. Catalog & global popularity features
        df['global_order_count'] = df['product_id'].map(lambda pid: item_stats.get(pid, {}).get('order_count', 0)).astype(np.int32)
        df['global_reorder_rate'] = df['product_id'].map(lambda pid: item_stats.get(pid, {}).get('reorder_rate', 0.0)).astype(np.float32)
        df['aisle_id'] = df['aisle_id'].astype(np.int16)
        df['department_id'] = df['department_id'].astype(np.int8)

        features = [
            'hist_order_count', 'user_total_orders', 'user_reorder_rate',
            'recency_days', 'global_order_count', 'global_reorder_rate',
            'aisle_id', 'department_id'
        ]

        print(f"[INFO] Feature engineering completed in {time.time() - t0:.2f}s.")
        return df[features], df['label'].astype(np.int8), df['user_id'], features

    def get_group_sizes(self, user_ids_series):
        """
        Computes group sizes for LightGBM LambdaRank (number of candidates per user in sorted order).
        """
        return user_ids_series.value_counts(sort=False)[user_ids_series.unique()].values

    def fit_global_item_stats(self, prior_path=None):
        if prior_path is None:
            prior_path = os.path.join(self.root_dir, "data/raw/order_products__prior.csv")
        prior_df = pd.read_csv(prior_path, usecols=['product_id', 'reordered'])
        stats = prior_df.groupby('product_id').agg(
            order_count=('reordered', 'count'),
            reorder_count=('reordered', 'sum')
        ).reset_index()
        stats['reorder_rate'] = stats['reorder_count'] / (stats['order_count'] + 1e-5)
        
        item_stats = {}
        for _, row in stats.iterrows():
            item_stats[row['product_id']] = {
                'order_count': row['order_count'],
                'reorder_rate': row['reorder_rate']
            }
        return item_stats

    def train_or_load(self):
        print("=========================================================================")
        print(" PHASE 2B: LIGHTGBM LAMBDARANKER TRAINING & CHECKPOINT VERIFICATION      ")
        print("=========================================================================")

        # 1. Mandatory Rule 4 & 9: Check existing checkpoint
        can_reuse, model_dir, existing_meta = self.cm.check_checkpoint(self.model_name, self.config)
        if can_reuse and existing_meta:
            booster = self.cm.load_lightgbm_model(model_dir, existing_meta["model_file"])
            return booster, existing_meta, 0.0

        print("[INFO] No compatible checkpoint found. Starting efficient training...")
        t0_train = time.time()

        # 2. Load splits & global item stats
        sm = SplitManager(root_dir=self.root_dir)
        splits = sm.load_splits()
        train_users = set(splits['user_train']['user_id'])
        val_users = set(splits['user_val']['user_id'])

        item_stats = self.fit_global_item_stats()

        # 3. Load product_data.csv for train and val users
        print("[INFO] Loading candidate rows for Train and Validation sets...")
        train_dfs = []
        val_dfs = []
        for chunk in pd.read_csv(os.path.join(self.root_dir, "data/processed/product_data.csv"), chunksize=1000000, keep_default_na=False):
            t_match = chunk[chunk['user_id'].isin(train_users)]
            if len(t_match) > 0:
                train_dfs.append(t_match)
            v_match = chunk[chunk['user_id'].isin(val_users)]
            if len(v_match) > 0:
                val_dfs.append(v_match)

        train_prod_df = pd.concat(train_dfs, ignore_index=True)
        val_prod_df = pd.concat(val_dfs, ignore_index=True)

        # Sort strictly by user_id so groups are contiguous for LambdaRank
        train_prod_df = train_prod_df.sort_values('user_id').reset_index(drop=True)
        val_prod_df = val_prod_df.sort_values('user_id').reset_index(drop=True)

        X_train, y_train, u_train, feat_names = self.prepare_features(train_prod_df, item_stats)
        X_val, y_val, u_val, _ = self.prepare_features(val_prod_df, item_stats)

        group_train = self.get_group_sizes(u_train)
        group_val = self.get_group_sizes(u_val)

        print(f"[INFO] Train shape: {X_train.shape}, Val shape: {X_val.shape}")
        print(f"[INFO] Train groups (users): {len(group_train):,}, Val groups (users): {len(group_val):,}")

        # 4. Train LightGBM LambdaRanker
        params = self.config["hyperparameters"].copy()
        n_est = params.pop("n_estimators")

        train_ds = lgb.Dataset(X_train, label=y_train, group=group_train, feature_name=feat_names, free_raw_data=False)
        val_ds = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_ds, free_raw_data=False)

        print("[INFO] Training LightGBM LambdaRanker with early stopping (30 rounds)...")
        booster = lgb.train(
            params=params,
            train_set=train_ds,
            num_boost_round=n_est,
            valid_sets=[train_ds, val_ds],
            valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=True), lgb.log_evaluation(period=20)]
        )

        train_time = time.time() - t0_train

        # 5. Save Checkpoint & Metadata
        feat_imp = dict(zip(feat_names, booster.feature_importance(importance_type='gain').tolist()))
        metadata = self.config.copy()
        metadata.update({
            "train_time_sec": train_time,
            "best_iteration": booster.best_iteration,
            "feature_importance_gain": feat_imp
        })

        self.cm.save_checkpoint(self.model_name, booster, metadata, is_lightgbm=True)
        return booster, metadata, train_time

    def evaluate_test_set(self, booster, train_time):
        t0 = time.time()
        print("=========================================================================")
        print(" PHASE 2B: LIGHTGBM LAMBDARANKER TEST SET EVALUATION                     ")
        print("=========================================================================")

        sm = SplitManager(root_dir=self.root_dir)
        splits = sm.load_splits()
        test_users = set(splits['user_test']['user_id'])

        item_stats = self.fit_global_item_stats()

        # Load test candidate rows
        test_dfs = []
        for chunk in pd.read_csv(os.path.join(self.root_dir, "data/processed/product_data.csv"), chunksize=1000000, keep_default_na=False):
            matched = chunk[chunk['user_id'].isin(test_users)]
            if len(matched) > 0:
                test_dfs.append(matched)
        test_prod_df = pd.concat(test_dfs, ignore_index=True)
        test_prod_df = test_prod_df.sort_values('user_id').reset_index(drop=True)

        X_test, _, _, _ = self.prepare_features(test_prod_df, item_stats)

        # Score candidates
        t_inf_start = time.time()
        scores = booster.predict(X_test)
        test_prod_df['lgb_score'] = scores
        inf_time = time.time() - t_inf_start

        # Rank Top-10 per user
        test_prod_df = test_prod_df.sort_values(by=['user_id', 'lgb_score'], ascending=[True, False])
        top_k_per_user = test_prod_df.groupby('user_id').head(10)
        preds_dict = top_k_per_user.groupby('user_id')['product_id'].apply(list).to_dict()

        # Extract ground truth
        gt_dict = self.evaluator.extract_ground_truth_from_product_data(test_prod_df)

        # Evaluate metrics
        metrics = self.evaluator.evaluate_all(
            predictions_dict=preds_dict,
            ground_truth_dict=gt_dict,
            model_name="LightGBM_LambdaRanker",
            time_info={
                "train_time_sec": train_time,
                "inference_time_sec": inf_time,
                "inference_ms_per_user": (inf_time / len(test_users)) * 1000.0,
                "total_test_users": len(test_users)
            }
        )

        metrics_path = os.path.join(self.metrics_dir, "lightgbm_ranker.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print("=========================================================================")
        print(" [SUCCESS] LIGHTGBM LAMBDARANKER RESULTS (UNSEEN TEST USERS):            ")
        print(f"   Precision@10:     {metrics['precision_at_k']:.6f}")
        print(f"   Recall@10:        {metrics['recall_at_k']:.6f}")
        print(f"   F1@10:            {metrics['f1_at_k']:.6f}")
        print(f"   NDCG@10:          {metrics['ndcg_at_k']:.6f}")
        print(f"   Hit Rate@10:      {metrics['hit_rate_at_k']:.6f}")
        print(f"   Catalog Coverage: {metrics['catalog_coverage']:.6f}")
        print(f"   Inference Time:   {metrics['inference_time_sec']:.2f}s ({metrics['inference_ms_per_user']:.2f} ms/user)")
        print(f" Saved metrics to: {metrics_path}")
        print("=========================================================================")

        return metrics

if __name__ == '__main__':
    lgb_ranker = LightGBMRanker(root_dir=".")
    booster, meta, t_train = lgb_ranker.train_or_load()
    lgb_ranker.evaluate_test_set(booster, t_train)
