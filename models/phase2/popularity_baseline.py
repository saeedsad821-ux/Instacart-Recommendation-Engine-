import os
import time
import json
import numpy as np
import pandas as pd
from split_manager import SplitManager
from evaluation import Evaluator

class PopularityBaseline:
    """
    Phase 2A: Non-Trained Popularity & Historical Reorder Baseline.
    Ranks items for each user by:
      1. User-specific purchase count (from pre-target history)
      2. Product global reorder frequency & popularity (from prior orders)
      3. Global popularity padding for users with < K historical items.
    """
    def __init__(self, root_dir="../.."):
        self.root_dir = root_dir
        self.splits_dir = os.path.join(root_dir, "artifacts", "splits")
        self.metrics_dir = os.path.join(root_dir, "artifacts", "metrics")
        self.preds_dir = os.path.join(root_dir, "artifacts", "predictions")
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.preds_dir, exist_ok=True)

        self.evaluator = Evaluator(k=10)
        self.global_top_items = []
        self.item_global_score = {}

    def fit_global_popularity(self, prior_path=None):
        """
        Computes global product popularity and reorder rate from strictly prior orders.
        """
        t0 = time.time()
        if prior_path is None:
            prior_path = os.path.join(self.root_dir, "data/raw/order_products__prior.csv")

        print(f"[INFO] Computing global product popularity from {prior_path}...")
        prior_df = pd.read_csv(prior_path, usecols=['product_id', 'reordered'])
        stats = prior_df.groupby('product_id').agg(
            order_count=('reordered', 'count'),
            reorder_count=('reordered', 'sum')
        ).reset_index()
        stats['reorder_rate'] = stats['reorder_count'] / (stats['order_count'] + 1e-5)
        # Score combining log frequency and reorder probability
        stats['global_score'] = np.log1p(stats['order_count']) * (0.5 + 0.5 * stats['reorder_rate'])
        stats = stats.sort_values(by='global_score', ascending=False)

        self.global_top_items = stats['product_id'].tolist()[:100]
        self.item_global_score = dict(zip(stats['product_id'], stats['global_score']))
        print(f"[INFO] Global popularity computed in {time.time() - t0:.2f}s. Top item: {self.global_top_items[0]}")

    def generate_recommendations_for_users(self, product_data_df, user_ids_set):
        """
        Generates Top-10 recommendations for a set of users using candidate history in product_data.csv.
        """
        t0 = time.time()
        print(f"[INFO] Generating baseline recommendations for {len(user_ids_set):,} users...")

        # Filter product_data_df for target users
        sub_df = product_data_df[product_data_df['user_id'].isin(user_ids_set)].copy()

        # Compute user-item history score: number of times ordered in history * global popularity boost
        def calc_hist_count(s):
            return sum(1 for x in str(s).split() if x == '1')

        # We vectorize by counting '1's in is_ordered_history string
        sub_df['hist_orders'] = sub_df['is_ordered_history'].apply(calc_hist_count)
        sub_df['global_score'] = sub_df['product_id'].map(lambda pid: self.item_global_score.get(pid, 0.0))
        sub_df['score'] = sub_df['hist_orders'] * 10.0 + sub_df['global_score']

        # Sort per user and take top 10
        sub_df = sub_df.sort_values(by=['user_id', 'score'], ascending=[True, False])
        top_k_per_user = sub_df.groupby('user_id').head(10)

        user_recs = top_k_per_user.groupby('user_id')['product_id'].apply(list).to_dict()

        # Pad with global top items if fewer than 10
        for uid in user_ids_set:
            recs = user_recs.get(uid, [])
            if len(recs) < 10:
                rec_set = set(recs)
                for gid in self.global_top_items:
                    if gid not in rec_set:
                        recs.append(gid)
                        rec_set.add(gid)
                        if len(recs) == 10:
                            break
                user_recs[uid] = recs

        inf_time = time.time() - t0
        print(f"[INFO] Recommendations generated in {inf_time:.2f}s ({inf_time/len(user_ids_set)*1000:.2f} ms/user).")
        return user_recs, inf_time

    def run_and_evaluate(self):
        t0 = time.time()
        print("=========================================================================")
        print(" PHASE 2A: POPULARITY BASELINE EVALUATION                                ")
        print("=========================================================================")

        # 1. Fit global popularity
        self.fit_global_popularity()
        train_time = time.time() - t0

        # 2. Load test split users
        sm = SplitManager(root_dir=self.root_dir)
        splits = sm.load_splits()
        test_users = set(splits['user_test']['user_id'])
        print(f"[INFO] Evaluating Popularity Baseline on {len(test_users):,} Unseen Test Users...")

        # 3. Load product_data.csv for evaluation users
        t_load = time.time()
        print("[INFO] Loading product_data.csv for test users...")
        # To save memory and speed up, we read chunk by chunk or filter
        # Since product_data is 7GB, let's stream it or use pyarrow if available, or read in chunks
        chunk_size = 1000000
        test_sub_dfs = []
        for chunk in pd.read_csv(os.path.join(self.root_dir, "data/processed/product_data.csv"), chunksize=chunk_size, keep_default_na=False):
            matched = chunk[chunk['user_id'].isin(test_users)]
            if len(matched) > 0:
                test_sub_dfs.append(matched)
        test_prod_df = pd.concat(test_sub_dfs, ignore_index=True)
        print(f"[INFO] Filtered {len(test_prod_df):,} candidate rows for test users in {time.time() - t_load:.2f}s.")

        # 4. Generate recommendations
        preds, inf_time = self.generate_recommendations_for_users(test_prod_df, test_users)

        # 5. Extract ground truth
        gt = self.evaluator.extract_ground_truth_from_product_data(test_prod_df)

        # 6. Evaluate metrics
        metrics = self.evaluator.evaluate_all(
            predictions_dict=preds,
            ground_truth_dict=gt,
            model_name="Popularity_Baseline",
            time_info={
                "train_time_sec": train_time,
                "inference_time_sec": inf_time,
                "inference_ms_per_user": (inf_time / len(test_users)) * 1000.0,
                "total_test_users": len(test_users)
            }
        )

        metrics_path = os.path.join(self.metrics_dir, "popularity_baseline.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print("=========================================================================")
        print(" [SUCCESS] POPULARITY BASELINE RESULTS (UNSEEN TEST USERS):              ")
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
    pb = PopularityBaseline(root_dir=".")
    pb.run_and_evaluate()
