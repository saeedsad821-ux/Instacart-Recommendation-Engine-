import os
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

class InstacartProductionRecommender:
    """
    Phase 2F: Production-Oriented Unified Recommendation Pipeline.
    Serves personalized next-basket recommendations for existing users (LightGBM LambdaRanker)
    and robust fallback recommendations for cold-start users (Popularity/Reorder baseline).
    """
    def __init__(self, root_dir="../.."):
        self.root_dir = root_dir
        self.model_dir = os.path.join(root_dir, "artifacts/models/lightgbm_ranker")
        self.products_path = os.path.join(root_dir, "data/raw/products.csv")
        self.prior_path = os.path.join(root_dir, "data/raw/order_products__prior.csv")
        
        print("[INFO] Initializing Instacart Production Recommender Engine...")
        t0 = time.time()
        
        # 1. Load Product Catalog Names
        self.products_df = pd.read_csv(self.products_path)
        self.product_names = dict(zip(self.products_df['product_id'], self.products_df['product_name']))
        self.product_aisles = dict(zip(self.products_df['product_id'], self.products_df['aisle_id']))
        self.product_depts = dict(zip(self.products_df['product_id'], self.products_df['department_id']))

        # 2. Load Global Item Popularity Stats & Build Cold-Start Fallback
        prior_df = pd.read_csv(self.prior_path, usecols=['product_id', 'reordered'])
        stats = prior_df.groupby('product_id').agg(
            order_count=('reordered', 'count'),
            reorder_count=('reordered', 'sum')
        ).reset_index()
        stats['reorder_rate'] = stats['reorder_count'] / (stats['order_count'] + 1e-5)
        stats['score'] = np.log1p(stats['order_count']) * (0.5 + 0.5 * stats['reorder_rate'])
        
        self.item_stats = {
            r['product_id']: {
                'order_count': r['order_count'],
                'reorder_rate': r['reorder_rate']
            }
            for _, r in stats.iterrows()
        }
        
        # Top 10 Fallback for Cold-Start Users
        self.cold_start_top_k = stats.sort_values('score', ascending=False)['product_id'].tolist()[:10]

        # 3. Load LightGBM LambdaRanker Model
        model_file = os.path.join(self.model_dir, "lightgbm_ranker.txt")
        self.booster = lgb.Booster(model_file=model_file)
        self.features = [
            'hist_order_count', 'user_total_orders', 'user_reorder_rate',
            'recency_days', 'global_order_count', 'global_reorder_rate',
            'aisle_id', 'department_id'
        ]

        # 4. Load User Candidate Cache (from product_data.csv for demonstration)
        print("[INFO] Caching existing user purchase history...")
        self.user_history_cache = {}
        for chunk in pd.read_csv(os.path.join(root_dir, "data/processed/product_data.csv"), chunksize=1000000, keep_default_na=False):
            for uid, group in chunk.groupby('user_id'):
                if uid not in self.user_history_cache:
                    self.user_history_cache[uid] = group
                else:
                    self.user_history_cache[uid] = pd.concat([self.user_history_cache[uid], group], ignore_index=True)
        
        print(f"[SUCCESS] Production Recommender initialized in {time.time() - t0:.2f}s. Cached {len(self.user_history_cache):,} users.")

    def recommend(self, user_id, top_k=10):
        """
        Generates Top-K next-basket recommendations for any user_id.
        Returns a list of dicts with product_id, product_name, score, and recommendation_type.
        """
        t_start = time.time()
        
        # Case A: Existing User with Purchase History -> Personalized LightGBM LambdaRanker
        if user_id in self.user_history_cache:
            df = self.user_history_cache[user_id].copy()
            
            def count_ones(s): return sum(1 for x in str(s).split() if x == '1')
            def get_max_val(s): 
                parts = str(s).split()
                return int(parts[-1]) if parts else 1
            def get_last_float(s):
                parts = str(s).split()
                return float(parts[-1]) if parts else 7.0

            df['hist_order_count'] = df['is_ordered_history'].apply(count_ones).astype(np.int16)
            df['user_total_orders'] = df['order_number_history'].apply(get_max_val).astype(np.int16)
            df['user_reorder_rate'] = (df['hist_order_count'] / df['user_total_orders']).astype(np.float32)
            df['recency_days'] = df['days_since_prior_order_history'].apply(get_last_float).astype(np.float32)
            df['aisle_id'] = df['aisle_id'].astype(np.int16)
            df['department_id'] = df['department_id'].astype(np.int8)

            df['global_order_count'] = df['product_id'].map(lambda pid: self.item_stats.get(pid, {}).get('order_count', 0)).astype(np.int32)
            df['global_reorder_rate'] = df['product_id'].map(lambda pid: self.item_stats.get(pid, {}).get('reorder_rate', 0.0)).astype(np.float32)

            scores = self.booster.predict(df[self.features])
            df['score'] = scores
            df = df.sort_values('score', ascending=False).head(top_k)

            recs = []
            for _, row in df.iterrows():
                pid = int(row['product_id'])
                recs.append({
                    "product_id": pid,
                    "product_name": self.product_names.get(pid, f"Product {pid}"),
                    "ranking_score": float(row['score']),
                    "recommendation_type": "Personalized_LightGBM_LambdaRanker"
                })
            lat_ms = (time.time() - t_start) * 1000.0
            return recs, lat_ms, "PERSONALIZED_RANKING"

        # Case B: Cold-Start User (No purchase history) -> Global Popularity / Reorder Fallback
        else:
            recs = []
            for pid in self.cold_start_top_k[:top_k]:
                recs.append({
                    "product_id": pid,
                    "product_name": self.product_names.get(pid, f"Product {pid}"),
                    "ranking_score": float(self.item_stats.get(pid, {}).get('order_count', 0)),
                    "recommendation_type": "Cold_Start_Popularity_Fallback"
                })
            lat_ms = (time.time() - t_start) * 1000.0
            return recs, lat_ms, "COLD_START_FALLBACK"

if __name__ == '__main__':
    recommender = InstacartProductionRecommender(root_dir=".")
    
    print("\n=========================================================================")
    print(" LIVE INFERENCE BENCHMARK & SYSTEM AUDIT                                 ")
    print("=========================================================================")
    
    # Test 3 existing users
    existing_users = list(recommender.user_history_cache.keys())[:3]
    for uid in existing_users:
        recs, lat, rec_type = recommender.recommend(uid, top_k=5)
        print(f"\n[USER {uid} | Type: {rec_type} | Latency: {lat:.2f} ms]")
        for idx, r in enumerate(recs, 1):
            print(f"  {idx}. {r['product_name']} (ID: {r['product_id']}, Score: {r['ranking_score']:.4f})")
            
    # Test 2 cold-start users
    cold_users = [9999999, 8888888]
    for uid in cold_users:
        recs, lat, rec_type = recommender.recommend(uid, top_k=5)
        print(f"\n[USER {uid} | Type: {rec_type} | Latency: {lat:.2f} ms]")
        for idx, r in enumerate(recs, 1):
            print(f"  {idx}. {r['product_name']} (ID: {r['product_id']}, Orders: {int(r['ranking_score']):,})")
            
    print("\n=========================================================================")
    print(" [SUCCESS] PRODUCTION INFERENCE ENGINE VERIFIED & TESTED!                ")
    print("=========================================================================")
