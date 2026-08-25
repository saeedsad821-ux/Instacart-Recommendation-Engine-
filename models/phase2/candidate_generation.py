import os
import time
import json
import numpy as np
import pandas as pd
from split_manager import SplitManager

class CandidateGenerator:
    """
    Phase 2 Candidate Generation Module.
    Generates a leakage-safe candidate set for each user from:
      1. Historically purchased products (from pre-target orders 1...N-1)
      2. Global/Aisle popularity fallback for users with short histories
    Evaluates candidate recall and latency before ranking.
    """
    def __init__(self, root_dir="../..", max_candidates=100):
        self.root_dir = root_dir
        self.max_candidates = max_candidates
        self.metrics_dir = os.path.join(root_dir, "artifacts", "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.global_top_items = []

    def fit_global_popularity(self, prior_path=None):
        if prior_path is None:
            prior_path = os.path.join(self.root_dir, "data/raw/order_products__prior.csv")
        prior_df = pd.read_csv(prior_path, usecols=['product_id'])
        counts = prior_df['product_id'].value_counts()
        self.global_top_items = counts.index[:self.max_candidates].tolist()

    def generate_candidates(self, product_data_df, user_ids_set):
        """
        Generates candidate product ID lists for each user in user_ids_set.
        Returns dict: {user_id: [prod_1, prod_2, ...]}
        """
        t0 = time.time()
        sub_df = product_data_df[product_data_df['user_id'].isin(user_ids_set)][['user_id', 'product_id']].copy()
        
        user_cands = sub_df.groupby('user_id')['product_id'].apply(list).to_dict()

        # Pad users who have fewer than max_candidates
        for uid in user_ids_set:
            cands = user_cands.get(uid, [])
            if len(cands) < self.max_candidates:
                c_set = set(cands)
                for gid in self.global_top_items:
                    if gid not in c_set:
                        cands.append(gid)
                        c_set.add(gid)
                        if len(cands) == self.max_candidates:
                            break
                user_cands[uid] = cands
            elif len(cands) > self.max_candidates:
                user_cands[uid] = cands[:self.max_candidates]

        inf_time = time.time() - t0
        return user_cands, inf_time

    def evaluate_candidate_recall(self):
        print("=========================================================================")
        print(" PHASE 2: CANDIDATE GENERATION VALIDATION & RECALL EVALUATION            ")
        print("=========================================================================")
        t0 = time.time()
        self.fit_global_popularity()

        sm = SplitManager(root_dir=self.root_dir)
        splits = sm.load_splits()
        test_users = set(splits['user_test']['user_id'])

        # Load product_data.csv for test users
        print("[INFO] Loading product_data.csv for candidate recall evaluation...")
        test_sub_dfs = []
        for chunk in pd.read_csv(os.path.join(self.root_dir, "data/processed/product_data.csv"), chunksize=1000000, keep_default_na=False):
            matched = chunk[chunk['user_id'].isin(test_users)]
            if len(matched) > 0:
                test_sub_dfs.append(matched)
        test_prod_df = pd.concat(test_sub_dfs, ignore_index=True)

        cands_dict, gen_time = self.generate_candidates(test_prod_df, test_users)

        # Extract ground truth target items
        pos_df = test_prod_df[test_prod_df['label'] == 1][['user_id', 'product_id']]
        gt_dict = pos_df.groupby('user_id')['product_id'].apply(set).to_dict()

        # Measure recall at 50 and 100
        recalls_50 = []
        recalls_100 = []
        cand_counts = []
        n_eval = 0

        for uid, target_set in gt_dict.items():
            if not target_set:
                continue
            c_list = cands_dict.get(uid, [])
            cand_counts.append(len(c_list))

            c_set_50 = set(c_list[:50])
            c_set_100 = set(c_list[:100])

            r50 = len(target_set.intersection(c_set_50)) / len(target_set)
            r100 = len(target_set.intersection(c_set_100)) / len(target_set)

            recalls_50.append(r50)
            recalls_100.append(r100)
            n_eval += 1

        res = {
            "eval_user_count": n_eval,
            "mean_candidate_count": float(np.mean(cand_counts)),
            "candidate_recall_at_50": float(np.mean(recalls_50)),
            "candidate_recall_at_100": float(np.mean(recalls_100)),
            "generation_time_sec": gen_time,
            "ms_per_user": (gen_time / len(test_users)) * 1000.0
        }

        out_path = os.path.join(self.metrics_dir, "candidate_generation_recall.json")
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)

        print("=========================================================================")
        print(" [SUCCESS] CANDIDATE GENERATION RECALL (UNSEEN TEST USERS):              ")
        print(f"   Candidate Recall@50:  {res['candidate_recall_at_50']:.6f}")
        print(f"   Candidate Recall@100: {res['candidate_recall_at_100']:.6f}")
        print(f"   Avg Candidates/User:  {res['mean_candidate_count']:.1f}")
        print(f"   Generation Time:      {res['generation_time_sec']:.2f}s ({res['ms_per_user']:.2f} ms/user)")
        print(f" Saved metrics to: {out_path}")
        print("=========================================================================")
        return res

if __name__ == '__main__':
    cg = CandidateGenerator(root_dir=".")
    cg.evaluate_candidate_recall()
