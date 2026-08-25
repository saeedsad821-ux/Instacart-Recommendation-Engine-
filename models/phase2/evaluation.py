import os
import json
import time
import numpy as np
import pandas as pd

class Evaluator:
    """
    Standardized Top-K Ranking & Generalization Evaluator for Phase 2.
    Computes: Precision@K, Recall@K, F1@K, NDCG@K, HitRate@K, and Catalog Coverage.
    """
    def __init__(self, k=10, catalog_size=49688):
        self.k = k
        self.catalog_size = catalog_size

    def evaluate_user(self, target_set, rec_list):
        """
        Evaluates ranking metrics for a single user.
        target_set: set of true product IDs in target basket
        rec_list: ordered list of recommended product IDs (up to K)
        """
        if not target_set:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "ndcg": 0.0,
                "hit_rate": 0.0
            }

        k_recs = rec_list[:self.k]
        hits = 0
        dcg = 0.0

        for idx, item in enumerate(k_recs):
            if item in target_set:
                hits += 1
                dcg += 1.0 / np.log2(idx + 2)  # 1-indexed -> log2(idx + 2)

        precision = hits / self.k
        recall = hits / len(target_set)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        hit_rate = 1.0 if hits > 0 else 0.0

        # IDCG
        ideal_hits = min(self.k, len(target_set))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ndcg": ndcg,
            "hit_rate": hit_rate
        }

    def evaluate_all(self, predictions_dict, ground_truth_dict, model_name="model", time_info=None):
        """
        Evaluates predictions across all users in ground_truth_dict.
        predictions_dict: {user_id: [prod_1, prod_2, ...]}
        ground_truth_dict: {user_id: set([true_1, true_2, ...])}
        time_info: dict containing 'train_time', 'inference_time', etc.
        """
        t0 = time.time()
        precisions = []
        recalls = []
        f1s = []
        ndcgs = []
        hit_rates = []
        all_recommended_items = set()

        n_eval = 0
        for uid, target_set in ground_truth_dict.items():
            if not target_set:
                continue
            rec_list = predictions_dict.get(uid, [])
            all_recommended_items.update(rec_list[:self.k])

            res = self.evaluate_user(target_set, rec_list)
            precisions.append(res["precision"])
            recalls.append(res["recall"])
            f1s.append(res["f1"])
            ndcgs.append(res["ndcg"])
            hit_rates.append(res["hit_rate"])
            n_eval += 1

        p_mean = float(np.mean(precisions)) if precisions else 0.0
        r_mean = float(np.mean(recalls)) if recalls else 0.0
        f_mean = float(np.mean(f1s)) if f1s else 0.0
        n_mean = float(np.mean(ndcgs)) if ndcgs else 0.0
        h_mean = float(np.mean(hit_rates)) if hit_rates else 0.0
        coverage = len(all_recommended_items) / float(self.catalog_size)

        metrics = {
            "model": model_name,
            "k": self.k,
            "eval_user_count": n_eval,
            "precision_at_k": p_mean,
            "recall_at_k": r_mean,
            "f1_at_k": f_mean,
            "ndcg_at_k": n_mean,
            "hit_rate_at_k": h_mean,
            "catalog_coverage": coverage,
            "eval_computation_time": time.time() - t0
        }
        if time_info:
            metrics.update(time_info)

        return metrics

    def extract_ground_truth_from_product_data(self, product_data_df):
        """
        Extracts ground truth target sets {user_id: set(product_ids)} for items with label == 1.
        """
        pos_df = product_data_df[product_data_df['label'] == 1][['user_id', 'product_id']]
        gt = pos_df.groupby('user_id')['product_id'].apply(set).to_dict()
        return gt

if __name__ == '__main__':
    ev = Evaluator(k=10)
    # Simple unit test
    gt = {101: {1, 2, 3}, 102: {5, 6}}
    preds = {101: [1, 4, 3, 7, 8, 9, 10, 11, 12, 13], 102: [99, 98, 97, 5, 1, 2, 3, 4, 7, 8]}
    res = ev.evaluate_all(preds, gt, model_name="unit_test")
    print(json.dumps(res, indent=2))
