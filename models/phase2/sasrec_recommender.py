import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from split_manager import SplitManager, SPLIT_VERSION
from checkpoint_manager import CheckpointManager
from evaluation import Evaluator

class SASRecModel(nn.Module):
    """
    Clean, Causal SASRec (Self-Attentive Sequential Recommendation) Transformer.
    Optimized for high-speed CPU training and candidate ranking.
    """
    def __init__(self, item_num=49688, max_seq_len=30, hidden_dim=64, num_heads=2, num_layers=2, dropout=0.2):
        super().__init__()
        self.item_num = item_num
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

        self.item_emb = nn.Embedding(item_num + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_seqs):
        device = input_seqs.device
        batch_size, seq_len = input_seqs.shape

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        seq_embs = self.item_emb(input_seqs) + self.pos_emb(positions)
        seq_embs = self.emb_dropout(seq_embs)

        mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        key_padding_mask = (input_seqs == 0)

        out = self.encoder(seq_embs, mask=mask, src_key_padding_mask=key_padding_mask)
        out = self.norm(out)

        final_rep = out[:, -1, :]
        return final_rep

    def score_items(self, user_reps, item_ids):
        """
        user_reps: (batch_size, hidden_dim)
        item_ids: (batch_size, num_items)
        """
        embs = self.item_emb(item_ids)  # (batch_size, num_items, hidden_dim)
        return torch.bmm(embs, user_reps.unsqueeze(-1)).squeeze(-1)

class InstacartSeqDataset(Dataset):
    def __init__(self, seq_dict, max_seq_len=30):
        self.users = list(seq_dict.keys())
        self.seqs = [seq_dict[u] for u in self.users]
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        uid = self.users[idx]
        s = self.seqs[idx]
        if len(s) > self.max_seq_len:
            s = s[-self.max_seq_len:]
        else:
            s = [0] * (self.max_seq_len - len(s)) + s
        return uid, torch.tensor(s, dtype=torch.long)

class SASRecRecommender:
    """
    Phase 2C: Professional SASRec Sequential Recommender with fast CPU training & checkpointing.
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
        self.model_name = "sasrec"
        self.config = {
            "dataset_version": "1.0",
            "preprocessing_version": "option_b_v1",
            "split_version": SPLIT_VERSION,
            "model_version": "2.0-sasrec",
            "hyperparameters": {
                "max_seq_len": 30,
                "hidden_dim": 64,
                "num_heads": 2,
                "num_layers": 2,
                "dropout": 0.2,
                "batch_size": 1024,
                "learning_rate": 0.002,
                "epochs": 3,
                "num_neg_samples": 100
            }
        }
        self.device = torch.device("cpu")

    def build_user_sequences_fast(self, target_users):
        """
        Ultra-fast sequence builder using numpy arrays instead of pandas groupby.apply(list).
        """
        t0 = time.time()
        print(f"[INFO] Building chronological pre-target sequences for {len(target_users):,} users...")
        prior_path = os.path.join(self.root_dir, "data/raw/order_products__prior.csv")
        orders_path = os.path.join(self.root_dir, "data/raw/orders.csv")

        orders_df = pd.read_csv(orders_path, usecols=['order_id', 'user_id', 'order_number', 'eval_set'])
        orders_df = orders_df[(orders_df['user_id'].isin(target_users)) & (orders_df['eval_set'] == 'prior')]

        prior_df = pd.read_csv(prior_path, usecols=['order_id', 'product_id', 'add_to_cart_order'])
        m_df = orders_df.merge(prior_df, on='order_id')
        m_df = m_df.sort_values(by=['user_id', 'order_number', 'add_to_cart_order'])

        # Fast O(N) Python dictionary building from numpy arrays
        seq_dict = {}
        u_vals = m_df['user_id'].values
        p_vals = m_df['product_id'].values

        for u, p in zip(u_vals, p_vals):
            if u in seq_dict:
                seq_dict[u].append(int(p))
            else:
                seq_dict[u] = [int(p)]

        for uid in target_users:
            if uid not in seq_dict:
                seq_dict[uid] = []

        print(f"[INFO] Sequences built in {time.time() - t0:.2f}s. Total users: {len(seq_dict):,}")
        return seq_dict

    def train_or_load(self):
        print("=========================================================================")
        print(" PHASE 2C: SASREC TRANSFORMER TRAINING & CHECKPOINT VERIFICATION         ")
        print("=========================================================================")

        can_reuse, model_dir, existing_meta = self.cm.check_checkpoint(self.model_name, self.config)

        model = SASRecModel(
            item_num=49688,
            max_seq_len=self.config["hyperparameters"]["max_seq_len"],
            hidden_dim=self.config["hyperparameters"]["hidden_dim"],
            num_heads=self.config["hyperparameters"]["num_heads"],
            num_layers=self.config["hyperparameters"]["num_layers"],
            dropout=self.config["hyperparameters"]["dropout"]
        ).to(self.device)

        if can_reuse and existing_meta:
            model_path = os.path.join(model_dir, existing_meta["model_file"])
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            return model, existing_meta, 0.0

        print("[INFO] No compatible checkpoint found. Starting efficient CPU training...")
        t0_train = time.time()

        sm = SplitManager(root_dir=self.root_dir)
        splits = sm.load_splits()
        train_users = set(splits['user_train']['user_id'])

        train_seqs = self.build_user_sequences_fast(train_users)
        train_ds = InstacartSeqDataset(train_seqs, max_seq_len=self.config["hyperparameters"]["max_seq_len"])
        train_loader = DataLoader(train_ds, batch_size=self.config["hyperparameters"]["batch_size"], shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["hyperparameters"]["learning_rate"])
        criterion = nn.CrossEntropyLoss()
        num_neg = self.config["hyperparameters"]["num_neg_samples"]

        model.train()
        for epoch in range(self.config["hyperparameters"]["epochs"]):
            t_epoch = time.time()
            total_loss = 0.0
            for uids, seqs in train_loader:
                seqs = seqs.to(self.device)
                inputs = seqs[:, :-1]
                targets = seqs[:, -1]

                valid_mask = (targets != 0)
                if not valid_mask.any():
                    continue

                optimizer.zero_grad()
                reps = model(inputs)

                # Sampled softmax: true target + num_neg random negative items
                batch_size = len(targets)
                neg_items = torch.randint(1, 49689, (batch_size, num_neg), device=self.device)
                cand_items = torch.cat([targets.unsqueeze(1), neg_items], dim=1) # (batch_size, 1 + num_neg)

                logits = model.score_items(reps, cand_items) # (batch_size, 1 + num_neg)
                # The true target is always at index 0
                labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)

                loss = criterion(logits[valid_mask], labels[valid_mask])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"[Epoch {epoch+1}/{self.config['hyperparameters']['epochs']}] Loss: {total_loss/len(train_loader):.4f} | Time: {time.time() - t_epoch:.2f}s")

        train_time = time.time() - t0_train

        metadata = self.config.copy()
        metadata["train_time_sec"] = train_time
        self.cm.save_checkpoint(self.model_name, model, metadata, is_pytorch=True)
        return model, metadata, train_time

    def evaluate_test_set(self, model, train_time):
        t0 = time.time()
        print("=========================================================================")
        print(" PHASE 2C: SASREC TRANSFORMER TEST SET EVALUATION                        ")
        print("=========================================================================")

        sm = SplitManager(root_dir=self.root_dir)
        splits = sm.load_splits()
        test_users = set(splits['user_test']['user_id'])

        test_seqs = self.build_user_sequences_fast(test_users)

        print("[INFO] Loading candidate sets from product_data.csv for test users...")
        test_dfs = []
        for chunk in pd.read_csv(os.path.join(self.root_dir, "data/processed/product_data.csv"), chunksize=1000000, keep_default_na=False):
            matched = chunk[chunk['user_id'].isin(test_users)]
            if len(matched) > 0:
                test_dfs.append(matched[['user_id', 'product_id', 'label']])
        test_prod_df = pd.concat(test_dfs, ignore_index=True)

        user_cands = test_prod_df.groupby('user_id')['product_id'].apply(list).to_dict()
        gt_dict = self.evaluator.extract_ground_truth_from_product_data(test_prod_df)

        model.eval()
        preds_dict = {}
        t_inf_start = time.time()

        test_ds = InstacartSeqDataset(test_seqs, max_seq_len=self.config["hyperparameters"]["max_seq_len"])
        test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

        with torch.no_grad():
            for uids, seqs in test_loader:
                seqs = seqs.to(self.device)
                reps = model(seqs)

                for i, uid_tensor in enumerate(uids):
                    uid = int(uid_tensor)
                    cands = user_cands.get(uid, [])
                    if not cands:
                        continue
                    cand_tensor = torch.tensor(cands, dtype=torch.long, device=self.device)
                    scores = torch.matmul(reps[i], model.item_emb(cand_tensor).t()).cpu().numpy()

                    top_idx = np.argsort(scores)[::-1][:10]
                    preds_dict[uid] = [cands[idx] for idx in top_idx]

        inf_time = time.time() - t_inf_start
        print(f"[INFO] SASRec inference completed in {inf_time:.2f}s ({inf_time/len(test_users)*1000:.2f} ms/user).")

        metrics = self.evaluator.evaluate_all(
            predictions_dict=preds_dict,
            ground_truth_dict=gt_dict,
            model_name="SASRec_Recommender",
            time_info={
                "train_time_sec": train_time,
                "inference_time_sec": inf_time,
                "inference_ms_per_user": (inf_time / len(test_users)) * 1000.0,
                "total_test_users": len(test_users)
            }
        )

        metrics_path = os.path.join(self.metrics_dir, "sasrec_recommender.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print("=========================================================================")
        print(" [SUCCESS] SASREC TRANSFORMER RESULTS (UNSEEN TEST USERS):               ")
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
    sasrec = SASRecRecommender(root_dir=".")
    model, meta, t_train = sasrec.train_or_load()
    sasrec.evaluate_test_set(model, t_train)
