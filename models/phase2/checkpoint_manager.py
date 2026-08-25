import os
import json
import time
import shutil

class CheckpointManager:
    """
    Mandatory Rule 4 & 9 Checkpoint Manager for Phase 2 Models.
    Verifies version compatibility, prevents unnecessary retraining, and handles resume policy.
    """
    def __init__(self, root_dir="../.."):
        self.root_dir = root_dir
        self.models_dir = os.path.join(root_dir, "artifacts", "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def print_checkpoint_status(self, exists, compatible, resume_possible, retraining_required, reason):
        print("\n=========================================================================")
        print(" CHECKPOINT STATUS:")
        print(f" * Existing checkpoint: {'YES' if exists else 'NO'}")
        print(f" * Compatible:          {'YES' if compatible else 'NO'}")
        print(f" * Resume possible:     {'YES' if resume_possible else 'NO'}")
        print(f" * Retraining required: {'YES' if retraining_required else 'NO'}")
        print(f" * Reason:              {reason}")
        print("=========================================================================\n")

    def check_checkpoint(self, model_name, current_config):
        """
        Checks if an existing compatible checkpoint exists for model_name.
        Returns: (can_reuse, checkpoint_dir, metadata)
        """
        model_dir = os.path.join(self.models_dir, model_name)
        meta_path = os.path.join(model_dir, "metadata.json")

        if not os.path.exists(model_dir) or not os.path.exists(meta_path):
            self.print_checkpoint_status(
                exists=False, compatible=False, resume_possible=False,
                retraining_required=True, reason=f"No checkpoint found at {model_dir}"
            )
            return False, model_dir, None

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)

            # Check required version signatures
            if meta.get("dataset_version") != current_config.get("dataset_version", "1.0"):
                self.print_checkpoint_status(
                    exists=True, compatible=False, resume_possible=False,
                    retraining_required=True, reason="dataset_version mismatch"
                )
                return False, model_dir, meta

            if meta.get("preprocessing_version") != current_config.get("preprocessing_version", "option_b_v1"):
                self.print_checkpoint_status(
                    exists=True, compatible=False, resume_possible=False,
                    retraining_required=True, reason="preprocessing_version mismatch"
                )
                return False, model_dir, meta

            if meta.get("split_version") != current_config.get("split_version", "2.0"):
                self.print_checkpoint_status(
                    exists=True, compatible=False, resume_possible=False,
                    retraining_required=True, reason="split_version mismatch"
                )
                return False, model_dir, meta

            # Check if model artifact exists
            model_file = meta.get("model_file")
            if not model_file or not os.path.exists(os.path.join(model_dir, model_file)):
                self.print_checkpoint_status(
                    exists=True, compatible=False, resume_possible=False,
                    retraining_required=True, reason=f"Model artifact {model_file} missing from disk"
                )
                return False, model_dir, meta

            self.print_checkpoint_status(
                exists=True, compatible=True, resume_possible=True,
                retraining_required=False,
                reason=f"Found valid compatible checkpoint (trained {meta.get('created_at', 'previously')}). Reusing!"
            )
            return True, model_dir, meta

        except Exception as e:
            self.print_checkpoint_status(
                exists=True, compatible=False, resume_possible=False,
                retraining_required=True, reason=f"Failed to load checkpoint metadata: {e}"
            )
            return False, model_dir, None

    def save_checkpoint(self, model_name, model_obj, metadata, is_lightgbm=False, is_pytorch=False):
        """
        Saves model and metadata to disk.
        """
        model_dir = os.path.join(self.models_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        if is_lightgbm:
            model_file = f"{model_name}.txt"
            model_path = os.path.join(model_dir, model_file)
            model_obj.save_model(model_path)
            metadata["model_file"] = model_file
        elif is_pytorch:
            import torch
            model_file = f"{model_name}.pt"
            model_path = os.path.join(model_dir, model_file)
            torch.save(model_obj.state_dict(), model_path)
            metadata["model_file"] = model_file
        else:
            import pickle
            model_file = f"{model_name}.pkl"
            model_path = os.path.join(model_dir, model_file)
            with open(model_path, "wb") as f:
                pickle.dump(model_obj, f)
            metadata["model_file"] = model_file

        metadata["created_at"] = time.ctime()
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[SUCCESS] Saved checkpoint and metadata for {model_name} in {model_dir}")

    def load_lightgbm_model(self, model_dir, model_file):
        import lightgbm as lgb
        model_path = os.path.join(model_dir, model_file)
        print(f"[INFO] Loading LightGBM model from {model_path}...")
        return lgb.Booster(model_file=model_path)

if __name__ == '__main__':
    cm = CheckpointManager(root_dir=".")
    print("CheckpointManager ready.")
