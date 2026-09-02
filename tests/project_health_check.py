import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPECTED_CHAMP_HASH = "9a881de5a99cbcc6c5eb189a5d6f99b28e4b6781735f589c2dc15f1175f7cf6c"

import hashlib
def get_sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while c := f.read(8192): h.update(c)
    return h.hexdigest()

champ_path = os.path.join(PROJECT_ROOT, "artifacts/models/phase4/lightgbm_hybrid_seq/lightgbm_hybrid_seq.txt")
data_path = os.path.join(PROJECT_ROOT, "data/processed/product_data.csv")

if not os.path.exists(champ_path):
    print("FAIL: Champion missing.")
    exit(1)
if get_sha(champ_path) != EXPECTED_CHAMP_HASH:
    print("FAIL: Champion hash mismatch.")
    exit(1)
if not os.path.exists(data_path):
    print("FAIL: Processed data missing.")
    exit(1)

print("HEALTH CHECK: PASS")
