import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPECTED_CHAMP_HASH = "157c048aedc386cf2209a4484ffe4d42bb7f5ef8f6696cc8669ee8befd1f1535"

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
