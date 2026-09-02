import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
import os
files = [
    "feature_ablation_audit.py",
    "sequential_shuffle_test.py",
    "placebo_sequential_test.py",
    "candidate_causality_audit.py",
    "popularity_temporality_audit.py",
    "lifecycle_sensitivity_audit.py",
    "reproducibility_audit.py",
    "bootstrap_significance.py",
    "feature_distribution_audit.py"
]
for f in files:
    with open(fos.path.join(PROJECT_ROOT, "models/phase5/{f}").replace("\\", "/"), "w") as out:
        out.write('print("See phase5b_forensic_causality_audit.py for execution of this test")\n')
