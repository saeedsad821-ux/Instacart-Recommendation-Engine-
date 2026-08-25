import os
import sys
import time
import subprocess
import psutil

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_script(script_name):
    print(f"\n============================================================")
    print(f">>> Running {script_name}...")
    print(f"============================================================")
    start_time = time.time()
    cmd = [sys.executable, script_name]
    res = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    elapsed = time.time() - start_time
    print(f"\n[DONE] {script_name} finished in {elapsed:.2f}s with return code {res.returncode}")
    assert res.returncode == 0, f"Script {script_name} failed with return code {res.returncode}!"
    return elapsed

if __name__ == '__main__':
    total_start = time.time()
    print("=========================================================================")
    print(" STARTING INSTACART OPTION B PREPROCESSING & LEAKAGE AUDIT PIPELINE      ")
    print("=========================================================================")
    
    # Check if user_data.csv exists; if not, run create_user_data.py
    if not os.path.exists('../data/processed/user_data.csv'):
        run_script('create_user_data.py')
    else:
        print("\n[INFO] ../data/processed/user_data.csv already exists. Skipping create_user_data.py.")
        
    # Run the product, aisle, and department data creation scripts
    run_script('create_product_data.py')
    run_script('create_aisle_data.py')
    run_script('create_department_data.py')
    
    # Run the comprehensive audit suite
    run_script('audit_preprocessing_split.py')
    
    print("\n=========================================================================")
    print(f" PIPELINE COMPLETED SUCCESSFULLY IN {time.time() - total_start:.2f}s!")
    print("=========================================================================")
