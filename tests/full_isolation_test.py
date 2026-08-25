import os
import sys
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ORIGINAL_PROJECT = r"C:\Users\Admin\Downloads\instacart-basket"

print("Starting FULL ISOLATION TEST...")
print(f"Checking if original project is accessible... {os.path.exists(ORIGINAL_PROJECT)}")

# Verify the inference works
res = subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "run_inference.py"), "--user-id", "1"], capture_output=True, text=True, cwd=PROJECT_ROOT)
if "[RESULT] Inference Success!" in res.stdout:
    print("FULL ISOLATION TEST: PASS")
else:
    print("FULL ISOLATION TEST: FAIL")
    print(res.stderr)
    print(res.stdout)
