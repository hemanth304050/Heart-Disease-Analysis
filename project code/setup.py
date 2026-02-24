"""
setup.py - Run this ONCE to set up the entire Heart Disease Analysis project.
"""
import subprocess, sys

def run(script, label):
    print("\n" + "="*60)
    print("[RUN] " + label)
    print("="*60)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print("[FAIL] " + label + " failed. See errors above.")
        sys.exit(1)
    print("[OK]  " + label + " done.")

print("\n=== Heart Disease Analysis - SmartBridge Setup ===\n")

run("generate_dataset.py",   "Step 1/4 | Data Collection & Extraction")
run("data_preparation.py",   "Step 2/4 | Data Preparation")
run("data_visualization.py", "Step 3/4 | Data Visualization (8 charts)")
run("performance_testing.py","Step 4/4 | ML Training & Performance Testing")

print("\n" + "="*60)
print("SUCCESS - All setup steps completed!")
print("="*60)
print("\nStart the web app:")
print("   python app.py")
print("\nThen open:  http://127.0.0.1:5000")
print()
