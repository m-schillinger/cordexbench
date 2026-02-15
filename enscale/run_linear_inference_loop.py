import os
import subprocess
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Configuration ---
ROOT_SAMPLES = "/r/scratch/groups/nm/downscaling/samples_cordexbench/"
DOMAINS = ["ALPS", "NZ", "SA"]
EXPERIMENTS = ["ESD_pseudo_reality", "Emulator_hist_future"]
TARGETVARS = ["pr", "tasmax"]

def run_command(cmd_args):
    """Function executed by each worker process."""
    exp, dom, var = cmd_args
    cmd = [
        "python", "linear_model_inference.py",
        "--domain", dom,
        "--target_var", var,
        "--training_experiment", exp
    ]
    
    print(f"[STARTING] {exp} | {dom} | {var}")
    try:
        # start_new_session=True allows us to group signals if needed, 
        # but check=True is standard for waiting on completion.
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return f"[SUCCESS] {exp} | {dom} | {var}"
    except subprocess.CalledProcessError as e:
        return f"[FAILED] {exp} | {dom} | {var}\nError: {e.stderr}"
    except KeyboardInterrupt:
        # This helps child processes exit quietly on Ctrl+C
        return f"[CANCELLED] {exp} | {dom} | {var}"

def main():
    tasks = []
    
    # 1. Identify what needs to be run
    for exp in EXPERIMENTS:
        for dom in DOMAINS:
            for var in TARGETVARS:
                target_path = os.path.join(ROOT_SAMPLES, exp, dom, "no-orog", var, "linear_pred")
                
                if os.path.exists(target_path) and len(os.listdir(target_path)) == 12:
                    print(f"[SKIPPING] {exp} | {dom} | {var} (Already exists)")
                else:
                    tasks.append((exp, dom, var))

    if not tasks:
        print("\nAll models are already present. Nothing to do.")
        return

    print(f"\nReady to run {len(tasks)} tasks in parallel.")
    print("Press Ctrl+C to cancel all processes.\n")

    # 2. Execute in Parallel
    # max_workers=None defaults to number of processors on the machine
    executor = ProcessPoolExecutor(max_workers=6)
    
    try:
        futures = {executor.submit(run_command, task): task for task in tasks}
        for future in as_completed(futures):
            print(future.result())
            
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Ctrl+C detected. Terminating all workers...")
        executor.shutdown(wait=False, cancel_futures=True)
        print("Shutdown complete.")

if __name__ == "__main__":
    main()