import subprocess
import time
import psutil
import sys

def run_script(script_path):
    # Start the subprocess
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    pid = process.pid
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"Process {pid} does not exist.")
        return None

    start_time = time.time()
    peak_memory = 0  # in MB

    while True:
        if process.poll() is not None:
            break
        try:
            mem = proc.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
            if mem > peak_memory:
                peak_memory = mem
        except psutil.NoSuchProcess:
            break
        time.sleep(0.1)  # Polling interval

    end_time = time.time()
    elapsed_time = end_time - start_time

    stdout, stderr = process.communicate()

    return {
        'time': elapsed_time,
        'memory': peak_memory,
        'stdout': stdout,
        'stderr': stderr
    }

def main():
    scripts = ['traditional.py', 'train2.py']
    results = {}

    for script in scripts:
        print(f"Running {script}...")
        result = run_script(script)
        if result:
            results[script] = result
            print(f"Completed {script}")
            print(f"Time Taken: {result['time']:.2f} seconds")
            print(f"Peak Memory Usage: {result['memory']:.2f} MB\n")

            if result['stdout'].strip():
                print(f"Output from {script}:\n{result['stdout']}")
            if result['stderr'].strip():
                print(f"Errors from {script}:\n{result['stderr']}")
        else:
            print(f"Failed to run {script}.\n")

    # Summary
    print("\n=== Comparison Summary ===")
    for script, result in results.items():
        print(f"{script}:")
        print(f"  Time Taken       : {result['time']:.2f} seconds")
        print(f"  Peak Memory Usage: {result['memory']:.2f} MB\n")

if __name__ == "__main__":
    main()
