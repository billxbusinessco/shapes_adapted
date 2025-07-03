import multiprocessing
import time

# Simulate CPU-heavy task
def heavy_task(x):
    total = 0
    for i in range(10000000):
        total += x * i
    return total

# Serial version
def run_serial():
    start = time.time()
    results = [heavy_task(x) for x in range(4)]
    end = time.time()
    print(f"Serial time: {end - start:.2f} seconds")
    return results

# Parallel version
def run_parallel():
    start = time.time()
    with multiprocessing.Pool() as pool:
        results = pool.map(heavy_task, range(4))
    end = time.time()
    print(f"Parallel time: {end - start:.2f} seconds")
    print(results)
    return results

if __name__ == "__main__":
    run_serial()
    run_parallel()
