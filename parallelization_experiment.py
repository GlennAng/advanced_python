from datetime import datetime
from process_users import load_users
import numpy as np
import time
import unittest

def compute_days_since_first_interaction(user, sleep_time = 0) -> dict:
    # Simulated complex processing
    time.sleep(sleep_time)
    return {"user_id": user.user_id, "days_since_first_interaction": (datetime(2025, 4, 30) - user.first_interaction).days}

# Sequential version without parallelization
def run_sequential(users, sleep_time) -> list:
    results = []
    for user in users:
        results.append(compute_days_since_first_interaction(user, sleep_time))
    return results

# Parallel version with joblib (n_jobs = -1 means using all available cores)
def run_parallel(users, sleep_time, n_jobs = -1) -> list:
    from joblib import Parallel, delayed
    results = Parallel(n_jobs = n_jobs)(delayed(compute_days_since_first_interaction)(user, sleep_time) for user in users)
    return results

def get_user_results(user_id, results) -> dict:
    for result in results:
        if result['user_id'] == user_id:
            return result
    raise ValueError(f"User ID {user_id} not found in results")

# Test class to compare sequential and parallel results (need to be equal)
class TestParallelization(unittest.TestCase):
    def test_results_equal(self, sequential_results, parallel_results):
        users_ids = list(users_ids_to_idxs.keys())
        for user_id in users_ids:
            sequential_result = get_user_results(user_id, sequential_results)
            parallel_result = get_user_results(user_id, parallel_results)
            self.assertEqual(sequential_result, parallel_result, f"Results for user {user_id} are not equal")

if __name__ == "__main__":
    users, users_ids_to_idxs = load_users()
    sleep_times = [0.0, 0.001]
    test = TestParallelization()
    for sleep_time in sleep_times:
        print(f"Sleep time: {sleep_time}")
        # Benchmark sequential version
        start = time.time()
        sequential_results = run_sequential(users, sleep_time)
        sequential_time = time.time() - start
        print(f"Sequential time: {sequential_time:.2f}s")

        # Benchmark parallel version
        start = time.time()
        parallel_results = run_parallel(users, sleep_time, n_jobs = -1)
        parallel_time = time.time() - start
        print(f"Parallel time: {parallel_time:.2f}s")
        print(f"Speedup: {sequential_time/parallel_time:.2f}x")
        test.test_results_equal(sequential_results, parallel_results)
        print("____________________________________________________________\n")