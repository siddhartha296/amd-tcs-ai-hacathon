import torch
import time
import os

N = 4096

# Generate matrices on CPU
A = torch.randn(N, N)
B = torch.randn(N, N)

# -------------------------
# Single-threaded CPU
# -------------------------
torch.set_num_threads(1)

start = time.perf_counter()
C = torch.matmul(A, B)
end = time.perf_counter()

single_thread_time = end - start

print(f"Single-thread CPU Time: {single_thread_time:.4f} sec")

# -------------------------
# Multi-threaded CPU
# -------------------------
torch.set_num_threads(os.cpu_count())

start = time.perf_counter()
C = torch.matmul(A, B)
end = time.perf_counter()

multi_thread_time = end - start

print(f"Multi-thread CPU Time: {multi_thread_time:.4f} sec")

# -------------------------
# GPU (ROCm)
# -------------------------
if torch.cuda.is_available():

    A_gpu = A.cuda()
    B_gpu = B.cuda()

    torch.cuda.synchronize()

    start = time.perf_counter()

    C_gpu = torch.matmul(A_gpu, B_gpu)

    torch.cuda.synchronize()

    end = time.perf_counter()

    gpu_time = end - start

    print(f"GPU Time: {gpu_time:.4f} sec")

    print(f"Speedup vs Single Thread: {single_thread_time/gpu_time:.2f}x")
    print(f"Speedup vs Multi Thread: {multi_thread_time/gpu_time:.2f}x")
