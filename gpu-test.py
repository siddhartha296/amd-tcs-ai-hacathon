import torch
import time

# Verify ROCm GPU
print("PyTorch:", torch.__version__)
print("GPU Available:", torch.cuda.is_available())

device = torch.device("cuda")

# Matrix size
N = 32768  # Increase if you have enough VRAM

# Use float16 for maximum throughput
dtype = torch.float16

print(f"Creating {N}x{N} matrices...")

A = torch.randn((N, N), device=device, dtype=dtype)
B = torch.randn((N, N), device=device, dtype=dtype)

# Warmup
for _ in range(5):
    C = torch.matmul(A, B)

torch.cuda.synchronize()

# Benchmark
iters = 20
start = time.perf_counter()

for _ in range(iters):
    C = torch.matmul(A, B)

torch.cuda.synchronize()
end = time.perf_counter()

avg_time = (end - start) / iters

# FLOPs for GEMM = 2*N^3
tflops = (2 * (N**3)) / avg_time / 1e12

print(f"\nMatrix size: {N}x{N}")
print(f"Average time: {avg_time:.4f} s")
print(f"Performance: {tflops:.2f} TFLOPS")
print(f"Result shape: {C.shape}")
