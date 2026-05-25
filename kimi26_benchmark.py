"""Kimi-K2.6 Scale Benchmark for Fused LayerNorm-Linear"""
import torch
import torch.nn as nn
import time

device = torch.device("cuda")

def transform_weights(gamma, beta, weight, bias):
    n = gamma.shape[0]
    new_weight = weight * gamma.unsqueeze(0)
    c = weight @ gamma
    correction = c.unsqueeze(1) * torch.ones(1, n, device=weight.device) / n
    return new_weight - correction, bias + weight @ beta

class FusedLayerNormLinear(nn.Module):
    def __init__(self, ln, linear):
        super().__init__()
        self.weight, self.bias = [nn.Parameter(t) for t in transform_weights(
            ln.weight, ln.bias, linear.weight, linear.bias)]
        self.eps = ln.eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = torch.sqrt(((x - mean) ** 2).mean(dim=-1, keepdim=True) + self.eps)
        return (x @ self.weight.T) / std + self.bias

def benchmark(batch, seq, hidden, out, name, warmup=20, iters=100):
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"Config: batch={batch}, seq={seq}, hidden={hidden}, out={out}")
    print(f"{'='*60}")
    
    ln = nn.LayerNorm(hidden).to(device)
    linear = nn.Linear(hidden, out).to(device)
    fused = FusedLayerNormLinear(ln, linear).to(device)
    
    # Compile both
    baseline_fn = torch.compile(lambda x: linear(ln(x)))
    fused_fn = torch.compile(fused)
    
    x = torch.randn(batch, seq, hidden, device=device)
    
    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {fused.weight.shape}")
    print(f"Memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    
    # Warmup
    print("Compiling & warming up...")
    for _ in range(warmup):
        _ = baseline_fn(x); _ = fused_fn(x)
    torch.cuda.synchronize()
    
    # Verify
    with torch.no_grad():
        diff = (baseline_fn(x) - fused_fn(x)).abs().max().item()
        print(f"Numerical accuracy: max diff = {diff:.2e}")
    
    # Benchmark
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): _ = baseline_fn(x)
    torch.cuda.synchronize(); baseline_t = (time.perf_counter() - t0) / iters * 1000
    
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): _ = fused_fn(x)
    torch.cuda.synchronize(); fused_t = (time.perf_counter() - t0) / iters * 1000
    
    speedup = baseline_t / fused_t
    print(f"\nResults:")
    print(f"  Baseline (LN+Linear): {baseline_t:.3f} ms")
    print(f"  Fused:                {fused_t:.3f} ms")
    print(f"  Speedup:              {speedup:.2f}x ({'faster' if speedup > 1 else 'slower'})")
    
    return baseline_t, fused_t, speedup

if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    
    # Kimi-K2.6: hidden=7168, MoE expert dim=2048 (model card)
    # FFN out dim ~3.5x hidden (same ratio as LLaMA-70B 8192->28672)
    kimi_ffn_out = 25088  # 7168 * 3.5
    
    configs = [
        (1, 2048, 7168, kimi_ffn_out, "Kimi-K2.6 FFN (batch=1, seq=2048)"),
        (1, 4096, 7168, kimi_ffn_out, "Kimi-K2.6 FFN (batch=1, seq=4096)"),
        (4, 2048, 7168, kimi_ffn_out, "Kimi-K2.6 FFN (batch=4, seq=2048)"),
        (1, 2048, 7168, 7168, "Kimi-K2.6 Attention proj"),
    ]
    
    results = []
    for batch, seq, hidden, out, name in configs:
        try:
            r = benchmark(batch, seq, hidden, out, name)
            results.append((name, *r))
        except Exception as e:
            print(f"FAILED: {name} - {e}")
    
    print("\n" + "="*60)
    print("SUMMARY - Kimi-K2.6 Scale")
    print("="*60)
    for name, baseline, fused, speedup in results:
        status = "faster" if speedup > 1.0 else "slower"
        print(f"{name}")
        print(f"  {baseline:.2f}ms -> {fused:.2f}ms = {speedup:.2f}x ({status})")
