# CUDA LayerNorm + Linear Fusion (Kimi-K2.6 Scale)

**Goal:** Speed up LayerNorm + Linear layer execution by fusing operations at Kimi-K2.6 tensor dimensions

## Method

Given LayerNorm with parameters γ (gamma) and β (beta) followed by Linear with weight F:

1. Replace Linear weights: `F ← (I - E/n) @ diag(γ) @ F`
   - I = identity matrix
   - E = matrix of all ones
   - n = hidden dimension
   - γ = LayerNorm gamma (scale)

2. Adjust Linear bias: `b = original_bias + β @ F`
   - β = LayerNorm beta (shift)

3. Compute LayerNorm denominator (std) concurrently with Linear matmul

4. Divide Linear output by denominator, add bias

## Advantage

- Same output as sequential LayerNorm → Linear
- Hides SIMD latency by parallelizing normalization computation

## Kimi-K2.6 dimensions

| Parameter | Value |
|-----------|-------|
| Hidden size | 7168 |
| FFN output | 25088 (7168 × 3.5, same ratio as large LLaMA-class FFN) |
| MoE expert dim (model card) | 2048 |

Benchmarks use synthetic `LayerNorm` + `Linear` modules at these shapes (not a full Kimi checkpoint).

## Run benchmark

```bash
python kimi26_benchmark.py
```

Requires NVIDIA GPU, PyTorch 2.0+ with CUDA (`torch.compile`).

## Status (Updated 2026-05-25)

- [x] GPU access (RunPod L4, 23GB VRAM)
- [x] Weight transformation and fused PyTorch module
- [x] Kimi-K2.6 scale benchmark with `torch.compile`

## Files

- `kimi26_benchmark.py` — Kimi-K2.6 scale benchmark (primary; self-contained)


## Benchmark results (NVIDIA L4)

**Environment:** PyTorch 2.6.0+cu124, CUDA 12.4, NVIDIA L4  
**Run:** `python kimi26_benchmark.py`  
**Numerical accuracy:** max diff ≈ 1.2×10⁻⁵ – 1.6×10⁻⁵ across configs (fused matches baseline)

| Config | Baseline (LN+Linear) | Fused | Speedup |
|--------|----------------------|-------|---------|
| FFN, batch=1, seq=2048 (7168→25088) | 78.73 ms | 80.96 ms | 0.97x (slower) |
| FFN, batch=1, seq=4096 (7168→25088) | 159.82 ms | 167.13 ms | 0.96x (slower) |
| FFN, batch=4, seq=2048 (7168→25088) | 329.72 ms | 345.03 ms | 0.96x (slower) |
| Attention proj, batch=1, seq=2048 (7168→7168) | 23.60 ms | 25.97 ms | 0.91x (slower) |

Peak GPU memory during run: ~2.9 GB (FFN configs), ~5.0 GB (attention proj config).

## Conclusion

At Kimi-K2.6 layer shapes on an L4, the fused LayerNorm→Linear path is **correct** (≈1e-5 max error vs baseline) but **consistently slower** (0.91x–0.97x) than separate `LayerNorm` + `Linear` with `torch.compile`.

PyTorch’s built-in LayerNorm and GEMM are already highly optimized; fusing in pure PyTorch does not recover enough from overlapping variance work to beat that stack.

**Recommendation:** Meaningful speedup would likely require custom CUDA/Triton kernels (and integration into a full model forward pass), not this standalone micro-benchmark alone.
