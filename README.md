# ISTA: Invariant Simplex Tensor Attention

**Sub-1% relative error on genuine 4-body geometric invariants with only 1,100 parameters**

---

## Overview

ISTA explores the trade-off between computational cost and parameter efficiency for learning complex geometric invariants. It implements a 4-simplex attention mechanism (`Simplex4Net`) that exhaustively scans all O(N⁴) 4-tuples of a 3D point cloud, computes determinant-squared geometric features, and learns attention weights for feature aggregation.

**Core Hypothesis:** Paying O(N⁴) compute cost enables extreme parameter efficiency. Standard O(N²) attention models need 100k–500k parameters and reach 2–5% error on complex invariant tasks. ISTA achieves **<1% error with ~1,100 parameters** by explicitly computing every 4-body interaction.

## Target Function

The learning target is a true 4-body geometric invariant:

$$\text{target}(P) = \frac{1}{N^4}\sum_{i,j,k,l=1}^{N} \det^2\bigl[\mathbf{p}_j - \mathbf{p}_i,\;\mathbf{p}_k - \mathbf{p}_i,\;\mathbf{p}_l - \mathbf{p}_i\bigr]$$

where $\det[\mathbf{a}, \mathbf{b}, \mathbf{c}] = \mathbf{a} \cdot (\mathbf{b} \times \mathbf{c})$ is the scalar triple product (signed volume of the parallelepiped).

**Properties:** Permutation invariant, rotation/reflection invariant (det² ≥ 0), translation invariant (uses displacements). This is a *genuine* 4-body function — it cannot be factorized into pair or triplet interactions.

With N=256 points, each sample involves **4.3 billion** ordered 4-tuples.

## Architecture

### Simplex4Net

The model uses four separate linear projections and a 6-term additive attention energy with sigmoid gating:

```
Projections (3 → 64 each):
  q = W_q · x      (query)
  k_j = W_j · x    (key 1)
  k_k = W_k · x    (key 2)
  k_l = W_l · x    (key 3)

Attention energy for each 4-tuple (i, j, k, l):
  E(i,j,k,l) = (q_i · k_j + q_i · k_k + q_i · k_l
               + k_j · k_k + k_j · k_l + k_k · k_l) / √d

Gating:
  A(i,j,k,l) = σ(E(i,j,k,l))         # sigmoid, not softmax

Geometric feature:
  F(i,j,k,l) = det²([p_j−p_i, p_k−p_i, p_l−p_i])

Aggregation:
  h = (1/N⁴) Σ_{i,j,k,l} A(i,j,k,l) · F(i,j,k,l)

Output head:
  ŷ = Dense(32) → GELU → Dense(1)     applied to h
```

**Key design choices:**
- **Sigmoid gating** (independent per 4-tuple) rather than softmax — each gate independently decides how much to weight its determinant feature
- **No learned value vectors** — the geometric det² features serve as the "values"
- **6-term additive energy** covers all $\binom{4}{2}$ pairwise interactions among the four projected representations

### Parameter Count

| Component | Parameters |
|---|---|
| 4 × Dense(3 → 64) | 4 × 256 = 1,024 |
| Dense(1 → 32) + bias | 64 |
| Dense(32 → 1) + bias | 33 |
| **Total** | **1,121** |

## Memory Optimization

A naïve O(N⁴) implementation would require materializing an N×N×N×N tensor — over 17 GB for N=256 in float32. ISTA makes this tractable via:

1. **`jax.lax.scan`** over the anchor dimension (i), reducing peak memory to O(N³) per iteration
2. **Gradient checkpointing** (`jax.checkpoint`) to recompute forward activations during backprop instead of storing them
3. **GSPMD sharding** across TPU cores — the batch dimension is partitioned so each device processes a subset of samples independently

## Results

**Task:** Learning the 4-body det² invariant of random 3D point clouds (N=256)

| Metric | Value |
|---|---|
| Parameters | 1,121 |
| Training samples | 128 |
| Epochs | 10,000 |
| RMSE (original scale) | 0.0266 |
| Target std | 3.42 |
| Target mean | 23.3 |
| **Relative error (RMSE / std)** | **0.78%** |
| Signal-to-noise ratio | 58.9 dB |

Compared to standard transformer models on comparable tasks (N=256, complex invariants), which typically use 100k–500k parameters and achieve 2–5% error, ISTA achieves superior accuracy with **~100× fewer parameters** by paying the O(N⁴) computational cost.

## Training Configuration

| Setting | Value |
|---|---|
| Framework | JAX + Flax |
| Loss | MSE (on normalized targets) |
| Target normalization | Zero mean, unit variance |
| Optimizer | Adam (ε = 1e-8) |
| Gradient clipping | Global norm ≤ 5.0 |
| LR schedule | Warmup-cosine decay |
| Initial / peak / final LR | 1e-4 → 5e-2 (warmup 100 steps) → 1e-7 |
| Hardware | TPU v5e pod (v5litepod-32) |
| Precision | float32 (float64 option available) |

## Usage

### Running on TPU

The included `TPU.py` script automates the full TPU workflow (create/reuse pod, upload code, launch training, tail logs):

```bash
python TPU.py
```

It will prompt you to select a zone, choose between creating a new TPU pod or reusing an existing one, then handle setup and launch across all 8 workers of a v5litepod-32.

### Running Locally (GPU/CPU)

```bash
pip install jax flax optax
python main.py
```

> **Note:** With N=256 points, the O(N⁴) computation is very demanding. A single GPU will be significantly slower than a TPU pod. Reduce `NUM_POINTS_PER_CLOUD` or `NUM_TRAINING_SAMPLES` in `main.py` for faster local experimentation.

## Research Directions

- **Numerical precision:** float64 mode is available (`jax_enable_x64`) to test whether float32 is the accuracy bottleneck
- **Scaling N:** Larger point clouds (N=512+) with improved memory optimization
- **Higher-order simplices:** O(N⁵), O(N⁶) for 5-body and 6-body invariants
- **Efficient approximations:** Pruning low-energy 4-tuples while maintaining accuracy
- **Applications:** Molecular dynamics, protein geometry, multi-agent systems

## Acknowledgments

This research is supported by **Google's Tensor Research Cloud (TRC)** program, which provides TPU access for academic and open-source research. The O(N⁴) complexity makes standard cloud compute economically infeasible — TRC enables exploration of architectures that prioritize *precision over efficiency*.

---

## Citation

```bibtex
@software{ista2026,
  title={ISTA: Invariant Simplex Tensor Attention},
  author={Aayush Bhatt},
  year={2026},
  url={https://github.com/AayushBhatt0/ISTA}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

**Note:** This is experimental research code. The O(N⁴) complexity is impractical for large N, but demonstrates that exhaustive combinatorial computation can replace large parameter counts for structured geometric tasks.
