# ISTA: Invariant Simplex Tensor Attention

**Engineering O(N⁴) computation on TPUs: a study in inductive bias alignment**

---

## Overview

ISTA explores the trade-off between computational cost and parameter efficiency for learning complex geometric invariants. It implements a 4-simplex attention mechanism (`Simplex4Net`) that exhaustively scans all O(N⁴) 4-tuples of a 3D point cloud, computes determinant-squared geometric features, and learns attention weights for feature aggregation.

**Core Hypothesis:** ISTA explores whether O(N⁴) exhaustive computation can replace learned parameters. The model achieves <1% error — but post-hoc analysis reveals this is due to architecture-target alignment: the model internally recomputes the target formula directly, making the learned parameters largely superfluous. This is documented as a cautionary result about inductive bias.

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

Note: These results reflect training-set performance only. No held-out test set was used. The low error is attributable to architecture-target alignment, not generalization.

## Honest Analysis (Post-Hoc)

After TRC feedback submission, a critical flaw was identified:

### Architecture-Target Alignment Problem

The target function is:
```
target = (1/N⁴) Σ det²(...)
```

The model computes:
```
output = f( (1/N⁴) Σ sigmoid(E_ijkl) · det²(...) )
```

These are **structurally identical**. When attention projections converge 
to near-zero (the path of least resistance for Adam), sigmoid(E) → 0.5 
everywhere, and the output head learns to multiply by 2. The model 
reduces to a ~2-parameter solution regardless of total capacity.

### Zero-Parameter Baseline

The honest baseline is simply computing the raw det² average with no 
learned weights. This baseline would achieve comparable error to ISTA, 
because ISTA's attention degenerates to uniform weights.

**TODO:** Add zero-parameter baseline comparison.

### What the O(N⁴) Compute Actually Does

The quartic compute is real and necessary — but it is spent *evaluating 
the target formula*, not learning anything. The model recomputes 4.3B 
determinants per forward pass, which is the same computation used to 
generate the training labels. The engineering infrastructure (scan, 
checkpointing, GSPMD sharding) is valid; the scientific claim is not.

### Actual Contribution

The genuine contribution of this work is:

1. A working O(N⁴) JAX pipeline on TPU v5e with `jax.lax.scan`, 
   gradient checkpointing, and GSPMD sharding
2. A concrete example of how perfect inductive bias alignment makes 
   learned parameters redundant
3. A negative result: you cannot claim "parameter efficiency" when the 
   architecture encodes the answer

### What a Valid Experiment Would Look Like

To genuinely test the compute-vs-parameters tradeoff, the target must 
**not** be recoverable by uniform attention weights. Better targets:
- Variance of det² values (not a simple average)
- Selective invariants depending on which 4-tuples are geometrically 
  special
- Any target that requires the model to *discriminate* between 4-tuples

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
