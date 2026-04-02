# LMF_V7 and Chain-Parallel Modification Report

## 1. Motivation

After `MLDC_Parallel` removed most of the MLDC bottleneck, the LMF frontend became the next dominant runtime hotspot in EchMo.

The original LMF path had two main issues:

- The serial LMF implementation (`LMF_Chain` + `Learnable_Match_FilterV6`) uses four real-valued convolutions to emulate one complex convolution.
- Multiple LMF chains were designed as parallel feature heads semantically, but were executed one chain at a time in Python.

The goal of this refactor was therefore:

1. Keep the V6 learning semantics that already work in training.
2. Reduce LMF latency significantly.
3. Introduce a true chain-parallel implementation.
4. Keep the architecture explicit and clean, with only one V7-named class.

## 2. Scope of the New LMF Changes

The new LMF-related changes include:

- `Learnable_Match_FilterV7`
  - A latency-optimized independent LMF layer.
- `LMF_Layer_Parallel`
  - A chain-parallel LMF layer backend shared by the parallel-chain path.
- `LMF_Chain_Parallel`
  - A chain-parallel pipeline that computes all chains together at each depth.
- Config switches:
  - `lmf_impl: "v6" | "v7"`
  - `lmf_chain_impl: "serial" | "parallel"`

These changes were integrated into the main EchMo path instead of remaining as isolated drafts.

## 3. Architectural Update

### 3.1 Serial Path

Serial LMF still uses `LMF_Chain`, but layer construction is now configurable:

- `lmf_impl = "v6"` -> instantiate `Learnable_Match_FilterV6`
- `lmf_impl = "v7"` -> instantiate `Learnable_Match_FilterV7`

This means the original chain structure is preserved while enabling an optimized layer backend.

### 3.2 Parallel Chain Path

The new `LMF_Chain_Parallel` computes all chains together at each layer depth:

$$
\text{input } [B, 1, P, R]
\rightarrow [B, K, 1, P, R]
\rightarrow [B, K, C, P, R]
\rightarrow [B, K \cdot C, P, R]
$$

Where:

- $K$ is the number of LMF chains
- $C$ is the LMF channel count per chain

This avoids serial Python dispatch over chain modules and makes chain-level parallelism explicit.

### 3.3 Why `LMF_Layer_Parallel` Exists

The parallel-chain implementation uses `LMF_Layer_Parallel` as the direct execution unit.

Important design point:

- `LMF_Chain` directly depends on `Learnable_Match_FilterV6` or `Learnable_Match_FilterV7`.
- `LMF_Chain_Parallel` directly depends on `LMF_Layer_Parallel`.
- `LMF_Layer_Parallel` is a neutral backend name, so the codebase keeps only one V7-named class.

This was chosen to preserve architectural clarity:

- one explicit V7 layer class only
- one neutral parallel backend for chain-parallel execution

## 4. Mathematical/Implementation Idea

### 4.1 V6 Complex Convolution

The original V6 layer computes complex convolution through four real convs:

$$
y_r = x_r * k_r - x_i * k_i
$$

$$
y_i = x_r * k_i + x_i * k_r
$$

implemented as four real-valued `conv2d` calls:

- `RR`
- `II`
- `RI`
- `IR`

### 4.2 V7 Packed-Conv Strategy

`Learnable_Match_FilterV7` reduces the four-conv form to one packed real convolution by concatenating real and imaginary inputs and building a packed kernel bank:

$$
\begin{bmatrix}
y_r \\
y_i
\end{bmatrix}
=
\mathrm{conv}
\left(
\begin{bmatrix}
x_r \\
x_i
\end{bmatrix},
\begin{bmatrix}
k_r & -k_i \\
k_i &  k_r
\end{bmatrix}
\right)
$$

This preserves the same complex-convolution mapping while greatly reducing kernel-launch overhead.

### 4.3 Parallel Layer Strategy

`LMF_Layer_Parallel` extends the same packed-conv idea to all chains at once by using grouped convolution:

- each chain occupies one group
- each group processes its own packed complex input slice
- all chains run in a single grouped convolution launch per layer

This changes the execution model from:

- serial over chains
- serial over layers

to:

- parallel over chains
- serial only over depth

which matches the original intended semantics much more closely.

## 5. Training Semantics Compatibility

The refactor preserves the V6 learning semantics that matter for training:

- layer 0 is prior-guided
- deeper layers are residual-style correction blocks
- residual layers use zero initialization
- prior layers store `prior_kernel_real/imag`
- `residual_scale` remains available

As a result:

- `Learnable_Match_FilterV7` stays curriculum-compatible with the V6 training strategy
- curriculum discovery in `learners/model.py` was updated to include the new LMF classes

## 6. Integration Files

Main code integration points:

- `models/echmo_debug.py`
  - added `Learnable_Match_FilterV7`
  - added `LMF_Layer_Parallel`
  - added `LMF_Chain_Parallel`
  - added `lmf_impl` selection in `LMF_Chain`
  - added `lmf_chain_impl` selection in `EchMo`
- `models/echmo_wide.py`
  - added chain-parallel support
- `run_echmo.py`
  - passes `lmf_impl` and `lmf_chain_impl`
- `learners/model.py`
  - updated curriculum class discovery

## 7. Config-Level Usage

Two axes are now configurable.

### 7.1 Layer Backend

- `lmf_impl = "v6"`
  - serial chain with V6 layer semantics and four-conv execution
- `lmf_impl = "v7"`
  - serial chain with packed-conv V7 layer execution

### 7.2 Chain Execution Style

- `lmf_chain_impl = "serial"`
  - instantiate one `LMF_Chain` per chain
- `lmf_chain_impl = "parallel"`
  - instantiate one `LMF_Chain_Parallel`

Example combinations:

- `v6 + serial`: original reference path
- `v7 + serial`: layer-optimized path
- `v7 + parallel`: layer + chain optimized path

Current ready-to-use config examples:

- `config_files/echmo_itc_scale_lmf7.json`
- `config_files/echmo_itc_scale_lmf7_chain_parallel.json`

## 8. Verified Numerical Equivalence

The new implementations were checked by copying aligned parameters from serial `LMF_Chain(v6)` and comparing outputs on the same input.

Reference: `LMF_Chain(v6)` output.

| Comparison | max abs err | mean abs err | relative mean abs err |
|---|---:|---:|---:|
| V6 serial vs V7 serial | `2.139e-4` | `1.390e-5` | `1.026e-6` |
| V6 serial vs Chain-Parallel | `3.158e-2` | `3.955e-3` | `2.926e-4` |

Interpretation:

1. `Learnable_Match_FilterV7` is numerically almost identical to V6 after parameter alignment.
2. `LMF_Chain_Parallel` is not bitwise-identical, but still very close in practical relative error.
3. The remaining difference is consistent with operator reordering and fused packed/grouped convolution behavior.

## 9. Latency Impact

Measured under the same EchMo scale setting used in the compute report:

| Mode | LMF-only latency (ms/iter) | End-to-end latency (ms/iter) |
|---|---:|---:|
| `LMF_Chain` + `LMF_V6` (serial) | `294.995` | `336.194` |
| `LMF_Chain` + `LMF_V7` (serial) | `78.589` | `115.732` |
| `LMF_Chain_Parallel` | `48.228` | `85.706` |

Speedup summary:

- `LMF_V7` vs `LMF_V6` (LMF-only): `3.75x`
- `LMF_Chain_Parallel` vs `LMF_V6` (LMF-only): `6.12x`
- `LMF_Chain_Parallel` vs `LMF_V7` (LMF-only): `1.63x`

End-to-end speedup summary:

- `LMF_V7` vs `LMF_V6`: `2.90x`
- `LMF_Chain_Parallel` vs `LMF_V6`: `3.92x`
- `LMF_Chain_Parallel` vs `LMF_V7`: `1.35x`

## 10. Practical Conclusion

This LMF refactor provides two distinct optimization levels:

1. `Learnable_Match_FilterV7`
   - best choice when the goal is to keep the original serial chain structure while cutting layer latency sharply
2. `LMF_Chain_Parallel`
   - best choice when the goal is to match the intended multi-chain parallel semantics and further reduce frontend latency

From an engineering perspective:

- `LMF_V7` is the low-risk optimization path
- `LMF_Chain_Parallel` is the highest-throughput path
- both preserve the current training-oriented V6 semantics closely enough for practical deployment and further research

## 11. Recommended Next Step

If tighter numerical agreement is needed while keeping most of the V7 speed advantage, the next step should be a dedicated alignment mode study:

- stricter deterministic evaluation settings
- fixed full-FP32 comparison path
- optional compromise backend between pure V6 and packed-conv V7

That work should be documented separately from this structural modification report.