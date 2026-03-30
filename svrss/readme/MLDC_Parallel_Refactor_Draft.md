# MLDC_Parallel Refactor Draft

## 1. Motivation

Current `Multi_Order_LDC` runs per-order loops and launches two convs per order (real/imag), which increases launch overhead and memory traffic.

`MLDC_Parallel` aims to preserve mathematical behavior while reducing conv launch count.

## 2. Key Idea

- Original order-$n$ kernel has height $k_n = n+2$.
- Use a common kernel height $k_{max}=orders+1$.
- Top-pad each order kernel to `k_max`, concatenate into one weight bank.
- Apply one real conv + one imag conv over a once-padded input.

So launches change from `2 * orders` convs to `2` convs.

## 3. Equivalence Sketch

Original output for order-$n$:
$$
y_n(t) = \sum_{j=0}^{k_n-1} w_n(j)\,x_n(t+j)
$$

Parallel version embeds $w_n$ into $\tilde{w}_n$ with top-padding offset
$$
\Delta_n = k_{max} - k_n
$$
so $\tilde{w}_n(\Delta_n + j)=w_n(j)$, others are zero, with a shared top-padded input.
This keeps the same valid convolution window and output length.

## 4. Implementation Status

- Added class: `MLDC_Parallel` in `models/echmo_debug.py`
- Current state: draft class available as an **independent implementation** (no inheritance from `Multi_Order_LDC`), not yet switched into `EchMo` by default.

## 5. Integration Plan

1. Add config switch (e.g., `mldc_impl: "serial" | "parallel"`).
2. In EchMo init, instantiate `MLDC_Parallel` when switch is `parallel`.
3. Run A/B validation:
   - output consistency (`max_abs_err`, `mean_abs_err`)
   - latency comparison
   - training stability on short run

## 6. Risks

- Kernel packing is rebuilt each forward in current draft; may still have overhead.
- For maximum speedup, next step is to cache packed kernels or redesign params directly in packed form.
- Numerical differences can appear due to operation ordering.

## 7. Suggested Next Benchmark

- Same input and seeds on GPU:
  - `Multi_Order_LDC` vs `MLDC_Parallel`
- Report:
  - per-stage latency
  - end-to-end throughput
  - output difference statistics
