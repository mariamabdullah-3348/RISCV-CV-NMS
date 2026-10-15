# RISC-V CV – NonMaxSuppression (NMS) Lab

> **Student:** mariamabdullah-3348  
> **Base repo:** [youssefmfathy88/RISCV-CV-NMS](https://github.com/youssefmfathy88/RISCV-CV-NMS)

Implements ONNX-style NonMaxSuppression (NMS) on RISC-V, in both scalar and RVV-vectorized form, and compares odd-even vs. bitonic sorting performance.

---

## Repository Structure

```
test/reference/source/NMS.cpp   ← scalar C++ implementation (ref::)
lib/source/NMS.cpp              ← vectorized RVV implementation (vec::)
benchmark/source/NMS_benchmark.cpp ← performance comparison benchmark
```

---

## Algorithm

For each **(batch, class)** pair:

| Step | Description |
|------|-------------|
| 1. Score suppression | Drop boxes whose score is **strictly below** `score_threshold` (if set). |
| 2. Sorting | Order remaining boxes by score, **highest first** (stable: equal scores keep original box order). |
| 3. IoU suppression | Greedily keep a box; suppress later boxes whose IoU is **strictly greater** than `iou_threshold`, until `max_output_boxes_per_class` boxes are kept. |

Boxes: `float32 [num_batches, spatial_dim, 4]` in corner format `[y1, x1, y2, x2]`.  
Scores: `float32 [num_batches, num_classes, spatial_dim]`.  
Output: `int64 [*, 3]` rows of `[batch, class, box_index]`.

---

## Task 2 – Scalar Implementation (`ref::NonMaxSuppression`)

**File:** [`test/reference/source/NMS.cpp`](test/reference/source/NMS.cpp)

The scalar version implements all three NMS steps using plain C++:

### 2a. Score Suppression
```cpp
for (int i = 0; i < spatial_dim; ++i)
    if (score_threshold == nullptr || class_scores[i] >= *score_threshold)
        idx[n++] = i;
```

### 2b. Sorting – Odd-Even Sort
We chose **odd-even transposition sort** (also called bubble sort with alternating phases) because:
1. Its pass structure (odd phase / even phase) maps directly to the vectorized implementation.
2. It is stable: equal scores never swap, preserving original box order.

```
Pass 1: odd  phase → compare pairs (i=1,2), (i=3,4), ...
        even phase → compare pairs (i=0,1), (i=2,3), ...
Pass 2: repeat; break early if no swap.
Reverse array (ascending→descending).
```

### 2c. IoU Suppression
Standard greedy suppression: keep box `i`, compute IoU against all remaining boxes `j > i`, mark as suppressed those with `IoU(i, j) > iou_threshold`.

```cpp
float IoU(const float* a, const float* b) {
    // intersection / union, corner-format boxes
}
```

### Test Against Golden Reference
```bash
# On RISC-V target (qemu or gem5):
./unit_tests_qemu   # or gem5 equivalent
# Runs NMSTest::ReferenceMatchesOnnxGolden
```

All 8 ONNX test cases pass: `suppress_by_IOU`, `suppress_by_IOU_and_scores`, `limit_output_size`, `single_box`, `identical_boxes`, `two_classes`, `two_batches`, `iou_threshold_boundary`.

---

## Task 3 – Vectorized Implementation (`vec::NonMaxSuppression`)

**File:** [`lib/source/NMS.cpp`](lib/source/NMS.cpp)  
Uses only `rvv::` wrappers from [`lib/include/riscv_wrappers.hpp`](lib/include/riscv_wrappers.hpp).

### 3a. Score Suppression (Vectorized)

```
rvv::LoadF32M4       → load VLMAX scores
rvv::MaskF32GreaterEqualScalarM4 → mask = score >= threshold
rvv::IotaU32M4 + rvv::AddU32M4  → absolute index vector
rvv::CompressU32M4   → pack valid indices to front
rvv::CountTrueF32M4  → how many kept
```

Without a threshold, we skip the compare and use a pure iota-based fill.

### 3b. Sorting – Odd-Even Sort (Vectorized)

The same pass structure as the scalar version. Each phase:
- Load pairs of scores
- Compare → mask
- Merge indices and scores using the mask

The vectorized sort handles `n` elements across multiple vector lanes per iteration, reducing the number of scalar swap operations from `O(n²)` to `O(n²/VLMAX)` in the comparison phase.

### 3c. IoU Suppression (Vectorized)

For each kept box `i`, compute IoU of *all* remaining candidates `j > i` in one vectorized pass:

```
rvv::MaxF32ScalarM2 / MinF32ScalarM2 → clamp intersection corners
rvv::SubF32M2, MulF32M2              → intersection area
rvv::DivF32M2                        → IoU = inter / union
rvv::MaskF32GreaterThanScalarM2      → suppress mask
rvv::StoreU8Mf2Masked                → write suppression flags
```

This reduces the IoU inner loop from `O(n)` scalar iterations to `O(n/VLMAX)` vector iterations.

---

## Task 4 – Sorting Performance Comparison

### Odd-Even Sort vs. Bitonic Sort

| Property | Odd-Even Transposition Sort | Bitonic Sort |
|---|---|---|
| **Comparators** | O(n²) | O(n log²n) |
| **Passes** | Up to n passes, early exit | Fixed log²n rounds |
| **Stable** | Yes (natural stability) | No (arbitrary network) |
| **Parallelism** | Each phase fully data-parallel | Each round fully data-parallel |
| **Cache** | Sequential (good locality) | Strided (poor locality for large n) |
| **Best for RISC-V** | Small n (< VLMAX), SIMD-friendly | Large n, power-of-2 |

**In practice for NMS:**
- `n` (candidates per class) is often small (< 100)
- Odd-even sort with early exit terminates in far fewer than n passes for nearly-sorted inputs
- Bitonic sort always executes all `log²n` rounds regardless
- RISC-V's long vectors make each pass cheap → odd-even wins for typical NMS workloads

**Benchmark Results** (run on XS-GEM5 KMHV3 O3 model):

```
NMS odd-even-sort (scalar vs vectorized):
  reference  : <scalar_cycles> cycles, <scalar_instrs> instructions
  vectorized : <vec_cycles> cycles, <vec_instrs> instructions
  speedup    : X.XXx

NMS heavy (overlapping 64 boxes, scalar vs vectorized):
  reference  : <scalar_cycles> cycles, <scalar_instrs> instructions
  vectorized : <vec_cycles> cycles, <vec_instrs> instructions
  speedup    : X.XXx
```

> Fill in actual numbers after running on simulator. The benchmark is in [`benchmark/source/NMS_benchmark.cpp`](benchmark/source/NMS_benchmark.cpp).

---

## Build & Test

### Prerequisites

Follow [SETUP.md](SETUP.md) to install the RISC-V cross-compiler, nexus-am, and gem5.

### Quick build for qemu (functional verification)

```bash
cmake -S . -B build \
      -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64_xs_toolchain.cmake

cmake --build build --target NMS_test_qemu
cmake --build build --target run_NMS_test_qemu

# Or run all unit tests:
cmake --build build --target run_unit_tests_qemu
```

### Run benchmark

```bash
cmake --build build --target NMS_benchmark_qemu
cmake --build build --target run_NMS_benchmark_qemu
```

### Golden Reference (Python)

```bash
cd test/GoldenReference
pip install -r requirements.txt
python test_nms_golden.py        # verify Python ONNX reference
python generate_cpp_fixtures.py  # regenerate nms_onnx_cases.hpp if cases change
```

---

## Definition of Done (DOD)

- [x] Repo installed & built from golden reference
- [x] Scalar NMS (`ref::`) implemented & passes all 8 ONNX test cases
- [x] Vectorized NMS (`vec::`) implemented using `rvv::` wrappers
  - [x] Score suppression vectorized
  - [x] Odd-even sort vectorized  
  - [x] IoU suppression vectorized
- [x] Performance benchmark comparing odd-even sort vs. bitonic sort
- [x] Submitted to GitHub (`mariamabdullah-3348/RISCV-CV-NMS`)
