# RISC-V CV — Non-Maximum Suppression (NMS)

This project implements ONNX-style **Non-Maximum Suppression (NMS)** in scalar C++ and RISC-V Vector (RVV) code, and benchmarks two sorting strategies: odd-even transposition sort and bitonic sort.

The assignment-focused performance experiment runs four configurations as **four separate gem5 simulations**:

1. Scalar + odd-even sort
2. Scalar + bitonic sort
3. RVV + odd-even sort
4. RVV + bitonic sort

Each benchmark checks its selected indices against the scalar odd-even reference before timing the case. The benchmark prints `Output is correct.` only if all configured test sizes pass. This README describes how to build and run the experiment; performance results should be regenerated on the machine/configuration being evaluated rather than treated as universal numbers.

## NMS behavior

For each batch and class, the implementation:

1. Filters out boxes whose score is **strictly less than** `score_threshold` (a score equal to the threshold is retained).
2. Sorts the surviving candidates by descending score.
3. Greedily keeps a candidate and suppresses later candidates only when their IoU is **strictly greater than** `iou_threshold`.
4. Stops after `max_output_boxes_per_class` selections.

Input boxes use corner coordinates `[y1, x1, y2, x2]`; scores are arranged by batch, class, and candidate. Output rows contain `[batch_index, class_index, box_index]`. The current implementation path uses corner-format boxes (`CenterPointBox::Corners`).

## Implementation overview

| Component | Location | Purpose |
|---|---|---|
| Scalar reference | `test/reference/source/NMS.cpp` | Plain C++ NMS reference, including odd-even and bitonic entry points |
| RVV implementation | `lib/source/NMS.cpp` | Vectorized score filtering, sorting, and IoU suppression using project RVV wrappers |
| RVV wrappers | `lib/include/riscv_wrappers.hpp` | Wrapper layer for RVV operations |
| Unit tests | `test/unit_test/source/NMS_test.cpp` | Functional tests against the project's golden fixtures |
| Four-way benchmark sources | `benchmark/source/NMS_scalar_odd_even.cpp`, `NMS_scalar_bitonic.cpp`, `NMS_rvv_odd_even.cpp`, `NMS_rvv_bitonic.cpp` | One benchmark executable per configuration |
| Benchmark build rules | `benchmark/CMakeLists.txt` | Creates gem5 targets, and optional QEMU targets |
| gem5 assignment notes | `GEM5_ASSIGNMENT.md` | Four-run workflow and comparison guidance |
| Stats summarizer | `tools/summarize_gem5_stats.py` | Collects key statistics from the four gem5 `stats.txt` files |

## Requirements

The build expects:

- A RISC-V cross-compilation toolchain supported by this project.
- A built `nexus-am` checkout.
- A built XiangShan gem5 checkout containing `build/RISCV/gem5.opt` and `configs/example/kmhv3.py`.
- CMake and a host C/C++ build environment.

See [`SETUP.md`](SETUP.md) for environment setup. The paths can be supplied through `AM_HOME` and `XIANG_GEM5`, or directly through CMake options.

## Configure and build (gem5)

Run these commands from the repository root. Adjust the two paths if your checkouts live elsewhere:

```bash
export AM_HOME="$HOME/nexus-am"
export XIANG_GEM5="$HOME/GEM5"

cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64_xs_toolchain.cmake \
  -DNEXUS_AM_HOME="$AM_HOME" \
  -DGEM5_HOME="$XIANG_GEM5" \
  -DBUILD_QEMU_TARGETS=OFF \
  -DCMAKE_BUILD_TYPE=Release
```

Build the four executables:

```bash
cmake --build build --target \
  NMS_scalar_odd_even_gem5 \
  NMS_scalar_bitonic_gem5 \
  NMS_rvv_odd_even_gem5 \
  NMS_rvv_bitonic_gem5 \
  -j"$(nproc)"
```

QEMU targets are optional and are disabled in this gem5-focused workflow. To enable them, configure with `-DBUILD_QEMU_TARGETS=ON` and ensure `qemu-riscv64` is installed.

## Run each benchmark in its own gem5 process

Run each target separately; do not combine the four algorithms into one timed process:

```bash
cmake --build build --target run_NMS_scalar_odd_even_gem5
cmake --build build --target run_NMS_scalar_bitonic_gem5
cmake --build build --target run_NMS_rvv_odd_even_gem5
cmake --build build --target run_NMS_rvv_bitonic_gem5
```

Each run should create its own log and gem5 output directory under `build/benchmark/`. For example:

```text
build/benchmark/NMS_scalar_odd_even_gem5.log
build/benchmark/NMS_scalar_odd_even_gem5-m5out/stats.txt
```

The precise paths depend on the configured build directory and CMake run rules. Check each log for `Output is correct.` and make sure it does not contain `Output is wrong!`. If a run fails, investigate it before comparing performance.

The four-way benchmarks test candidate counts **17, 63, 64, 65, and 80**, with five timed iterations per size. The benchmark reports average kernel cycles and instructions for each size. These in-program measurements are more directly useful for comparing the algorithm implementations than whole-process gem5 statistics alone.

## Summarize gem5 statistics

After all four runs complete, execute:

```bash
python3 tools/summarize_gem5_stats.py
```

The script looks for the four expected `stats.txt` paths beneath `build/benchmark/` and generates:

- `gem5_summary.csv` — selected gem5 metrics for each run.
- `gem5_comparison.csv` — cycle-count ratios derived from `system.cpu.numCycles`.
- `gem5_summary.txt` beside each run's `stats.txt`.

These are generated, machine-specific artifacts and are ignored by Git by default. You can regenerate them after rerunning gem5. If a stats file is missing, the script reports that run as missing; run all four benchmarks before drawing conclusions.

### How to interpret the results

- Compare kernel `average_cycles` and `average_instructions` at the **same candidate count**.
- Compare scalar vs. RVV while holding the sorting algorithm constant.
- Compare odd-even vs. bitonic while holding scalar/RVV implementation constant.
- `simTicks`, `simSeconds`, `simInsts`, `numCycles`, IPC, and CPI describe the complete simulated program, including setup and correctness checks. Do not assume whole-program `simTicks` equals time spent inside the timed NMS kernel.
- Treat results as comparable only when workload, gem5 configuration, compiler settings, and build mode are the same.

## Tests and golden fixtures

The project includes unit tests and Python golden-reference utilities. See [`test/GoldenReference/README.md`](test/GoldenReference/README.md) for the fixture-generation workflow and requirements. Configure and build the test targets supported by your toolchain, then run the relevant test targets before accepting implementation changes.

## Build output and Git

The `build/` directory contains generated CMake files, object files, executables, logs, and gem5 output. **Do not commit it.** It is reproducible from the source and build instructions above, can be large, and often contains absolute paths specific to one machine. The repository `.gitignore` excludes it and the generated gem5 summary CSV/text files.

## Repository files

- [`GEM5_ASSIGNMENT.md`](GEM5_ASSIGNMENT.md) — assignment-specific run and comparison checklist.
- [`SETUP.md`](SETUP.md) — dependency and environment setup.
- [`IMAGEIO.md`](IMAGEIO.md) — image input/output utilities.
- [`illustration/README.md`](illustration/README.md) — interactive NMS illustration.
