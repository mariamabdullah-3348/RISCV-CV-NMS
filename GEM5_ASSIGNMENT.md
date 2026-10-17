# NMS assignment: separate gem5 runs

Each configuration is a separate bare-metal ELF and a separate gem5 process. Do not time all four algorithms inside one process. Run each target independently so each gets its own gem5 output directory, log, and stats.txt. Compare results only after collecting all four outputs.

## Build (QEMU disabled)

```bash
cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64_xs_toolchain.cmake \
  -DNEXUS_AM_HOME="$HOME/nexus-am" \
  -DGEM5_HOME="$HOME/GEM5" \
  -DBUILD_QEMU_TARGETS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target NMS_scalar_odd_even_gem5 NMS_scalar_bitonic_gem5 NMS_rvv_odd_even_gem5 NMS_rvv_bitonic_gem5 -j"$(nproc)"
```

## Run each case separately

```bash
cmake --build build --target run_NMS_scalar_odd_even_gem5
cmake --build build --target run_NMS_scalar_bitonic_gem5
cmake --build build --target run_NMS_rvv_odd_even_gem5
cmake --build build --target run_NMS_rvv_bitonic_gem5
```

Each run writes a separate `build/benchmark/<target>.log` and `build/benchmark/<target>-m5out/stats.txt` (exact output path follows the configured CMake binary directory). Check that each log contains `Output is correct.` and no `Output is wrong!` before comparing.

## Comparison

Record average kernel cycles/instructions printed by each executable for candidate counts 17, 63, 64, 65, and 80. Also record whole-process gem5 stats from each run separately. Compare scalar vs RVV for each sorting algorithm, and odd-even vs bitonic for scalar and RVV. Do not compare raw total gem5 `simTicks` across runs unless the benchmark workload and simulator configuration are identical; use kernel cycle/instruction counts for the algorithm-level comparison and `stats.txt` for whole-program context.
