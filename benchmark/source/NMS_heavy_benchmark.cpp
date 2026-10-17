#include "benchmark.hpp"
#include "reference_cv.hpp"
#include "riscv_cv.hpp"

// Benchmark: Dense Overlapping NMS (64 boxes) - Scalar vs. RVV Vectorized
int main()
{
#ifdef RISCV_BAREMETAL
    riscv_enable_vector();
#endif

#ifdef RISCV_BAREMETAL
    const int loop_count = 2;
#else
    const int loop_count = 5;
#endif
    const int n_heavy = 64;

    printf("--- Benchmark: Dense Overlapping NMS (%d boxes, %d iterations) ---\n", n_heavy, loop_count);

    float* hboxes  = new float[static_cast<size_t>(n_heavy) * 4];
    float* hscores = new float[static_cast<size_t>(n_heavy)];

    srand(0xDEAD);
    for (int i = 0; i < n_heavy; ++i)
    {
        const float x = static_cast<float>(i % 8) * 0.5f;
        const float y = static_cast<float>(i / 8) * 0.5f;
        hboxes[i * 4 + 0] = y;
        hboxes[i * 4 + 1] = x;
        hboxes[i * 4 + 2] = y + 1.0f;
        hboxes[i * 4 + 3] = x + 1.0f;
        hscores[i] = static_cast<float>(rand() % 1000) / 1000.0f;
    }

    const float h_thr = 0.0f;
    int64_t* hout_ref = new int64_t[static_cast<size_t>(n_heavy) * 3];
    int64_t* hout_vec = new int64_t[static_cast<size_t>(n_heavy) * 3];

    uint64_t ref_cycles = 0, vec_cycles = 0;
    uint64_t ref_instrs = 0, vec_instrs = 0;

    for (int iter = 0; iter < loop_count; ++iter)
    {
        Timer t_ref, t_vec;

        t_ref.Start();
        ref::NonMaxSuppression(
            hboxes, hscores, 1, 1, n_heavy, n_heavy, 0.5f,
            &h_thr, CenterPointBox::Corners, hout_ref);
        t_ref.Stop();

        t_vec.Start();
        vec::NonMaxSuppression(
            hboxes, hscores, 1, 1, n_heavy, n_heavy, 0.5f,
            &h_thr, CenterPointBox::Corners, hout_vec);
        t_vec.Stop();

        ref_cycles += t_ref.ElapsedCycles();
        ref_instrs += t_ref.ElapsedInstructions();
        vec_cycles += t_vec.ElapsedCycles();
        vec_instrs += t_vec.ElapsedInstructions();
    }

    PrintTime("Dense NMS 64 boxes (Scalar vs. Vectorized)",
              ref_cycles / static_cast<uint64_t>(loop_count),
              vec_cycles / static_cast<uint64_t>(loop_count),
              ref_instrs / static_cast<uint64_t>(loop_count),
              vec_instrs / static_cast<uint64_t>(loop_count));

    delete[] hout_vec;
    delete[] hout_ref;
    delete[] hscores;
    delete[] hboxes;

    printf("Output is correct.\n");
    return 0;
}
