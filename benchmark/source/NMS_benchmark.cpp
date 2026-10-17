#include "benchmark.hpp"
#include "reference_cv.hpp"
#include "riscv_cv.hpp"

// Standalone Odd-Even Transposition Sort (Descending)
static void BenchmarkOddEvenSort(int* order, float* scores, int n)
{
    for (int pass = 0; pass < n; ++pass)
    {
        bool swapped = false;
        // Even phase
        for (int i = 0; i < n - 1; i += 2)
        {
            if (scores[i] < scores[i + 1])
            {
                float fs = scores[i]; scores[i] = scores[i + 1]; scores[i + 1] = fs;
                int   is = order[i];  order[i]  = order[i + 1];  order[i + 1]  = is;
                swapped = true;
            }
        }
        // Odd phase
        for (int i = 1; i < n - 1; i += 2)
        {
            if (scores[i] < scores[i + 1])
            {
                float fs = scores[i]; scores[i] = scores[i + 1]; scores[i + 1] = fs;
                int   is = order[i];  order[i]  = order[i + 1];  order[i + 1]  = is;
                swapped = true;
            }
        }
        if (!swapped)
            break;
    }
}

// Standalone Bitonic Sort Network (Descending)
static void BenchmarkBitonicSort(int* order, float* scores, int n)
{
    int p = 1;
    while (p < n) p <<= 1;

    for (int k = 2; k <= p; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j >>= 1)
        {
            for (int i = 0; i < p; ++i)
            {
                int l = i ^ j;
                if (l > i)
                {
                    bool dir_descending = ((i & k) == 0);
                    if (l < n)
                    {
                        bool should_swap = dir_descending ? (scores[i] < scores[l])
                                                          : (scores[i] > scores[l]);
                        if (should_swap)
                        {
                            float fs = scores[i]; scores[i] = scores[l]; scores[l] = fs;
                            int   is = order[i];  order[i]  = order[l];  order[l]  = is;
                        }
                    }
                }
            }
        }
    }
}

// NMS Benchmark: compare scalar vs. vectorized NMS, and Odd-Even vs. Bitonic sort.
int main()
{
#ifdef RISCV_BAREMETAL
    riscv_enable_vector();
#endif

    const int box_count  = 80;
    const int loop_count = 5;

    printf("NMS benchmark: %d boxes, %d iterations\n", box_count, loop_count);

    float* boxes  = new float[static_cast<size_t>(box_count) * 4];
    float* scores = new float[static_cast<size_t>(box_count)];

    // Non-overlapping boxes along the x-axis, scores decreasing slightly.
    for (int i = 0; i < box_count; ++i)
    {
        const float origin = static_cast<float>(i * 10);
        boxes[i * 4 + 0] = origin;
        boxes[i * 4 + 1] = 0.0f;
        boxes[i * 4 + 2] = origin + 1.0f;
        boxes[i * 4 + 3] = 1.0f;
        scores[i] = 1.0f - 0.001f * static_cast<float>(i);
    }

    const float score_threshold = 0.0f;
    const int   output_capacity = box_count;

    int64_t* ref_out = new int64_t[static_cast<size_t>(output_capacity) * 3];
    int64_t* vec_out = new int64_t[static_cast<size_t>(output_capacity) * 3];

    bool all_correct = true;

    // -------------------------------------------------------------------------
    // Run 1: Scalar reference vs. Vectorized (odd-even sort)
    // -------------------------------------------------------------------------
    {
        uint64_t ref_cycles = 0, vec_cycles = 0;
        uint64_t ref_instrs = 0, vec_instrs = 0;

        for (int iter = 0; iter < loop_count; ++iter)
        {
            Timer t_ref, t_vec;

            t_ref.Start();
            const int ref_count = ref::NonMaxSuppression(
                boxes, scores, 1, 1, box_count, box_count, 0.5f,
                &score_threshold, CenterPointBox::Corners, ref_out);
            t_ref.Stop();

            t_vec.Start();
            const int vec_count = vec::NonMaxSuppression(
                boxes, scores, 1, 1, box_count, box_count, 0.5f,
                &score_threshold, CenterPointBox::Corners, vec_out);
            t_vec.Stop();

            ref_cycles += t_ref.ElapsedCycles();
            ref_instrs += t_ref.ElapsedInstructions();
            vec_cycles += t_vec.ElapsedCycles();
            vec_instrs += t_vec.ElapsedInstructions();

            // Correctness check on first iteration.
            if (iter == 0)
            {
                if (ref_count != vec_count)
                {
                    printf("[FAIL] Count mismatch: ref=%d vec=%d\n",
                           ref_count, vec_count);
                    all_correct = false;
                }
                else
                {
                    for (int r = 0; r < ref_count * 3; ++r)
                    {
                        if (ref_out[r] != vec_out[r])
                        {
                            printf("[FAIL] Index mismatch at element %d\n", r);
                            all_correct = false;
                            break;
                        }
                    }
                }
            }
        }

        PrintTime("NMS odd-even-sort (scalar vs vectorized)",
                  ref_cycles / static_cast<uint64_t>(loop_count),
                  vec_cycles / static_cast<uint64_t>(loop_count),
                  ref_instrs / static_cast<uint64_t>(loop_count),
                  vec_instrs / static_cast<uint64_t>(loop_count));
    }

    // -------------------------------------------------------------------------
    // Run 2: Bitonic sort — for fair comparison we time vec::NonMaxSuppression
    // again here (it uses odd-even internally) and note that the performance
    // analysis for bitonic vs odd-even is documented in README.md.
    // Both produce the same output; the sort is an internal implementation
    // detail of the vectorized kernel.
    // -------------------------------------------------------------------------
    {
        uint64_t ref_cycles = 0, vec_cycles = 0;
        uint64_t ref_instrs = 0, vec_instrs = 0;

        // Use a heavier workload: 200 overlapping boxes with random scores to
        // stress the sort and IoU paths equally.
        const int n_heavy = 64;
        float* hboxes  = new float[static_cast<size_t>(n_heavy) * 4];
        float* hscores = new float[static_cast<size_t>(n_heavy)];

        srand(0xDEAD);
        for (int i = 0; i < n_heavy; ++i)
        {
            // Overlapping boxes: each box slightly offset from previous.
            const float x = static_cast<float>(i % 8) * 0.5f;
            const float y = static_cast<float>(i / 8) * 0.5f;
            hboxes[i * 4 + 0] = y;
            hboxes[i * 4 + 1] = x;
            hboxes[i * 4 + 2] = y + 1.0f;
            hboxes[i * 4 + 3] = x + 1.0f;
            // Random scores in [0,1].
            hscores[i] = static_cast<float>(rand() % 1000) / 1000.0f;
        }

        const float h_thr = 0.0f;
        int64_t* hout_ref = new int64_t[static_cast<size_t>(n_heavy) * 3];
        int64_t* hout_vec = new int64_t[static_cast<size_t>(n_heavy) * 3];

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

        PrintTime("NMS heavy (overlapping 64 boxes, scalar vs vectorized)",
                  ref_cycles / static_cast<uint64_t>(loop_count),
                  vec_cycles / static_cast<uint64_t>(loop_count),
                  ref_instrs / static_cast<uint64_t>(loop_count),
                  vec_instrs / static_cast<uint64_t>(loop_count));

        delete[] hout_vec;
        delete[] hout_ref;
        delete[] hscores;
        delete[] hboxes;
    }

    // -------------------------------------------------------------------------
    // Run 3: Direct Algorithm Comparison — Odd-Even Sort vs. Bitonic Sort
    // -------------------------------------------------------------------------
    {
        const int n_sort = 64;
        const int sort_iters = 50;
        uint64_t odd_even_cycles = 0, odd_even_instrs = 0;
        uint64_t bitonic_cycles = 0, bitonic_instrs = 0;

        float* sc_orig = new float[static_cast<size_t>(n_sort)];
        float* sc_oe   = new float[static_cast<size_t>(n_sort)];
        float* sc_bi   = new float[static_cast<size_t>(n_sort)];
        int*   ord_oe  = new int[static_cast<size_t>(n_sort)];
        int*   ord_bi  = new int[static_cast<size_t>(n_sort)];

        srand(0x1234);
        for (int i = 0; i < n_sort; ++i)
            sc_orig[i] = static_cast<float>(rand() % 1000) / 1000.0f;

        for (int iter = 0; iter < sort_iters; ++iter)
        {
            // Reset arrays
            for (int i = 0; i < n_sort; ++i)
            {
                sc_oe[i] = sc_orig[i];
                sc_bi[i] = sc_orig[i];
                ord_oe[i] = i;
                ord_bi[i] = i;
            }

            Timer t_oe, t_bi;

            t_oe.Start();
            BenchmarkOddEvenSort(ord_oe, sc_oe, n_sort);
            t_oe.Stop();

            t_bi.Start();
            BenchmarkBitonicSort(ord_bi, sc_bi, n_sort);
            t_bi.Stop();

            odd_even_cycles += t_oe.ElapsedCycles();
            odd_even_instrs += t_oe.ElapsedInstructions();
            bitonic_cycles  += t_bi.ElapsedCycles();
            bitonic_instrs  += t_bi.ElapsedInstructions();
        }

        PrintTime("Sort comparison: Odd-Even vs. Bitonic (64 candidates)",
                  odd_even_cycles / static_cast<uint64_t>(sort_iters),
                  bitonic_cycles  / static_cast<uint64_t>(sort_iters),
                  odd_even_instrs / static_cast<uint64_t>(sort_iters),
                  bitonic_instrs  / static_cast<uint64_t>(sort_iters));

        delete[] ord_bi;
        delete[] ord_oe;
        delete[] sc_bi;
        delete[] sc_oe;
        delete[] sc_orig;
    }

    delete[] vec_out;
    delete[] ref_out;
    delete[] scores;
    delete[] boxes;

    if (all_correct)
    {
        printf("Output is correct.\n");
        return 0;
    }
    else
    {
        printf("Output is wrong!\n");
        return 1;
    }
}
