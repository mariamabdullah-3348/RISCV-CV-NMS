#include "benchmark.hpp"
#include "reference_cv.hpp"
#include "riscv_cv.hpp"

// Benchmark: Standard NMS (80 boxes) - Scalar vs. RVV Vectorized
int main()
{
#ifdef RISCV_BAREMETAL
    riscv_enable_vector();
#endif

    const int box_count  = 80;
#ifdef RISCV_BAREMETAL
    const int loop_count = 2;
#else
    const int loop_count = 5;
#endif

    printf("--- Benchmark: Standard NMS (80 boxes, %d iterations) ---\n", loop_count);

    float* boxes  = new float[static_cast<size_t>(box_count) * 4];
    float* scores = new float[static_cast<size_t>(box_count)];

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

        if (iter == 0)
        {
            if (ref_count != vec_count)
            {
                printf("[FAIL] Count mismatch: ref=%d vec=%d\n", ref_count, vec_count);
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

    PrintTime("Standard NMS (Scalar vs. Vectorized)",
              ref_cycles / static_cast<uint64_t>(loop_count),
              vec_cycles / static_cast<uint64_t>(loop_count),
              ref_instrs / static_cast<uint64_t>(loop_count),
              vec_instrs / static_cast<uint64_t>(loop_count));

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
