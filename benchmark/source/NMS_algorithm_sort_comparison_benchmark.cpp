#include "benchmark.hpp"
#include "reference_cv.hpp"
#include "riscv_cv.hpp"

// Benchmark: Full NMS Algorithm using Odd-Even Sort vs. Bitonic Sort
int main()
{
#ifdef RISCV_BAREMETAL
    riscv_enable_vector();
    const int loop_count = 2;
#else
    const int loop_count = 10;
#endif

    const int n_boxes = 64;

    printf("--- Benchmark: NMS Algorithm (Odd-Even Sort vs. Bitonic Sort, %d boxes) ---\n", n_boxes);

    float* boxes  = new float[static_cast<size_t>(n_boxes) * 4];
    float* scores = new float[static_cast<size_t>(n_boxes)];

    srand(0x1234);
    for (int i = 0; i < n_boxes; ++i)
    {
        const float x = static_cast<float>(i % 8) * 0.5f;
        const float y = static_cast<float>(i / 8) * 0.5f;
        boxes[i * 4 + 0] = y;
        boxes[i * 4 + 1] = x;
        boxes[i * 4 + 2] = y + 1.0f;
        boxes[i * 4 + 3] = x + 1.0f;
        // Unique scores avoid ambiguous tie ordering in the algorithm comparison.
        scores[i] = static_cast<float>((i * 37) % n_boxes) / static_cast<float>(n_boxes);
    }

    const float score_threshold = 0.0f;
    int64_t* out_oe = new int64_t[static_cast<size_t>(n_boxes) * 3];
    int64_t* out_bi = new int64_t[static_cast<size_t>(n_boxes) * 3];
    int64_t* expected_oe = new int64_t[static_cast<size_t>(n_boxes) * 3];
    int64_t* expected_bi = new int64_t[static_cast<size_t>(n_boxes) * 3];
    int oe_count = 0, bi_count = 0;

    uint64_t oe_cycles = 0, bi_cycles = 0;
    uint64_t oe_instrs = 0, bi_instrs = 0;

    for (int iter = 0; iter < loop_count; ++iter)
    {
        Timer t_oe, t_bi;

        t_oe.Start();
        oe_count = vec::NonMaxSuppression(
            boxes, scores, 1, 1, n_boxes, n_boxes, 0.5f,
            &score_threshold, CenterPointBox::Corners, out_oe);
        t_oe.Stop();

        t_bi.Start();
        bi_count = vec::NonMaxSuppressionBitonic(
            boxes, scores, 1, 1, n_boxes, n_boxes, 0.5f,
            &score_threshold, CenterPointBox::Corners, out_bi);
        t_bi.Stop();

        oe_cycles += t_oe.ElapsedCycles();
        oe_instrs += t_oe.ElapsedInstructions();
        bi_cycles += t_bi.ElapsedCycles();
        bi_instrs += t_bi.ElapsedInstructions();
    }

    PrintTime("NMS Algorithm: Odd-Even Sort vs. Bitonic Sort (64 boxes)",
              oe_cycles / static_cast<uint64_t>(loop_count),
              bi_cycles / static_cast<uint64_t>(loop_count),
              oe_instrs / static_cast<uint64_t>(loop_count),
              bi_instrs / static_cast<uint64_t>(loop_count));

    // Correctness is checked outside the timed region against the scalar golden
    // reference for each sort variant. Never report success unconditionally.
    const int expected_oe_count = ref::NonMaxSuppression(
        boxes, scores, 1, 1, n_boxes, n_boxes, 0.5f, &score_threshold,
        CenterPointBox::Corners, expected_oe);
    const int expected_bi_count = ref::NonMaxSuppressionBitonic(
        boxes, scores, 1, 1, n_boxes, n_boxes, 0.5f, &score_threshold,
        CenterPointBox::Corners, expected_bi);
    bool correct = (oe_count == expected_oe_count) && (bi_count == expected_bi_count);
    for (int i = 0; correct && i < oe_count * 3; ++i)
        correct = out_oe[i] == expected_oe[i];
    for (int i = 0; correct && i < bi_count * 3; ++i)
        correct = out_bi[i] == expected_bi[i];

    delete[] expected_bi;
    delete[] expected_oe;
    delete[] out_bi;
    delete[] out_oe;
    delete[] scores;
    delete[] boxes;

    if (!correct) { printf("Output is wrong!\n"); return 1; }
    printf("Output is correct.\n");
    return 0;
}
