#define NMS_CASE_SCALAR_ODD_EVEN
#include "benchmark.hpp"
#include "reference_cv.hpp"
#include "riscv_cv.hpp"

#if !defined(NMS_CASE_SCALAR_ODD_EVEN) && !defined(NMS_CASE_SCALAR_BITONIC) && !defined(NMS_CASE_RVV_ODD_EVEN) && !defined(NMS_CASE_RVV_BITONIC)
#error "Define exactly one NMS_CASE_* macro for this benchmark"
#endif

static bool SameRows(const int64_t* a, int na, const int64_t* b, int nb)
{
    if (na != nb) return false;
    for (int i = 0; i < na * 3; ++i) if (a[i] != b[i]) return false;
    return true;
}

int main()
{
    riscv_enable_vector();
    const int iterations = 5;
    const int sizes[] = {17, 63, 64, 65, 80};
    const float score_threshold = 0.0f;
#if defined(NMS_CASE_SCALAR_ODD_EVEN)
    const char* case_name = "scalar_odd_even";
#elif defined(NMS_CASE_SCALAR_BITONIC)
    const char* case_name = "scalar_bitonic";
#elif defined(NMS_CASE_RVV_ODD_EVEN)
    const char* case_name = "rvv_odd_even";
#else
    const char* case_name = "rvv_bitonic";
#endif
    bool all_correct = true;
    printf("NMS_CASE=%s\n", case_name);
    printf("iterations=%d; each case runs in its own gem5 process\n", iterations);

    for (unsigned si = 0; si < sizeof(sizes) / sizeof(sizes[0]); ++si) {
        const int n = sizes[si];
        float* boxes = new float[static_cast<size_t>(n) * 4];
        float* scores = new float[static_cast<size_t>(n)];
        int64_t* output = new int64_t[static_cast<size_t>(n) * 3];
        int64_t* expected = new int64_t[static_cast<size_t>(n) * 3];
        for (int i = 0; i < n; ++i) {
            const float x = static_cast<float>(i) * 2.0f;
            boxes[i*4] = 0.0f; boxes[i*4+1] = x;
            boxes[i*4+2] = 1.0f; boxes[i*4+3] = x + 1.0f;
            scores[i] = static_cast<float>((i * 37) % n) / static_cast<float>(n);
        }
        const int expected_n = ref::NonMaxSuppression(boxes, scores, 1, 1, n, n,
            0.5f, &score_threshold, CenterPointBox::Corners, expected);
        // Correctness is checked before timing. Timing is only for this case.
        int got_n = 0;
#if defined(NMS_CASE_SCALAR_ODD_EVEN)
        got_n = ref::NonMaxSuppression(boxes, scores, 1, 1, n, n, 0.5f, &score_threshold, CenterPointBox::Corners, output);
#elif defined(NMS_CASE_SCALAR_BITONIC)
        got_n = ref::NonMaxSuppressionBitonic(boxes, scores, 1, 1, n, n, 0.5f, &score_threshold, CenterPointBox::Corners, output);
#elif defined(NMS_CASE_RVV_ODD_EVEN)
        got_n = vec::NonMaxSuppression(boxes, scores, 1, 1, n, n, 0.5f, &score_threshold, CenterPointBox::Corners, output);
#else
        got_n = vec::NonMaxSuppressionBitonic(boxes, scores, 1, 1, n, n, 0.5f, &score_threshold, CenterPointBox::Corners, output);
#endif
        bool correct = SameRows(expected, expected_n, output, got_n);
        all_correct = all_correct && correct;
        printf("n=%d correctness=%s\n", n, correct ? "PASS" : "FAIL");
        if (!correct) { delete[] expected; delete[] output; delete[] scores; delete[] boxes; continue; }
        uint64_t cycles = 0, instructions = 0;
        for (int it = 0; it < iterations; ++it) {
            Timer timer; timer.Start();
#if defined(NMS_CASE_SCALAR_ODD_EVEN)
            got_n = ref::NonMaxSuppression(boxes, scores, 1, 1, n, n, 0.5f, &score_threshold, CenterPointBox::Corners, output);
#elif defined(NMS_CASE_SCALAR_BITONIC)
            got_n = ref::NonMaxSuppressionBitonic(boxes, scores, 1, 1, n, n, 0.5f, &score_threshold, CenterPointBox::Corners, output);
#elif defined(NMS_CASE_RVV_ODD_EVEN)
            got_n = vec::NonMaxSuppression(boxes, scores, 1, 1, n, n, 0.5f, &score_threshold, CenterPointBox::Corners, output);
#else
            got_n = vec::NonMaxSuppressionBitonic(boxes, scores, 1, 1, n, n, 0.5f, &score_threshold, CenterPointBox::Corners, output);
#endif
            timer.Stop(); cycles += timer.ElapsedCycles(); instructions += timer.ElapsedInstructions();
        }
        printf("n=%d average_cycles=%llu average_instructions=%llu\n", n,
            (unsigned long long)(cycles / iterations), (unsigned long long)(instructions / iterations));
        delete[] expected; delete[] output; delete[] scores; delete[] boxes;
    }
    if (!all_correct) { printf("Output is wrong!\n"); return 1; }
    printf("Output is correct.\n");
    return 0;
}
