#include "gtest_lite.hpp"

#include <cstdint>

#include "riscv_cv.hpp"
#include "reference_cv.hpp"
#include "test_utils.hpp"
#include "nms_onnx_cases.hpp"

namespace
{
    int OutputRows(const NMSOnnxCase& test_case)
    {
        const int64_t rows = static_cast<int64_t>(test_case.num_batches) *
            test_case.num_classes * test_case.max_output_boxes_per_class;
        return rows > 0 ? static_cast<int>(rows) : 1;
    }
}

TEST(NMSTest, ReferenceMatchesOnnxGolden)
{
    for (int i = 0; i < kNMSOnnxCaseCount; i++)
    {
        const NMSOnnxCase& test_case = kNMSOnnxCases[i];
        int64_t* actual = new int64_t[OutputRows(test_case) * 3];

        const int actual_count = ref::NonMaxSuppression(
            test_case.boxes,
            test_case.scores,
            test_case.num_batches,
            test_case.num_classes,
            test_case.spatial_dim,
            test_case.max_output_boxes_per_class,
            test_case.iou_threshold,
            test_case.score_threshold,
            test_case.center_point_box,
            actual);

        ExpectSelectedIndicesEqual(test_case.expected_indices, actual,
                                   test_case.expected_count, actual_count);
        delete[] actual;
    }
}

TEST(NMSTest, VectorizedMatchesOnnxGolden)
{
    for (int i = 0; i < kNMSOnnxCaseCount; i++)
    {
        const NMSOnnxCase& test_case = kNMSOnnxCases[i];
        int64_t* actual = new int64_t[OutputRows(test_case) * 3];

        const int actual_count = vec::NonMaxSuppression(
            test_case.boxes,
            test_case.scores,
            test_case.num_batches,
            test_case.num_classes,
            test_case.spatial_dim,
            test_case.max_output_boxes_per_class,
            test_case.iou_threshold,
            test_case.score_threshold,
            test_case.center_point_box,
            actual);

        ExpectSelectedIndicesEqual(test_case.expected_indices, actual,
                                   test_case.expected_count, actual_count);
        delete[] actual;
    }
}

TEST(NMSTest, VectorizedMatchesReference)
{
    for (int i = 0; i < kNMSOnnxCaseCount; i++)
    {
        const NMSOnnxCase& test_case = kNMSOnnxCases[i];
        int64_t* expected = new int64_t[OutputRows(test_case) * 3];
        int64_t* actual = new int64_t[OutputRows(test_case) * 3];

        const int expected_count = ref::NonMaxSuppression(
            test_case.boxes,
            test_case.scores,
            test_case.num_batches,
            test_case.num_classes,
            test_case.spatial_dim,
            test_case.max_output_boxes_per_class,
            test_case.iou_threshold,
            test_case.score_threshold,
            test_case.center_point_box,
            expected);
        const int actual_count = vec::NonMaxSuppression(
            test_case.boxes,
            test_case.scores,
            test_case.num_batches,
            test_case.num_classes,
            test_case.spatial_dim,
            test_case.max_output_boxes_per_class,
            test_case.iou_threshold,
            test_case.score_threshold,
            test_case.center_point_box,
            actual);

        ExpectSelectedIndicesEqual(expected, actual, expected_count, actual_count);
        delete[] actual;
        delete[] expected;
    }
}

// Larger than a typical e32m4/e32m2 VL so compress, odd-even, and IoU tails run.

TEST(NMSTest, ScalarBitonicMatchesOnnxGolden)
{
    for (int i = 0; i < kNMSOnnxCaseCount; i++)
    {
        const NMSOnnxCase& test_case = kNMSOnnxCases[i];
        int64_t* actual = new int64_t[OutputRows(test_case) * 3];
        const int actual_count = ref::NonMaxSuppressionBitonic(
            test_case.boxes, test_case.scores, test_case.num_batches,
            test_case.num_classes, test_case.spatial_dim,
            test_case.max_output_boxes_per_class, test_case.iou_threshold,
            test_case.score_threshold, test_case.center_point_box, actual);
        ExpectSelectedIndicesEqual(test_case.expected_indices, actual,
                                   test_case.expected_count, actual_count);
        delete[] actual;
    }
}

TEST(NMSTest, VectorizedBitonicMatchesOnnxGolden)
{
    for (int i = 0; i < kNMSOnnxCaseCount; i++)
    {
        const NMSOnnxCase& test_case = kNMSOnnxCases[i];
        int64_t* actual = new int64_t[OutputRows(test_case) * 3];
        const int actual_count = vec::NonMaxSuppressionBitonic(
            test_case.boxes, test_case.scores, test_case.num_batches,
            test_case.num_classes, test_case.spatial_dim,
            test_case.max_output_boxes_per_class, test_case.iou_threshold,
            test_case.score_threshold, test_case.center_point_box, actual);
        ExpectSelectedIndicesEqual(test_case.expected_indices, actual,
                                   test_case.expected_count, actual_count);
        delete[] actual;
    }
}

TEST(NMSTest, VectorizedMatchesReferenceManyBoxes)
{
    const int box_count = 80;
    float* boxes = new float[static_cast<size_t>(box_count) * 4];
    float* scores = new float[box_count];
    for (int i = 0; i < box_count; i++)
    {
        const float origin = static_cast<float>(i * 10);
        boxes[i * 4 + 0] = origin;
        boxes[i * 4 + 1] = 0.0f;
        boxes[i * 4 + 2] = origin + 1.0f;
        boxes[i * 4 + 3] = 1.0f;
        scores[i] = 1.0f - 0.001f * static_cast<float>(i);
    }
    const float score_threshold = 0.0f;
    int64_t* expected = new int64_t[static_cast<size_t>(box_count) * 3];
    int64_t* actual = new int64_t[static_cast<size_t>(box_count) * 3];
    int64_t* actual_bitonic = new int64_t[static_cast<size_t>(box_count) * 3];

    const int expected_count = ref::NonMaxSuppression(
        boxes, scores, 1, 1, box_count, box_count, 0.5f, &score_threshold,
        CenterPointBox::Corners, expected);
    const int actual_count = vec::NonMaxSuppression(
        boxes, scores, 1, 1, box_count, box_count, 0.5f, &score_threshold,
        CenterPointBox::Corners, actual);
    const int actual_bitonic_count = vec::NonMaxSuppressionBitonic(
        boxes, scores, 1, 1, box_count, box_count, 0.5f, &score_threshold,
        CenterPointBox::Corners, actual_bitonic);

    ExpectSelectedIndicesEqual(expected, actual, expected_count, actual_count);
    ExpectSelectedIndicesEqual(expected, actual_bitonic, expected_count, actual_bitonic_count);

    for (int i = 0; i < box_count; i++)
    {
        boxes[i * 4 + 0] = 0.0f;
        boxes[i * 4 + 1] = 0.0f;
        boxes[i * 4 + 2] = 1.0f;
        boxes[i * 4 + 3] = 1.0f;
        scores[i] = 0.9f;
    }
    const int expected_identical = ref::NonMaxSuppression(
        boxes, scores, 1, 1, box_count, 5, 0.5f, &score_threshold,
        CenterPointBox::Corners, expected);
    const int actual_identical = vec::NonMaxSuppression(
        boxes, scores, 1, 1, box_count, 5, 0.5f, &score_threshold,
        CenterPointBox::Corners, actual);
    const int actual_identical_bitonic = vec::NonMaxSuppressionBitonic(
        boxes, scores, 1, 1, box_count, 5, 0.5f, &score_threshold,
        CenterPointBox::Corners, actual_bitonic);
    ExpectSelectedIndicesEqual(expected, actual, expected_identical, actual_identical);
    ExpectSelectedIndicesEqual(expected, actual_bitonic, expected_identical, actual_identical_bitonic);

    delete[] actual_bitonic;
    delete[] actual;
    delete[] expected;
    delete[] scores;
    delete[] boxes;
}
