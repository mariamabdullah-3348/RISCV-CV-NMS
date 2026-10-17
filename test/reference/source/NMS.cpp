#include "NMS.hpp"

#include <cstddef>
#include <cstdint>

// Scalar educational NonMaxSuppression.
// Algorithm (ONNX NonMaxSuppression-11, corners-only):
//   For each batch and class:
//     1. Score suppression: drop boxes whose score < score_threshold (if set).
//     2. Sorting:          order remaining boxes by score, highest first
//                          (stable: equal scores keep original box order).
//     3. IoU suppression:  greedily keep a box; suppress any later box whose
//                          IoU with the kept box is *strictly greater* than
//                          iou_threshold.
//   Write each kept box as [batch, class, box_index] into selected_indices.
//   Return the number of rows written.

namespace
{
    // Compute Intersection-over-Union for two boxes in corner format [y1,x1,y2,x2].
    // Assumes y1 <= y2 and x1 <= x2.
    float IoU(const float* a, const float* b)
    {
        const float ay1 = a[0], ax1 = a[1], ay2 = a[2], ax2 = a[3];
        const float by1 = b[0], bx1 = b[1], by2 = b[2], bx2 = b[3];

        // Intersection rectangle
        const float iy1 = ay1 > by1 ? ay1 : by1;
        const float ix1 = ax1 > bx1 ? ax1 : bx1;
        const float iy2 = ay2 < by2 ? ay2 : by2;
        const float ix2 = ax2 < bx2 ? ax2 : bx2;

        const float iw = ix2 - ix1;
        const float ih = iy2 - iy1;
        if (iw <= 0.0f || ih <= 0.0f)
            return 0.0f;

        const float intersection = iw * ih;
        const float area_a = (ay2 - ay1) * (ax2 - ax1);
        const float area_b = (by2 - by1) * (bx2 - bx1);
        const float union_area = area_a + area_b - intersection;

        if (union_area <= 0.0f)
            return 0.0f;
        return intersection / union_area;
    }

    // Odd-even (bubble) sort: sort an index array by descending score.
    // Stable: equal scores preserve original (ascending) index order.
    // Uses the odd-even transposition sort (parallel bubble sort) pattern
    // so the vectorized version can use the same pass structure.
    void OddEvenSort(int* indices, const float* scores, int n)
    {
        // Each full pass = one odd phase + one even phase.
        // After n passes at most, the sequence is sorted.
        for (int pass = 0; pass < n; ++pass)
        {
            bool swapped = false;

            // Odd phase: compare pairs (1,2), (3,4), ...
            for (int i = 1; i < n - 1; i += 2)
            {
                if (scores[indices[i]] < scores[indices[i + 1]])
                {
                    const int tmp = indices[i];
                    indices[i] = indices[i + 1];
                    indices[i + 1] = tmp;
                    swapped = true;
                }
            }

            // Even phase: compare pairs (0,1), (2,3), ...
            for (int i = 0; i < n - 1; i += 2)
            {
                if (scores[indices[i]] < scores[indices[i + 1]])
                {
                    const int tmp = indices[i];
                    indices[i] = indices[i + 1];
                    indices[i + 1] = tmp;
                    swapped = true;
                }
            }

            if (!swapped)
                break;
        }
    }
    // Bitonic sort (descending order)
    void BitonicSort(int* indices, const float* scores, int n)
    {
        if (n <= 1)
            return;

        int p = 1;
        while (p < n) p <<= 1;
        float* padded_scores = new float[static_cast<size_t>(p)];
        int* padded_indices = new int[static_cast<size_t>(p)];
        for (int i = 0; i < p; ++i)
        {
            if (i < n)
            {
                padded_indices[i] = indices[i];
                padded_scores[i] = scores[indices[i]];
            }
            else
            {
                padded_indices[i] = -1;
                padded_scores[i] = -__builtin_inff();
            }
        }

        for (int k = 2; k <= p; k <<= 1)
        {
            for (int j = k >> 1; j > 0; j >>= 1)
            {
                for (int i = 0; i < p; ++i)
                {
                    const int l = i ^ j;
                    if (l > i)
                    {
                        const bool descending = ((i & k) == 0);
                        const bool equal_score = padded_scores[i] == padded_scores[l];
                        const bool swap = descending
                            ? (padded_scores[i] < padded_scores[l] ||
                               (equal_score && padded_indices[i] > padded_indices[l]))
                            : (padded_scores[i] > padded_scores[l] ||
                               (equal_score && padded_indices[i] < padded_indices[l]));
                        if (swap)
                        {
                            const float fs = padded_scores[i];
                            padded_scores[i] = padded_scores[l];
                            padded_scores[l] = fs;
                            const int ix = padded_indices[i];
                            padded_indices[i] = padded_indices[l];
                            padded_indices[l] = ix;
                        }
                    }
                }
            }
        }
        for (int i = 0; i < n; ++i)
            indices[i] = padded_indices[i];
        delete[] padded_indices;
        delete[] padded_scores;
    }

    int NonMaxSuppressionImpl(const float* boxes,
                              const float* scores,
                              int num_batches,
                              int num_classes,
                              int spatial_dim,
                              int64_t max_output_boxes_per_class,
                              float iou_threshold,
                              const float* score_threshold,
                              CenterPointBox center_point_box,
                              int64_t* selected_indices,
                              bool use_bitonic_sort)
    {
        (void)center_point_box;

        if (num_batches <= 0 || num_classes <= 0 || spatial_dim <= 0 ||
            max_output_boxes_per_class <= 0)
        {
            return 0;
        }

        int* idx = new int[static_cast<size_t>(spatial_dim)];
        bool* suppressed = new bool[static_cast<size_t>(spatial_dim)];
        int total_written = 0;

        for (int b = 0; b < num_batches; ++b)
        {
            const float* batch_boxes = boxes + static_cast<ptrdiff_t>(b) * spatial_dim * 4;

            for (int c = 0; c < num_classes; ++c)
            {
                const float* class_scores =
                    scores + (static_cast<ptrdiff_t>(b) * num_classes + c) * spatial_dim;

                int n = 0;
                for (int i = 0; i < spatial_dim; ++i)
                {
                    if (score_threshold == nullptr || class_scores[i] >= *score_threshold)
                    {
                        idx[n++] = i;
                    }
                }

                if (n == 0)
                    continue;

                if (use_bitonic_sort)
                {
                    BitonicSort(idx, class_scores, n);
                }
                else
                {
                    OddEvenSort(idx, class_scores, n);
                }

                for (int i = 0; i < n; ++i)
                    suppressed[i] = false;

                int kept = 0;
                for (int i = 0; i < n && kept < max_output_boxes_per_class; ++i)
                {
                    if (suppressed[i])
                        continue;

                    selected_indices[(total_written + kept) * 3 + 0] = static_cast<int64_t>(b);
                    selected_indices[(total_written + kept) * 3 + 1] = static_cast<int64_t>(c);
                    selected_indices[(total_written + kept) * 3 + 2] = static_cast<int64_t>(idx[i]);
                    ++kept;

                    const float* box_i = batch_boxes + idx[i] * 4;

                    for (int j = i + 1; j < n; ++j)
                    {
                        if (suppressed[j])
                            continue;
                        const float* box_j = batch_boxes + idx[j] * 4;
                        if (IoU(box_i, box_j) > iou_threshold)
                            suppressed[j] = true;
                    }
                }

                total_written += kept;
            }
        }

        delete[] suppressed;
        delete[] idx;
        return total_written;
    }
} // namespace

int ref::NonMaxSuppression(const float* boxes,
                           const float* scores,
                           int num_batches,
                           int num_classes,
                           int spatial_dim,
                           int64_t max_output_boxes_per_class,
                           float iou_threshold,
                           const float* score_threshold,
                           CenterPointBox center_point_box,
                           int64_t* selected_indices)
{
    return NonMaxSuppressionImpl(boxes, scores, num_batches, num_classes,
                                 spatial_dim, max_output_boxes_per_class,
                                 iou_threshold, score_threshold,
                                 center_point_box, selected_indices,
                                 /*use_bitonic_sort=*/false);
}

int ref::NonMaxSuppressionBitonic(const float* boxes,
                                  const float* scores,
                                  int num_batches,
                                  int num_classes,
                                  int spatial_dim,
                                  int64_t max_output_boxes_per_class,
                                  float iou_threshold,
                                  const float* score_threshold,
                                  CenterPointBox center_point_box,
                                  int64_t* selected_indices)
{
    return NonMaxSuppressionImpl(boxes, scores, num_batches, num_classes,
                                 spatial_dim, max_output_boxes_per_class,
                                 iou_threshold, score_threshold,
                                 center_point_box, selected_indices,
                                 /*use_bitonic_sort=*/true);
}
