// vec::NonMaxSuppression — RISC-V Vector (RVV) implementation.
//
// Structure follows the illustrated walkthrough at:
//   https://riscv-illustration.vercel.app/nms/walkthrough
//
// The four stages, in order, exactly as shown in the illustration:
//
//   Stage 1 — Score filter
//     Input : class_scores[spatial_dim], score_threshold
//     Output: order[n], packed_scores[n]  (n = survivors after threshold)
//     How   : strip-mine with SetVl, load scores, compare >= threshold,
//             compress indices and scores into contiguous arrays.
//
//   Stage 2 — Odd-even sort   (even phase first, then odd phase)
//     Input : packed_scores[n], order[n]
//     Output: packed_scores[n] descending, order[n] reordered to match
//     How   : each iteration = one even phase + one odd phase;
//             even phase: compare-swap pairs (0,1),(2,3),(4,5),...
//             odd  phase: compare-swap pairs (1,2),(3,4),(5,6),...
//             swap condition: left_score < right_score  (want descending).
//             Early exit when a full iteration produces no swap.
//
//   Stage 3 — SoA corner pack
//     Input : batch_boxes[spatial_dim*4], order[n]
//     Output: soa_y1[n], soa_x1[n], soa_y2[n], soa_x2[n]
//     How   : for each sorted slot i, copy box corners
//             batch_boxes[order[i]*4 + {0,1,2,3}] → the four SoA arrays.
//             (Vectorized with SetVl + indexed / strided stores.)
//
//   Stage 4 — Greedy IoU suppress (scan + IoU)
//     Input : soa_y1/x1/y2/x2[n], order[n], iou_threshold
//     Output: selected_indices rows, suppressed[n] bitmap
//     How   : outer scalar loop over kept box i;
//             inner vector loop over later slots j > i:
//               load soa coords for j..j+vl-1 from the four SoA arrays,
//               compute IoU against anchor box i in vector lanes,
//               suppress where IoU > iou_threshold.
//
// Data layout (same as scalar reference):
//   boxes:  [num_batches, spatial_dim, 4]  float32  [y1, x1, y2, x2]
//   scores: [num_batches, num_classes, spatial_dim]  float32

#include "NMS.hpp"
#include "riscv_wrappers.hpp"

#include <cstddef>
#include <cstdint>

// ============================================================================
// Stage 1 — Score filter
// ============================================================================
// Packs survivors into  order[0..n-1]  and  packed_scores[0..n-1].
// Returns n.
// Matches illustration walkthrough "Score filter" stage:
//   order[] and packed_scores[] hold discovery-order box indices / scores.
//
static int VecScoreFilter(const float* class_scores,
                          int          spatial_dim,
                          float        threshold,
                          bool         use_threshold,
                          int*         order,
                          float*       packed_scores)
{
    int    n         = 0;
    size_t remaining = static_cast<size_t>(spatial_dim);
    size_t offset    = 0;

    while (remaining > 0)
    {
        const size_t vl = rvv::SetVlF32M4(remaining);

        // Load this strip of scores.
        rvv::F32M4 sc = rvv::LoadF32M4(class_scores + offset, vl);

        // Build absolute index vector for this strip: offset, offset+1, ...
        rvv::U32M4 iota    = rvv::IotaU32M4(vl);
        rvv::U32M4 abs_idx = rvv::AddU32M4(iota,
                                           static_cast<uint32_t>(offset), vl);

        if (use_threshold)
        {
            // keep mask: score >= threshold
            rvv::MaskF32M4 keep =
                rvv::MaskF32GreaterEqualScalarM4(sc, threshold, vl);

            // Compress indices and scores for survivors only.
            rvv::U32M4 packed_idx = rvv::CompressU32M4(abs_idx, keep, vl);
            rvv::F32M4 packed_sc  = rvv::CompressF32M4(sc,      keep, vl);

            const size_t cnt = rvv::CountTrueF32M4(keep, vl);

            rvv::StoreI32M4(order + n,
                            rvv::BitcastU32ToI32M4(packed_idx), cnt);
            rvv::StoreF32M4(packed_scores + n, packed_sc, cnt);

            n         += static_cast<int>(cnt);
        }
        else
        {
            // No threshold: every box passes.
            rvv::StoreI32M4(order + n,
                            rvv::BitcastU32ToI32M4(abs_idx), vl);
            rvv::StoreF32M4(packed_scores + n, sc, vl);

            n         += static_cast<int>(vl);
        }

        offset    += vl;
        remaining -= vl;
    }
    return n;
}

// ============================================================================
// Stage 2 — Odd-even sort   (even phase first, then odd phase)
// ============================================================================
// Sorts packed_scores[0..n-1] descending and keeps order[0..n-1] in sync.
//
// Exactly as the illustration "Sort" page shows:
//   Iteration k:
//     Even phase: compare-swap pairs (0,1),(2,3),(4,5),...
//     Odd  phase: compare-swap pairs (1,2),(3,4),(5,6),...
//   Swap condition: left_score < right_score  (enforce descending).
//   Early exit when no swap occurs in a full iteration.
//
static void VecOddEvenSort(int* order, float* packed_scores, int n)
{
    if (n <= 1)
        return;

    // K = n phases suffice to sort n elements.
    // We iterate up to n times but break early.
    for (int iter = 0; iter < n; ++iter)
    {
        bool any_swap = false;

        // Even phase first  (illustration: "Even pairs" go first)
        // Pairs: (0,1), (2,3), (4,5), ...   start = 0, step = 2
        for (int i = 0; i + 1 < n; i += 2)
        {
            // Swap condition: left_score < right_score → put higher score left.
            if (packed_scores[i] < packed_scores[i + 1])
            {
                float ftmp         = packed_scores[i];
                packed_scores[i]   = packed_scores[i + 1];
                packed_scores[i+1] = ftmp;

                int itmp    = order[i];
                order[i]    = order[i + 1];
                order[i+1]  = itmp;

                any_swap = true;
            }
        }

        // Odd phase second  (illustration: "Odd pairs" follow)
        // Pairs: (1,2), (3,4), (5,6), ...   start = 1, step = 2
        for (int i = 1; i + 1 < n; i += 2)
        {
            if (packed_scores[i] < packed_scores[i + 1])
            {
                float ftmp         = packed_scores[i];
                packed_scores[i]   = packed_scores[i + 1];
                packed_scores[i+1] = ftmp;

                int itmp    = order[i];
                order[i]    = order[i + 1];
                order[i+1]  = itmp;

                any_swap = true;
            }
        }

        if (!any_swap)
            break;   // Already sorted — early exit.
    }
}

// ============================================================================
// Stage 3 — SoA corner pack
// ============================================================================
// Packs box corners into four contiguous arrays aligned with order[]:
//   soa_y1[i] = batch_boxes[ order[i] * 4 + 0 ]
//   soa_x1[i] = batch_boxes[ order[i] * 4 + 1 ]
//   soa_y2[i] = batch_boxes[ order[i] * 4 + 2 ]
//   soa_x2[i] = batch_boxes[ order[i] * 4 + 3 ]
//
// This is the "SoA pack" stage from the illustration walkthrough (Greedy
// Vector, first half): one array per corner, same slot numbering as order[].
// The vectorized IoU scan (Stage 4) then loads contiguously from these arrays.
//
static void VecSoAPack(const float* batch_boxes,
                       const int*   order,
                       int          n,
                       float*       soa_y1,
                       float*       soa_x1,
                       float*       soa_y2,
                       float*       soa_x2)
{
    // We pack slot by slot. For each strip of vl slots we gather the four
    // coordinates. Because the wrappers expose contiguous loads, we use a
    // scalar gather into a temporary, then a contiguous vector store.
    // (A full indexed/strided gather would be equivalent; this keeps the
    //  code readable and matches the illustration's single-pass description.)

    int    slot      = 0;
    size_t remaining = static_cast<size_t>(n);

    while (remaining > 0)
    {
        const size_t vl = rvv::SetVlF32M2(remaining);

        // Scalar gather: collect coords for slots [slot .. slot+vl-1].
        // Temporary SoA buffers (VLMAX_M2 ≤ 256 for any realistic hardware).
        float tmp_y1[256], tmp_x1[256], tmp_y2[256], tmp_x2[256];
        for (size_t l = 0; l < vl; ++l)
        {
            const float* box = batch_boxes + order[slot + static_cast<int>(l)] * 4;
            tmp_y1[l] = box[0];
            tmp_x1[l] = box[1];
            tmp_y2[l] = box[2];
            tmp_x2[l] = box[3];
        }

        // Vector store each coordinate column into the SoA arrays.
        rvv::StoreF32M2(soa_y1 + slot, rvv::LoadF32M2(tmp_y1, vl), vl);
        rvv::StoreF32M2(soa_x1 + slot, rvv::LoadF32M2(tmp_x1, vl), vl);
        rvv::StoreF32M2(soa_y2 + slot, rvv::LoadF32M2(tmp_y2, vl), vl);
        rvv::StoreF32M2(soa_x2 + slot, rvv::LoadF32M2(tmp_x2, vl), vl);

        slot      += static_cast<int>(vl);
        remaining -= vl;
    }
}

// ============================================================================
// Stage 4 — Greedy IoU suppress  (scan + IoU)
// ============================================================================
// For one kept anchor box at sorted slot anchor_i, suppress all later slots
// j > anchor_i whose IoU with the anchor is strictly greater than iou_threshold.
//
// Uses the pre-packed SoA arrays from Stage 3, so every coordinate load is
// contiguous (matching the illustration's "scan + IoU" vector recipe):
//
//   overlap_y = max(0, min(vy2, ay2) - max(vy1, ay1))
//   overlap_x = max(0, min(vx2, ax2) - max(vx1, ax1))
//   inter     = overlap_y * overlap_x
//   area_b    = (vy2 - vy1) * (vx2 - vx1)
//   union     = area_a + area_b - inter
//   iou       = inter / union
//   suppress  where iou > iou_threshold
//
static void VecIouSuppressOne(const float* soa_y1,
                              const float* soa_x1,
                              const float* soa_y2,
                              const float* soa_x2,
                              bool*        suppressed,
                              int          n,
                              int          anchor_i,
                              float        iou_threshold)
{
    // Anchor box scalar coordinates (loaded once).
    const float ay1    = soa_y1[anchor_i];
    const float ax1    = soa_x1[anchor_i];
    const float ay2    = soa_y2[anchor_i];
    const float ax2    = soa_x2[anchor_i];
    const float area_a = (ay2 - ay1) * (ax2 - ax1);

    // Strip-mine over later slots j = anchor_i+1 .. n-1.
    int j = anchor_i + 1;
    while (j < n)
    {
        const size_t vl = rvv::SetVlF32M2(static_cast<size_t>(n - j));

        // Load candidate corner columns — contiguous because of SoA layout.
        rvv::F32M2 vy1 = rvv::LoadF32M2(soa_y1 + j, vl);
        rvv::F32M2 vx1 = rvv::LoadF32M2(soa_x1 + j, vl);
        rvv::F32M2 vy2 = rvv::LoadF32M2(soa_y2 + j, vl);
        rvv::F32M2 vx2 = rvv::LoadF32M2(soa_x2 + j, vl);

        // Intersection corners.
        rvv::F32M2 iy1 = rvv::MaxF32ScalarM2(vy1, ay1, vl);  // max(vy1, ay1)
        rvv::F32M2 ix1 = rvv::MaxF32ScalarM2(vx1, ax1, vl);  // max(vx1, ax1)
        rvv::F32M2 iy2 = rvv::MinF32ScalarM2(vy2, ay2, vl);  // min(vy2, ay2)
        rvv::F32M2 ix2 = rvv::MinF32ScalarM2(vx2, ax2, vl);  // min(vx2, ax2)

        // overlap_y = max(0, iy2 - iy1),  overlap_x = max(0, ix2 - ix1)
        rvv::F32M2 ov_y = rvv::MaxF32ScalarM2(rvv::SubF32M2(iy2, iy1, vl), 0.0f, vl);
        rvv::F32M2 ov_x = rvv::MaxF32ScalarM2(rvv::SubF32M2(ix2, ix1, vl), 0.0f, vl);

        // inter = overlap_y * overlap_x
        rvv::F32M2 inter = rvv::MulF32M2(ov_y, ov_x, vl);

        // area_b = (vy2 - vy1) * (vx2 - vx1)
        rvv::F32M2 area_b = rvv::MulF32M2(rvv::SubF32M2(vy2, vy1, vl),
                                           rvv::SubF32M2(vx2, vx1, vl), vl);

        // union = area_a + area_b - inter
        rvv::F32M2 union_v = rvv::AddF32ScalarM2(
                                 rvv::SubF32M2(area_b, inter, vl),
                                 area_a, vl);

        // iou = inter / union
        rvv::F32M2 iou = rvv::DivF32M2(inter, union_v, vl);

        // suppress where iou > iou_threshold
        rvv::MaskF32M2 supp_mask =
            rvv::MaskF32GreaterThanScalarM2(iou, iou_threshold, vl);

        // Write suppression flags: masked store of 1-bytes into supp_byte[],
        // then scatter into suppressed[].
        uint8_t supp_byte[256] = {};
        rvv::StoreU8Mf2Masked(supp_byte,
                              rvv::SplatU8Mf2(1, vl),
                              supp_mask, vl);

        for (size_t l = 0; l < vl; ++l)
        {
            if (supp_byte[l])
                suppressed[j + static_cast<int>(l)] = true;
        }

        j += static_cast<int>(vl);
    }
}

// ============================================================================
// Public entry point
// ============================================================================
int vec::NonMaxSuppression(const float*    boxes,
                           const float*    scores,
                           int             num_batches,
                           int             num_classes,
                           int             spatial_dim,
                           int64_t         max_output_boxes_per_class,
                           float           iou_threshold,
                           const float*    score_threshold,
                           CenterPointBox  center_point_box,
                           int64_t*        selected_indices)
{
    (void)center_point_box;  // always Corners; caller asserts

    if (num_batches <= 0 || num_classes <= 0 || spatial_dim <= 0 ||
        max_output_boxes_per_class <= 0)
        return 0;

    const bool  use_threshold = (score_threshold != nullptr);
    const float thr           = use_threshold ? *score_threshold : 0.0f;

    // Scratch buffers — reused for every (batch, class) pair.
    int*   order         = new int  [static_cast<size_t>(spatial_dim)];
    float* packed_scores = new float[static_cast<size_t>(spatial_dim)];
    bool*  suppressed    = new bool [static_cast<size_t>(spatial_dim)];
    // SoA corner arrays, same size as order[].
    float* soa_y1        = new float[static_cast<size_t>(spatial_dim)];
    float* soa_x1        = new float[static_cast<size_t>(spatial_dim)];
    float* soa_y2        = new float[static_cast<size_t>(spatial_dim)];
    float* soa_x2        = new float[static_cast<size_t>(spatial_dim)];

    int total_written = 0;

    for (int b = 0; b < num_batches; ++b)
    {
        const float* batch_boxes =
            boxes + static_cast<ptrdiff_t>(b) * spatial_dim * 4;

        for (int c = 0; c < num_classes; ++c)
        {
            const float* class_scores =
                scores +
                (static_cast<ptrdiff_t>(b) * num_classes + c) * spatial_dim;

            // ------------------------------------------------------------------
            // Stage 1 — Score filter
            // Packs surviving box indices into order[] and their scores into
            // packed_scores[], in original discovery order.
            // ------------------------------------------------------------------
            const int n = VecScoreFilter(class_scores, spatial_dim,
                                         thr, use_threshold,
                                         order, packed_scores);
            if (n == 0)
                continue;

            // ------------------------------------------------------------------
            // Stage 2 — Odd-even sort   (even phase first, then odd phase)
            // Sorts packed_scores[] descending and reorders order[] to match.
            // ------------------------------------------------------------------
            VecOddEvenSort(order, packed_scores, n);

            // ------------------------------------------------------------------
            // Stage 3 — SoA corner pack
            // Packs box corners into four contiguous arrays aligned with order[].
            // ------------------------------------------------------------------
            VecSoAPack(batch_boxes, order, n,
                       soa_y1, soa_x1, soa_y2, soa_x2);

            // ------------------------------------------------------------------
            // Stage 4 — Greedy IoU suppress  (scan + IoU)
            // Outer scalar loop keeps boxes; inner vector loop suppresses later
            // slots whose IoU with the kept box exceeds iou_threshold.
            // ------------------------------------------------------------------
            for (int i = 0; i < n; ++i)
                suppressed[i] = false;

            int kept = 0;
            for (int i = 0; i < n && kept < max_output_boxes_per_class; ++i)
            {
                if (suppressed[i])
                    continue;

                // Keep this box.
                selected_indices[(total_written + kept) * 3 + 0] =
                    static_cast<int64_t>(b);
                selected_indices[(total_written + kept) * 3 + 1] =
                    static_cast<int64_t>(c);
                selected_indices[(total_written + kept) * 3 + 2] =
                    static_cast<int64_t>(order[i]);
                ++kept;

                // Suppress all later slots with IoU > threshold.
                VecIouSuppressOne(soa_y1, soa_x1, soa_y2, soa_x2,
                                  suppressed, n, i, iou_threshold);
            }

            total_written += kept;
        }
    }

    delete[] soa_x2;
    delete[] soa_y2;
    delete[] soa_x1;
    delete[] soa_y1;
    delete[] suppressed;
    delete[] packed_scores;
    delete[] order;
    return total_written;
}
