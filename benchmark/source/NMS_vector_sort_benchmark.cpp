#include "benchmark.hpp"
#include "riscv_wrappers.hpp"

// Standalone Vectorized Odd-Even Transposition Sort (RVV)
static void BenchmarkVectorOddEvenSort(int* order, float* scores, int n)
{
    const ptrdiff_t stride_f = 2 * static_cast<ptrdiff_t>(sizeof(float));
    const ptrdiff_t stride_i = 2 * static_cast<ptrdiff_t>(sizeof(int));

    for (int pass = 0; pass < n; ++pass)
    {
        size_t total_swaps = 0;

        // Even phase: pairs (0,1), (2,3), ...
        size_t rem = static_cast<size_t>(n / 2);
        size_t off = 0;
        while (rem > 0)
        {
            size_t vl = rvv::SetVlF32M4(rem);
            rvv::F32M4 l_sc = rvv::LoadStridedF32M4(scores + off * 2, stride_f, vl);
            rvv::F32M4 r_sc = rvv::LoadStridedF32M4(scores + off * 2 + 1, stride_f, vl);
            rvv::MaskF32M4 swap_m = rvv::MaskF32LessThanM4(l_sc, r_sc, vl);

            size_t cnt = rvv::CountTrueF32M4(swap_m, vl);
            total_swaps += cnt;
            if (cnt > 0)
            {
                rvv::F32M4 new_l_sc = rvv::MergeF32M4(l_sc, r_sc, swap_m, vl);
                rvv::F32M4 new_r_sc = rvv::MergeF32M4(r_sc, l_sc, swap_m, vl);
                rvv::StoreStridedF32M4(scores + off * 2, stride_f, new_l_sc, vl);
                rvv::StoreStridedF32M4(scores + off * 2 + 1, stride_f, new_r_sc, vl);

                rvv::I32M4 l_id = rvv::LoadStridedI32M4(order + off * 2, stride_i, vl);
                rvv::I32M4 r_id = rvv::LoadStridedI32M4(order + off * 2 + 1, stride_i, vl);
                rvv::I32M4 new_l_id = rvv::MergeI32M4(l_id, r_id, swap_m, vl);
                rvv::I32M4 new_r_id = rvv::MergeI32M4(r_id, l_id, swap_m, vl);
                rvv::StoreStridedI32M4(order + off * 2, stride_i, new_l_id, vl);
                rvv::StoreStridedI32M4(order + off * 2 + 1, stride_i, new_r_id, vl);
            }
            off += vl;
            rem -= vl;
        }

        // Odd phase: pairs (1,2), (3,4), ...
        rem = static_cast<size_t>((n - 1) / 2);
        off = 0;
        while (rem > 0)
        {
            size_t vl = rvv::SetVlF32M4(rem);
            rvv::F32M4 l_sc = rvv::LoadStridedF32M4(scores + off * 2 + 1, stride_f, vl);
            rvv::F32M4 r_sc = rvv::LoadStridedF32M4(scores + off * 2 + 2, stride_f, vl);
            rvv::MaskF32M4 swap_m = rvv::MaskF32LessThanM4(l_sc, r_sc, vl);

            size_t cnt = rvv::CountTrueF32M4(swap_m, vl);
            total_swaps += cnt;
            if (cnt > 0)
            {
                rvv::F32M4 new_l_sc = rvv::MergeF32M4(l_sc, r_sc, swap_m, vl);
                rvv::F32M4 new_r_sc = rvv::MergeF32M4(r_sc, l_sc, swap_m, vl);
                rvv::StoreStridedF32M4(scores + off * 2 + 1, stride_f, new_l_sc, vl);
                rvv::StoreStridedF32M4(scores + off * 2 + 2, stride_f, new_r_sc, vl);

                rvv::I32M4 l_id = rvv::LoadStridedI32M4(order + off * 2 + 1, stride_i, vl);
                rvv::I32M4 r_id = rvv::LoadStridedI32M4(order + off * 2 + 2, stride_i, vl);
                rvv::I32M4 new_l_id = rvv::MergeI32M4(l_id, r_id, swap_m, vl);
                rvv::I32M4 new_r_id = rvv::MergeI32M4(r_id, l_id, swap_m, vl);
                rvv::StoreStridedI32M4(order + off * 2 + 1, stride_i, new_l_id, vl);
                rvv::StoreStridedI32M4(order + off * 2 + 2, stride_i, new_r_id, vl);
            }
            off += vl;
            rem -= vl;
        }

        if (total_swaps == 0)
            break;
    }
}

// Standalone Vectorized Bitonic Sort Network (RVV)
static void BenchmarkVectorBitonicSort(int* order, float* scores, int n)
{
    int p = 1;
    while (p < n) p <<= 1;

    for (int k = 2; k <= p; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j >>= 1)
        {
            for (int i = 0; i < p; i += 2 * j)
            {
                bool dir_desc = ((i & k) == 0);
                size_t rem = static_cast<size_t>(j);
                size_t off = 0;

                while (rem > 0)
                {
                    size_t vl = rvv::SetVlF32M4(rem);
                    size_t idx_a = static_cast<size_t>(i) + off;
                    size_t idx_b = idx_a + static_cast<size_t>(j);

                    if (idx_a < static_cast<size_t>(n) && idx_b < static_cast<size_t>(n))
                    {
                        rvv::F32M4 a_sc = rvv::LoadF32M4(scores + idx_a, vl);
                        rvv::F32M4 b_sc = rvv::LoadF32M4(scores + idx_b, vl);

                        rvv::MaskF32M4 swap_m = dir_desc
                            ? rvv::MaskF32LessThanM4(a_sc, b_sc, vl)
                            : rvv::MaskF32LessThanM4(b_sc, a_sc, vl);

                        rvv::F32M4 new_a_sc = rvv::MergeF32M4(a_sc, b_sc, swap_m, vl);
                        rvv::F32M4 new_b_sc = rvv::MergeF32M4(b_sc, a_sc, swap_m, vl);
                        rvv::StoreF32M4(scores + idx_a, new_a_sc, vl);
                        rvv::StoreF32M4(scores + idx_b, new_b_sc, vl);

                        rvv::I32M4 a_id = rvv::LoadI32M4(order + idx_a, vl);
                        rvv::I32M4 b_id = rvv::LoadI32M4(order + idx_b, vl);
                        rvv::I32M4 new_a_id = rvv::MergeI32M4(a_id, b_id, swap_m, vl);
                        rvv::I32M4 new_b_id = rvv::MergeI32M4(b_id, a_id, swap_m, vl);
                        rvv::StoreI32M4(order + idx_a, new_a_id, vl);
                        rvv::StoreI32M4(order + idx_b, new_b_id, vl);
                    }
                    off += vl;
                    rem -= vl;
                }
            }
        }
    }
}

// Benchmark: Vector Odd-Even Sort vs. Vector Bitonic Sort (RVV)
int main()
{
#ifdef RISCV_BAREMETAL
    riscv_enable_vector();
    const int sort_iters = 3;
#else
    const int sort_iters = 50;
#endif

    const int n_sort = 64;

    printf("--- Benchmark: Vectorized Odd-Even vs. Bitonic Sort (RVV, %d candidates, %d iterations) ---\n", n_sort, sort_iters);

    uint64_t vec_oe_cycles = 0, vec_oe_instrs = 0;
    uint64_t vec_bi_cycles = 0, vec_bi_instrs = 0;

    float* sc_orig = new float[static_cast<size_t>(n_sort)];
    float* sc_oe   = new float[static_cast<size_t>(n_sort)];
    float* sc_bi   = new float[static_cast<size_t>(n_sort)];
    int*   ord_oe  = new int[static_cast<size_t>(n_sort)];
    int*   ord_bi  = new int[static_cast<size_t>(n_sort)];

    srand(0x5678);
    for (int i = 0; i < n_sort; ++i)
        sc_orig[i] = static_cast<float>(rand() % 1000) / 1000.0f;

    for (int iter = 0; iter < sort_iters; ++iter)
    {
        for (int i = 0; i < n_sort; ++i)
        {
            sc_oe[i] = sc_orig[i];
            sc_bi[i] = sc_orig[i];
            ord_oe[i] = i;
            ord_bi[i] = i;
        }

        Timer t_oe, t_bi;

        t_oe.Start();
        BenchmarkVectorOddEvenSort(ord_oe, sc_oe, n_sort);
        t_oe.Stop();

        t_bi.Start();
        BenchmarkVectorBitonicSort(ord_bi, sc_bi, n_sort);
        t_bi.Stop();

        vec_oe_cycles += t_oe.ElapsedCycles();
        vec_oe_instrs += t_oe.ElapsedInstructions();
        vec_bi_cycles += t_bi.ElapsedCycles();
        vec_bi_instrs += t_bi.ElapsedInstructions();
    }

    PrintTime("Vector Sort (RVV): Odd-Even vs. Bitonic (64 candidates)",
              vec_oe_cycles / static_cast<uint64_t>(sort_iters),
              vec_bi_cycles / static_cast<uint64_t>(sort_iters),
              vec_oe_instrs / static_cast<uint64_t>(sort_iters),
              vec_bi_instrs / static_cast<uint64_t>(sort_iters));

    delete[] ord_bi;
    delete[] ord_oe;
    delete[] sc_bi;
    delete[] sc_oe;
    delete[] sc_orig;

    printf("Output is correct.\n");
    return 0;
}
