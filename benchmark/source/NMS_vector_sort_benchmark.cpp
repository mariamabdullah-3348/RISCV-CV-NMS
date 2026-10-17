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
    float* ps = new float[static_cast<size_t>(p)];
    int* pi = new int[static_cast<size_t>(p)];
    for (int i = 0; i < p; ++i)
    {
        ps[i] = i < n ? scores[i] : -__builtin_inff();
        pi[i] = i < n ? order[i] : -1;
    }
    for (size_t k = 2; k <= static_cast<size_t>(p); k <<= 1)
        for (size_t j = k >> 1; j > 0; j >>= 1)
            for (size_t i = 0; i < static_cast<size_t>(p); i += 2 * j)
            {
                const bool desc = ((i & k) == 0);
                size_t rem = j, off = 0;
                while (rem > 0)
                {
                    size_t vl = rvv::SetVlF32M4(rem);
                    size_t a = i + off, b = a + j;
                    rvv::F32M4 as = rvv::LoadF32M4(ps + a, vl);
                    rvv::F32M4 bs = rvv::LoadF32M4(ps + b, vl);
                    rvv::MaskF32M4 m = desc ? rvv::MaskF32LessThanM4(as, bs, vl)
                                            : rvv::MaskF32LessThanM4(bs, as, vl);
                    rvv::StoreF32M4(ps + a, rvv::MergeF32M4(as, bs, m, vl), vl);
                    rvv::StoreF32M4(ps + b, rvv::MergeF32M4(bs, as, m, vl), vl);
                    rvv::I32M4 ai = rvv::LoadI32M4(pi + a, vl);
                    rvv::I32M4 bi = rvv::LoadI32M4(pi + b, vl);
                    rvv::StoreI32M4(pi + a, rvv::MergeI32M4(ai, bi, m, vl), vl);
                    rvv::StoreI32M4(pi + b, rvv::MergeI32M4(bi, ai, m, vl), vl);
                    off += vl; rem -= vl;
                }
            }
    for (int i = 0; i < n; ++i) { scores[i] = ps[i]; order[i] = pi[i]; }
    delete[] pi;
    delete[] ps;
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

    const bool sort_ok = ValidateSortedScores(ord_oe, sc_oe, sc_orig, n_sort) &&
                         ValidateSortedScores(ord_bi, sc_bi, sc_orig, n_sort);

    delete[] ord_bi;
    delete[] ord_oe;
    delete[] sc_bi;
    delete[] sc_oe;
    delete[] sc_orig;

    if (!sort_ok) { printf("Output is wrong!\n"); return 1; }
    printf("Output is correct.\n");
    return 0;
}
