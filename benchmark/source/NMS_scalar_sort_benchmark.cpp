#include "benchmark.hpp"

// Standalone Scalar Odd-Even Transposition Sort (Descending)
static void BenchmarkScalarOddEvenSort(int* order, float* scores, int n)
{
    for (int pass = 0; pass < n; ++pass)
    {
        bool swapped = false;
        for (int i = 0; i < n - 1; i += 2)
        {
            if (scores[i] < scores[i + 1])
            {
                float fs = scores[i]; scores[i] = scores[i + 1]; scores[i + 1] = fs;
                int   is = order[i];  order[i]  = order[i + 1];  order[i + 1]  = is;
                swapped = true;
            }
        }
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

// Standalone Scalar Bitonic Sort Network (Descending)
static void BenchmarkScalarBitonicSort(int* order, float* scores, int n)
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
    for (int k = 2; k <= p; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1)
            for (int i = 0; i < p; ++i)
            {
                int l = i ^ j;
                if (l <= i) continue;
                bool desc = ((i & k) == 0);
                bool swap = desc ? (ps[i] < ps[l]) : (ps[i] > ps[l]);
                if (swap)
                {
                    float fs = ps[i]; ps[i] = ps[l]; ps[l] = fs;
                    int ix = pi[i]; pi[i] = pi[l]; pi[l] = ix;
                }
            }
    for (int i = 0; i < n; ++i) { scores[i] = ps[i]; order[i] = pi[i]; }
    delete[] pi;
    delete[] ps;
}

// Benchmark: Scalar Odd-Even Sort vs. Scalar Bitonic Sort
int main()
{
#ifdef RISCV_BAREMETAL
    riscv_enable_vector();
    const int sort_iters = 3;
#else
    const int sort_iters = 50;
#endif

    const int n_sort = 64;

    printf("--- Benchmark: Scalar Odd-Even vs. Bitonic Sort (%d candidates, %d iterations) ---\n", n_sort, sort_iters);

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
        for (int i = 0; i < n_sort; ++i)
        {
            sc_oe[i] = sc_orig[i];
            sc_bi[i] = sc_orig[i];
            ord_oe[i] = i;
            ord_bi[i] = i;
        }

        Timer t_oe, t_bi;

        t_oe.Start();
        BenchmarkScalarOddEvenSort(ord_oe, sc_oe, n_sort);
        t_oe.Stop();

        t_bi.Start();
        BenchmarkScalarBitonicSort(ord_bi, sc_bi, n_sort);
        t_bi.Stop();

        odd_even_cycles += t_oe.ElapsedCycles();
        odd_even_instrs += t_oe.ElapsedInstructions();
        bitonic_cycles  += t_bi.ElapsedCycles();
        bitonic_instrs  += t_bi.ElapsedInstructions();
    }

    PrintTime("Scalar Sort: Odd-Even vs. Bitonic (64 candidates)",
              odd_even_cycles / static_cast<uint64_t>(sort_iters),
              bitonic_cycles  / static_cast<uint64_t>(sort_iters),
              odd_even_instrs / static_cast<uint64_t>(sort_iters),
              bitonic_instrs  / static_cast<uint64_t>(sort_iters));

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
