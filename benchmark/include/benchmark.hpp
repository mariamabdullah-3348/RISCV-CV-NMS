#pragma once

#include "riscv_port.hpp"  // printf, srand, rand (klib on bare metal)
#include "riscv_cv.hpp"
#include "riscv_timer.hpp"


#include <cstdint>

// Print cycle/instruction counts and the speedup for a kernel pair.
inline void PrintTime(const char* name, uint64_t scalar_cycles,
                      uint64_t vectorized_cycles, uint64_t scalar_instrs,
                      uint64_t vectorized_instrs)
{
    printf("%s:\n", name);
    printf("  reference  : %llu cycles, %llu instructions\n",
           (unsigned long long)scalar_cycles, (unsigned long long)scalar_instrs);
    printf("  vectorized : %llu cycles, %llu instructions\n",
           (unsigned long long)vectorized_cycles,
           (unsigned long long)vectorized_instrs);
    if (vectorized_cycles > 0)
    {
        // Print speedup as x100 to avoid floating-point printf.
        uint64_t speedup_x100 = (scalar_cycles * 100) / vectorized_cycles;
        printf("  speedup    : %llu.%02llux\n",
               (unsigned long long)(speedup_x100 / 100),
               (unsigned long long)(speedup_x100 % 100));
    }
}


// Validate descending order and score/index association for standalone sort
// benchmarks. original_scores[id] is the score belonging to original index id.
inline bool ValidateSortedScores(const int* order, const float* sorted_scores,
                                 const float* original_scores, int n)
{
    for (int i = 0; i < n; ++i)
    {
        if (order[i] < 0 || order[i] >= n)
            return false;
        if (sorted_scores[i] != original_scores[order[i]])
            return false;
        if (i > 0 && sorted_scores[i - 1] < sorted_scores[i])
            return false;
    }
    return true;
}
