#include <cmath>
// #include <immintrin.h>
#include <iomanip>
#include <iostream>

#include "util.hpp"

template <typename T> T calculateF(T X, int NumIter) {
  T Sum{static_cast<T>(1.0) / X};
  for (int I = 0; I < NumIter; ++I) {
    T Val = std::pow(X, 1. * I) / std::tgamma(I + 1);
    Sum += Val;
  }
  return Sum;
}

int main(int argc, char *argv[]) {

  // __m256i First = _mm256_set_epi32(10, 20, 30, 40, 50, 60, 70, 80);
  // __m256i Second = _mm256_set_epi32(5, 5, 5, 5, 5, 5, 5, 5);
  // __m256i Result = _mm256_add_epi32(First, Second);

  {
    auto Start{nanoseconds()};
    for (auto I{0}; I < 1e6; ++I) {
      calculateF(10., 160);
    }
    auto End{nanoseconds()};
    std::cout << End - Start << std::endl;
  }

  {
    auto Start{nanoseconds()};
    for (auto I{0}; I < 1e6; ++I) {
      calculateF(10.f, 160);
    }
    auto End{nanoseconds()};
    std::cout << End - Start << std::endl;
  }

  std::cout << std::setprecision(32) << calculateF(10., 160) << std::endl;
  std::cout << std::setprecision(32) << calculateF(10.f, 160) << std::endl;

  return 0;
}