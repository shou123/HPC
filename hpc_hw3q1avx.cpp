#include <cmath>
#include <iomanip>
#include <iostream>

#ifdef AVX2
#include <immintrin.h>
#endif

template <typename T> T calculateF(T X, int NumIter) {
  T Sum{static_cast<T>(1.0) / X};
  for (int I = 0; I < NumIter; ++I) {
    T Val = std::pow(X, 1. * I) / std::tgamma(I + 1);
    Sum += Val;
  }
  return Sum;
}

int main(int argc, char *argv[]) {
#ifdef AVX2
  __m256i First = _mm256_set_epi32(10, 20, 30, 40, 50, 60, 70, 80);
  __m256i Second = _mm256_set_epi32(5, 5, 5, 5, 5, 5, 5, 5);
  __m256i Result = _mm256_add_epi32(First, Second);
#endif
  std::cout << std::setprecision(32) << calculateF(10., 160) << std::endl;
  std::cout << std::setprecision(32) << calculateF(10.f, 160) << std::endl;

  return 0;
}
