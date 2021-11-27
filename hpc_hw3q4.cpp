#include <initializer_list>
#include <iostream>
#include <random>
#include <vector>

#ifdef OPENMP
#include <omp.h>
#endif

template <unsigned R, unsigned C, typename T> struct Matrix {
  T M[C * R];

  Matrix(std::initializer_list<T> L) {
    for (const auto *I{L.begin()}; I != L.end(); ++I) {
      M[I - L.begin()] = *I;
    }
  }

  Matrix(std::initializer_list<std::initializer_list<T>> L) {
    for (auto I{L.begin()}; I != L.end(); ++I) {
      for (const auto *J{I->begin()}; J != I->end(); ++J) {
        M[(I - L.begin()) * C + (J - I->begin())] = *J;
      }
    }
  }

  Matrix() = default;

  auto print() {
    for (auto I{0}; I < R; I++) {
      for (auto J{0}; J < C; J++) {
        std::cout << M[I * C + J] << " ";
      }
      std::cout << std::endl;
    }
  }

  auto populate() {
    std::random_device RndDevice;
    std::mt19937 MersenneEngine{RndDevice()};
    std::uniform_real_distribution<T> Dist;
    for (auto I{0}; I < R; I++) {
      for (auto J{0}; J < C; J++) {
        M[I * C + J] = Dist(MersenneEngine);
      }
    }
  }
};

template <unsigned A, unsigned B, unsigned C, typename T>
auto operator*(const Matrix<A, B, T> &Lhs, const Matrix<B, C, T> &Rhs) {
  Matrix<A, C, T> M;
#ifdef OPENMP
#pragma omp parallel for
#endif
  for (auto I = 0; I < C; ++I) {
    for (auto J = 0; J < A; ++J) {
      auto S = T();
      for (auto K = 0; K < B; ++K) {
        S += Lhs.M[B * J + K] * Rhs.M[C * K + I];
      }
      #ifdef OPENMP
      #pragma omp critical
      #endif
      M.M[J * C + I] = S;
    }
  }
  return M;
}

int main() {
  using T = double;
  constexpr auto Size{512};
  // clang-format off
  Matrix<2, 3, T> A = {{1., 2., 3.,}, {4., 5., 6.,}};
  // clang-format on

  Matrix<3, 6, T> B = {{1., 2., 3., 4., 5., 6.},
                       {7., 8., 9., 1., 2., 3.},
                       {4., 5., 6., 7., 8., 9.}};

  Matrix<6, 1, T> C = {1., 2., 3., 4., 5., 6.};

  auto D{A * B * C};
  D.print();

  for (auto I = 0; I < 1000; ++I) {
    Matrix<1, Size, T> E;
    Matrix<Size, Size, T> F;
    E.populate();
    F.populate();
    auto G{E * F};
  }

  return 0;
}
