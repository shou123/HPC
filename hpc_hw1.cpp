// #define MULTITHREAD

#include "hpc_hw1.hpp"

constexpr auto N = 1023;
constexpr auto M = 100;

int main() {

  std::vector<int> Vec(N);
  populateRandomVector(Vec, M);
  auto Output = redixSort(Vec, N, M);

  std::fill(Vec.begin(), Vec.end(), 0);
  populateRandomVector(Vec, M);
  mergeSort(Vec);

  return !(std::is_sorted(Output.begin(), Output.end()) &&
           std::is_sorted(Vec.begin(), Vec.end()));
}