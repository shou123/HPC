#include "fce_hw1.hpp"
#include <iostream>

using namespace std;

int main() {
  using T = int;
  auto NRepeat = 100;
  for (auto Size = 2; Size < 100; Size++) {
    std::cout << Size << '\t';

    auto Start = nanoseconds();
    for (auto Repeat = 0; Repeat < NRepeat; Repeat++) {
      std::vector<T> Vec(Size);
      populateRandomVector(Vec, 100);
      mergeSort(Vec, 0, Vec.size() - 1);
    }
    std::cout << (nanoseconds() - Start) / NRepeat << '\t';

    Start = nanoseconds();
    for (auto Repeat = 0; Repeat < NRepeat; Repeat++) {
      std::vector<T> Vec(Size);
      populateRandomVector(Vec, 100);
      quickSort(std::begin(Vec), std::end(Vec));
    }
    std::cout << (nanoseconds() - Start) / NRepeat << '\n';
  }

  return 0;
}
