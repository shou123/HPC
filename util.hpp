#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

template <typename T> void printVector(std::vector<T> Vec) {
  std::for_each(Vec.begin(), Vec.end(),
                [&](const auto I) { std::cout << I << ' '; });
  std::cout << '\n';
  // lambda expression: [](){};
}

inline auto nanoseconds() {
  std::chrono::high_resolution_clock Clock;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             Clock.now().time_since_epoch())
      .count();
}

inline auto populateRandomVector(std::vector<int> &Vec, int M) {
  std::random_device RndDevice;
  std::mt19937 MersenneEngine{RndDevice()};
  std::uniform_int_distribution<int> Dist{1, M};
  std::generate(Vec.begin(), Vec.end(),
                [&Dist, &MersenneEngine]() { return Dist(MersenneEngine); });
}

inline auto populateSequentialVector(std::vector<int> &Vec) {
  auto Counter{0};
  std::generate(Vec.begin(), Vec.end(), [&Counter]() { return ++Counter; });
}
