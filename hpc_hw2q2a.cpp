#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

auto sieveOfEratosthenes(std::vector<bool> &Prime, int First, int Last,
                         int Max) {
  for (int P = First; P <= Last; ++P) {
    if (Prime[P]) {
      for (int I = P * 2; I <= Max; I += P) {
        Prime[I] = 0;
      }
    }
  }
}

inline auto nanoseconds() {
  std::chrono::high_resolution_clock Clock;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             Clock.now().time_since_epoch())
      .count();
}

int main() {
  std::vector<int> NumThreads = {1, 2, 4, 8, 16, 32, 64};
  std::vector<int> MaxNums = {1 << 16, 1 << 17, 1 << 18,
                              1 << 19, 1 << 20, 1 << 21};
  auto NumIter{300};
  for (auto &MaxNum : MaxNums) {
    for (auto &NumThread : NumThreads) {
      std::vector<bool> Prime(MaxNum + 1);
      std::fill_n(Prime.begin(), MaxNum + 1, 1);
      Prime[0] = Prime[1] = 0;

      // Measure copy vector time
      auto CopyStart = nanoseconds();
      for (int I = 0; I < NumIter; ++I) {
        auto PrimeCopy = Prime;
      }
      auto CopyEnd = nanoseconds();
      std::cout << NumThread << "," << MaxNum << ","
                << (CopyEnd - CopyStart) * 1.0 / NumIter / 1e6 << ",";

      // Measure parllel execution time
      auto Start = nanoseconds();
      for (int I = 0; I < NumIter; ++I) {
        auto PrimeCopy = Prime;

        std::vector<std::thread> Threads;
        auto Interval = static_cast<int>(MaxNum / NumThread);

        for (int I = 0; I < NumThread; ++I) {
          auto First = I * Interval;
          auto Last = (I + 1) * Interval;
          Threads.push_back(std::thread(
              sieveOfEratosthenes, std::ref(PrimeCopy), First, Last, MaxNum));
        }
        for (auto &Th : Threads) {
          Th.join();
        }
      }
      auto End = nanoseconds();

      std::cout << (End - Start) * 1.0 / NumIter / 1e6 << "\n";
    }
  }
}