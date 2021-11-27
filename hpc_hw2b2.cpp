#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

auto sieveofEratosthenes(std::vector<bool> &Prime, int First, int Last,
                         int Max) {
  for (int P = First; P <= Last; P++) {
    if (Prime[P] == 1) {
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
  std::vector<int> NumThreads = {1, 2, 4, 8};
  std::vector<int> MaxNums = {100, 1000, 10000, 100000, 1000000};

  for (auto NumThread : NumThreads) {
    for (auto MaxNum : MaxNums) {

      auto Start = nanoseconds();
      for (int I = 0; I < 100; I++) {

        std::vector<std::thread> Threads;
        std::vector<bool> Prime(MaxNum + 1);
        std::fill_n(Prime.begin(), MaxNum + 1, 1);
        Prime[0] = Prime[1] = 0;
        auto Interval = MaxNum / NumThread;

        for (int I = 0; I < NumThread; I++) {
          auto First = I * Interval;
          auto Last = (I + 1) * Interval;
          Threads.push_back(std::thread(sieveofEratosthenes, std::ref(Prime),
                                        First, Last, MaxNum));
        }

        for (auto &Th : Threads) {
          Th.join();
        }
      }
      auto End = nanoseconds();
      std::cout << NumThread << ", " << MaxNum << ","
                << (End - Start) * 1.0 / 100 / 1e6 << "ms"
                << "\n";
    }
  }
}