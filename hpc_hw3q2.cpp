#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#ifdef PARALLEL
#include <semaphore>
#endif

auto GCount{0};

#ifdef PARALLEL
std::binary_semaphore GSmph(0);
#endif

auto count(int Start, int End) {
  auto Count{0};
  for (auto I{Start}; I < End; ++I) {
    if (I % 3 == 0 || I % 7 == 0)
      Count++;
  }
#ifdef PARALLEL
  GSmph.acquire();
  GCount += Count;
  GSmph.release();
#else

  GCount += Count;
#endif
  return;
}

int main() {

  auto NThread{std::thread::hardware_concurrency()};
  auto Interval = 10000 / NThread;

  std::vector<std::thread> ThreadPool;

  for (int I = 0; I < NThread; ++I) {
    auto Start = I * Interval;
    auto End = (I + 1) * Interval;
#ifdef PARALLEL
    GSmph.release();
    ThreadPool.push_back(std::thread(count, Start, End));
#else
    count(Start, End);
#endif
  }

#ifdef PARALLEL
  for (auto &Th : ThreadPool) {
    Th.join();
  }
#endif

  std::cout << GCount << std::endl;
  return 0;
}