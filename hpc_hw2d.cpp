#include <algorithm>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

auto crossOut(std::vector<bool> &Sieve, int Start, int End, int Max) {
  for (int I = Start; I < End; ++I) {
    if (Sieve[I]) {
      for (int J = I * 2u; J <= Max; J += I) {
        Sieve[J] = 0;
      }
    }
  }
}

int main() {
  auto Max{1000 * 1000 * 1000u};

  std::vector<bool> Sieve(Max + 1);
  std::fill_n(Sieve.begin(), Max + 1, 1);
  Sieve[0] = Sieve[1] = 0;

  auto NThread{std::thread::hardware_concurrency()};
  auto Interval = Max / NThread;
  
  std::vector<std::thread> ThreadPool;

  for (int I = 0; I < NThread; ++I) {
    auto Start = I * Interval;
    auto End = (I + 1) * Interval;
#ifdef PARALLEL
    ThreadPool.push_back(
        std::thread(crossOut, std::ref(Sieve), Start, End, Max));
#else
    crossOut(Sieve, Start, End, Max);
#endif
  }
#ifdef PARALLEL
  for (auto &Th : ThreadPool) {
    Th.join();
  }
#endif
  std::cout << std::accumulate(Sieve.begin(), Sieve.end(), 0) << std::endl;
}
