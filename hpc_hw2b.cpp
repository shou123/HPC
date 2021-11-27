#include <atomic>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>

template <typename Integer, typename T>
void strikeOutMultiples(Integer N, std::vector<T> &Vec) {
  for (Integer I = N * 2u; I < Vec.size(); I += N) {
    Vec[I] = false;
  }
}

template <typename Integer>
auto sieveEratosthenesSeq(Integer N) -> std::vector<Integer> {
  if (N < 2u) {
    return {};
  }

  std::vector<char> IsPrime(N + 1u, true);

  // Strike out the multiples of 2 so that
  // the following loop can be faster
  strikeOutMultiples(2u, IsPrime);

  // Strike out the multiples of the prime
  // number between 3 and end
  auto End = static_cast<Integer>(std::sqrt(N));
  for (Integer N = 3u; N <= End; N += 2u) {
    if (IsPrime[N]) {
      strikeOutMultiples(N, IsPrime);
    }
  }

  std::vector<Integer> Res = {2u};
  for (Integer I = 3u; I < IsPrime.size(); I += 2u) {
    if (IsPrime[I]) {
      Res.push_back(I);
    }
  }
  return Res;
}

template <typename Integer>
auto sieveEratosthenesPar(Integer N) -> std::vector<Integer> {
  if (N < 2u) {
    return {};
  }

  // Only the prime numbers <= sqrt(n) are
  // needed to find the other ones
  auto End = static_cast<Integer>(std::sqrt(N));
  // Find the primes numbers <= sqrt(n) thanks
  // to a sequential sieve of Eratosthenes
  const auto Primes = sieveEratosthenesSeq(End);

  std::vector<std::atomic<bool>> IsPrime(N + 1u);
  for (auto I = 0u; I < N + 1u; ++I) {
    IsPrime[I].store(true, std::memory_order_relaxed);
  }
  std::vector<std::thread> Threads;

  // Computes the number of primes numbers that will
  // be handled by each thread. This number depends on
  // the maximum number of concurrent threads allowed
  // by the implementation and on the total number of
  // elements in primes
  std::size_t NbPrimesPerThread = static_cast<std::size_t>(
      std::ceil(static_cast<float>(Primes.size()) /
                static_cast<float>(std::thread::hardware_concurrency())));

  for (std::size_t First = 0u; First < Primes.size();
       First += NbPrimesPerThread) {
    auto Last = std::min(First + NbPrimesPerThread, Primes.size());
    // Spawn a thread to strike out the multiples
    // of the prime numbers corresponding to the
    // elements of primes between first and last
    Threads.emplace_back(
        [&Primes, &IsPrime](Integer Begin, Integer End) {
          for (std::size_t I = Begin; I < End; ++I) {
            auto Prime = Primes[I];
            for (Integer N = Prime * 2u; N < IsPrime.size(); N += Prime) {
              IsPrime[N].store(false, std::memory_order_relaxed);
            }
          }
        },
        First, Last);
  }

  for (auto &Thr : Threads) {
    Thr.join();
  }

  std::vector<Integer> Res = {2u};
  for (Integer I = 3u; I < IsPrime.size(); I += 2u) {
    if (IsPrime[I].load(std::memory_order_relaxed)) {
      Res.push_back(I);
    }
  }
  return Res;
}

int main() {
  auto Primes = sieveEratosthenesPar(100u);
  std::for_each (Primes.begin(), Primes.end(), [&](int i) {std::cout << i << ' ';});
  std::cout << Primes.size() << std::endl;
  // std::cout << "\n";
}
