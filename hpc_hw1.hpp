#include "util.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

#ifndef MULTITHREAD
#define MULTITHREAD
#endif

std::mutex GMutex;

inline auto workerRedix(std::vector<int> &Bucket, std::vector<int> &Value) {
  // auto Pid = static_cast<std::thread::id>(std::this_thread::get_id());
  // std::cout << "PID: " << Pid << std::endl;
  // std::this_thread::sleep_for(std::chrono::seconds(5));
  std::lock_guard<std::mutex> Guard(GMutex);
  for (auto &I : Value) {
    Bucket[I]++;
  }
}

inline auto redixSort(std::vector<int> &Vec, int N, int M) {

  std::vector<int> Bucket(M + 1);
#ifdef MULTITHREAD
  std::vector<int> FirstHalf(Vec.begin(), Vec.begin() + (Vec.size() >> 1));
  std::vector<int> SecondHalf(Vec.begin() + (Vec.size() >> 1), Vec.end());

  std::thread Thread1(workerRedix, std::ref(Bucket), std::ref(FirstHalf));
  std::thread Thread2(workerRedix, std::ref(Bucket), std::ref(SecondHalf));

  Thread1.join();
  Thread2.join();
#else
  for (auto &I : Vec) {
    Bucket[I]++;
  }
#endif

  std::vector<int> Output(N);
  auto Counter{0};
  for (std::size_t I = 1; I <= M; ++I) {
    while (Bucket[I]--) {
      Output[Counter++] = I;
    }
  }
  return Output;
}

inline auto workerMerge(std::vector<int> &Vec) {
  std::sort(Vec.begin(), Vec.end());
}

template <class T1, class T2, class T3>
auto mergeVec(T1 First1, T1 Last1, T2 First2, T2 Last2, T3 DFirst) {
  for (; First1 != Last1; ++DFirst) {
    if (First2 == Last2) {
      return std::copy(First1, Last1, DFirst);
    }
    if (*First2 < *First1) {
      *DFirst = *(First2++);
    } else {
      *DFirst = *(First1++);
    }
  }
  return std::copy(First2, Last2, DFirst);
}

template <class T1, class T2>
auto mergeSort(T1 SourceBegin, T1 SourceEnd, T2 TargetBegin, T2 TargetEnd) {
  auto RangeLength = std::distance(SourceBegin, SourceEnd);
  if (RangeLength < 2) {
    return;
  }

  auto LeftChunkLength = RangeLength >> 1;
  auto SourceLeftChunkEnd = SourceBegin;
  auto TargetLeftChunkEnd = TargetBegin;

  std::advance(SourceLeftChunkEnd, LeftChunkLength);
  std::advance(TargetLeftChunkEnd, LeftChunkLength);

#ifdef MULTITHREAD
  std::vector<int> FirstHalf(SourceBegin, SourceLeftChunkEnd);
  std::vector<int> SecondHalf(SourceLeftChunkEnd, SourceEnd);

  std::thread Thread1(workerMerge, std::ref(FirstHalf));
  std::thread Thread2(workerMerge, std::ref(SecondHalf));

  Thread1.join();
  Thread2.join();

  // printVector(FirstHalf);
  // printVector(SecondHalf);

  mergeVec(FirstHalf.begin(), FirstHalf.end(), SecondHalf.begin(),
           SecondHalf.end(), TargetBegin);
#else
  mergeSort(TargetBegin, TargetLeftChunkEnd, SourceBegin, SourceLeftChunkEnd);
  mergeSort(TargetLeftChunkEnd, TargetEnd, SourceLeftChunkEnd, SourceEnd);
  mergeVec(SourceBegin, SourceLeftChunkEnd, SourceLeftChunkEnd, SourceEnd,
           TargetBegin);
#endif
}

template <class T> auto mergeSort(std::vector<T> &Vec) {
  auto Aux = Vec;
  mergeSort(Aux.begin(), Aux.end(), Vec.begin(), Vec.end());
}
